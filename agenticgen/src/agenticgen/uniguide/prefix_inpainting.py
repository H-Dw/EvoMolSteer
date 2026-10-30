"""Optional native FLOWR fragment conditioning confined to intervals [0, 10)."""

import types
import torch

from .absolute_coordinate import match, reference
from .chem_inpainting import project_chemistry

POLICY = "flowr-native-fragment-prefix-0-9-v1"


def policy(config):
    value = config.get("prefix_inpainting", {"enabled": False})
    if (
        not isinstance(value, dict)
        or set(value) != {"enabled"}
        or type(value["enabled"]) is not bool
    ):
        raise ValueError("prefix_inpainting accepts a boolean enabled field")
    return dict(
        policy=POLICY,
        enabled=value["enabled"],
        step_bounds=[0, 9],
        first_gradient_step=10,
        conditioned_atom_network_time=1.0,
        native_builder="inpaint_molecule",
        symmetrize=True,
        self_conditioning_replaced=False,
        terminal_projection=False,
    )


def native_times(model):
    """Scope native time inpainting to this call; do not activate native loop branches."""
    original = model._get_times
    if getattr(original, "_prefix_wrapped", False):
        return

    def get_times(this, times, lig_mask, inpaint_mask):
        control = getattr(this, "_adaptive", None)
        active = (
            control is not None
            and getattr(control, "prefix_policy", {}).get("enabled")
            and getattr(control, "active_step", 100) < 10
        )
        old = this.inpainting_mode
        try:
            if active:
                this.inpainting_mode = True
            packed = original(times, lig_mask, inpaint_mask)
            if active and inpaint_mask is not None:
                selected = inpaint_mask.bool() & lig_mask.bool()
                for index, channel in enumerate(("coordinates", "chemistry")):
                    values = packed[index].squeeze(-1)[selected]
                    if values.numel():
                        summary = control.prefix_time_observations.setdefault(
                            control.active_step, {}
                        ).setdefault(channel, dict(minimum=1.0, maximum=1.0, calls=0))
                        summary["minimum"] = min(
                            summary["minimum"], float(values.min())
                        )
                        summary["maximum"] = max(
                            summary["maximum"], float(values.max())
                        )
                        summary["calls"] += 1
            return packed
        finally:
            this.inpainting_mode = old

    get_times._prefix_wrapped = True
    model._get_times = types.MethodType(get_times, model)


class PrefixInpainting:
    prefix_policy = {"enabled": False}

    def configure_prefix(self):
        self.prefix_policy = policy(self.absolute_config)
        self.prefix_time_observations = {}
        if self.prefix_policy["enabled"]:
            if self.absolute_state is None:
                raise ValueError("Native prefix needs an absolute-coordinate condition")
            if self.model.inpainting_mode or self.model.graph_inpainting:
                raise ValueError("Other model-wide inpainting must be disabled")
            native_times(self.model)

    def project_prefix(self, state, ids):
        # Native builder mutates tensors: supply owned copies, preserving unselected state.
        template, _ = project_chemistry(state, ids, self.chemical_features)
        frame = self.geometry_condition["protein_alignment"]
        target = reference(self.geometry_condition, state["coords"])
        rotation = target.new_tensor(frame["rotation"])
        translation = target.new_tensor(frame["translation_angstrom"])
        target = (target - translation) @ rotation.T
        target = (target[None] - self.com[:, None]) / self.model.coord_scale
        batch = torch.arange(len(ids), device=ids.device)[:, None]
        template["coords"] = state["coords"].clone()
        template["coords"][batch, ids] = target
        fragment = torch.zeros_like(state["mask"], dtype=torch.bool)
        fragment[batch, ids] = True
        template.update(fragment_mask=fragment, fragment_mode=["fragment"] * len(ids))
        prediction = {
            k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()
        }
        result = self.model.builder.inpaint_molecule(
            data=template,
            prediction=prediction,
            keep_interactions=getattr(self.model, "flow_interactions", False),
            symmetrize=True,
        )
        return dict(state, **result)

    def prepare_prefix_forward(self, current, step):
        if not self.prefix_policy["enabled"]:
            return
        if step == 0 and not self.resume:
            endpoint = current["coords"] * self.model.coord_scale + self.com[:, None]
            ids = match(endpoint, self.geometry_condition, current["mask"].bool())
            current.update(self.project_prefix(current, ids))
        if step >= 10:
            # Removing the mask also disables native fixed-position/node embeddings.
            current.pop("fragment_mask", None)
            current.pop("fragment_mode", None)

    def apply_prefix_inpainting(self, current, step):
        active = self.prefix_policy["enabled"] and 0 <= step < 10
        self.rows[-1]["prefix_joint_inpainting_applied"] = [active] * len(
            current["coords"]
        )
        if not active:
            return current
        result = self.project_prefix(current, self.last_assignment)
        self.store(
            "prefix_post_coordinates_A",
            step,
            result["coords"] * self.model.coord_scale + self.com[:, None],
        )
        self.store("prefix_fragment_mask", step, result["fragment_mask"])
        return result


def verify_trace(folder, config, condition):
    """Check actual native prefix times, projection, and the complete cutoff."""
    from pathlib import Path
    import h5py
    import numpy as np
    from .io import read
    from .absolute_coordinate import aligned

    if not policy(config)["enabled"]:
        return dict(enabled=False)
    folder = Path(folder)
    output = read(folder / "OUTPUT_POLICY.json")
    times = output["prefix_native_times"]
    if set(times) != {str(i) for i in range(10)} or any(
        row[channel]["minimum"] != 1.0 or row[channel]["maximum"] != 1.0
        for row in times.values()
        for channel in ("coordinates", "chemistry")
    ):
        raise ValueError(
            "Native prefix network times do not match FLOWR fragment conditioning"
        )
    steps = read(folder / "steps.json")
    if any(
        any(r["prefix_joint_inpainting_applied"]) != (r["step"] < 10) for r in steps
    ):
        raise ValueError("Coordinate inpainting crossed the prefix cutoff")
    with h5py.File(folder / "trace.h5") as trace:
        points = torch.from_numpy(trace["prefix_post_coordinates_A"][:10])
        ids = torch.from_numpy(trace["assignment"][:10]).long()
        points = aligned(points, condition).gather(
            2, ids[..., None].expand(-1, -1, -1, 3)
        )
        target = reference(condition, points)
        if not np.allclose(points.numpy(), target.numpy(), rtol=0, atol=2e-5):
            raise ValueError("Native prefix coordinates differ from the condition")
        if trace["prefix_fragment_mask"][10:].any():
            raise ValueError("Native prefix fixed-position mask remains after step 9")
    return dict(
        enabled=True,
        native_network_times_equal_one=True,
        coordinate_and_chemical_projection_steps=list(range(10)),
        no_coordinate_projection_after_step_9=True,
        native_builder_used=True,
    )
