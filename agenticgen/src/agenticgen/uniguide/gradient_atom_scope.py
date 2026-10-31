"""Mask only the added coordinate-guidance VJP; preserve native FLOWR motion."""
import torch

KEY = "guidance_atom_scope"
POLICY = "endpoint-bond-hop-guidance-mask-v1"


def policy(config):
    value = config.get(KEY, {"mode": "all"})
    if value == {"mode": "all"}:
        return {"policy": POLICY, "mode": "all", "native_motion_masked": False}
    if (not isinstance(value, dict) or set(value) != {"mode", "hops"}
            or value["mode"] != "condition_hops" or type(value["hops"]) is not int
            or value["hops"] not in (0, 1, 2)):
        raise ValueError("guidance_atom_scope requires all or condition_hops with hops 0/1/2")
    return {"policy": POLICY, **value,
            "graph": "same_step_predicted_endpoint_bonds_argmax_nonzero_undirected_union",
            "seeds": "same_step_loss_hungarian_assignment", "native_motion_masked": False,
            "loss_region_expanded": False, "mask_after_full_input_vjp": True}


def atom_mask(valid, assignment, bonds, hops):
    if type(hops) is not int or hops not in (0, 1, 2):
        raise ValueError("Only 0, 1 and 2 bond hops are supported")
    valid = valid.bool()
    if valid.ndim != 2 or assignment.ndim != 2 or len(assignment) != len(valid):
        raise ValueError("Candidate mask and condition assignment shapes differ")
    ids = assignment.to(device=valid.device)
    if ids.dtype != torch.long or not ids.shape[1]:
        raise ValueError("Condition assignment requires nonempty integer slots")
    if bool(((ids < 0) | (ids >= valid.shape[1])).any()):
        raise ValueError("Condition assignment outside generated slots")
    if ids.shape[1] > 1 and bool((ids.sort(dim=1).values.diff(dim=1) == 0).any()):
        raise ValueError("Repeated condition slot")
    batch = torch.arange(len(valid), device=valid.device)[:, None]
    if not bool(valid[batch, ids].all()):
        raise ValueError("Condition assignment selects padding")
    selected = torch.zeros_like(valid)
    selected[batch, ids] = True
    if hops:
        n = valid.shape[1]
        if bonds is None or bonds.shape != (len(valid), n, n, 5):
            raise ValueError("Hop expansion requires native five-class endpoint bonds")
        classes = bonds.detach().argmax(-1)
        adjacency = (classes != 0) | (classes.transpose(1, 2) != 0)
        adjacency &= valid[:, :, None] & valid[:, None, :]
        adjacency &= ~torch.eye(n, dtype=torch.bool, device=valid.device)[None]
        for _ in range(hops):
            selected = selected | (adjacency & selected[:, :, None]).any(dim=1)
    return selected & valid


def apply(gradient, valid, assignment, bonds, settings):
    if settings["mode"] == "all":
        return gradient, None  # Preserve the old path without even a multiplication.
    mask = atom_mask(valid, assignment, bonds, settings["hops"])
    if gradient.shape != (*valid.shape, 3):
        raise ValueError("Coordinate VJP shape differs from the generated atom mask")
    return torch.where(mask[..., None], gradient, torch.zeros_like(gradient)), mask
