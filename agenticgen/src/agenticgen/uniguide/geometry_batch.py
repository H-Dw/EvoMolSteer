"""UG04 scheduling: one teacher, one configuration, serial ten-row network calls.

Promotion shards retain candidate identities and measurements. Different
configurations never share an integration call; failures are never retried.
"""
from .guidance_windows import expand as expand_windows

import copy
import functools
import math
import time
from pathlib import Path
import h5py
import torch
from . import sampler
from .batching import take
from .chunking import concatenate
from .geometry_shards import SlotStreams, initial_shard
from .geometry_sampling import (
    GeometryControl,
    compatibility,
    scale_binding,
    reserve_group,
    execute_shard,
    execute,
    validate_receipt,
    candidate_identity,
    measurement_binding,
)
from .geometry_reward import WEIGHTS
from .geometry_safety import CandidateSafety
from .geometry_protocol import EFFECTIVE, MULTIPLIER, expand_effective, validate_domain, sampling_reference, validate_sampling_schedule
from .io import frozen, read, write, value_sha, sha
from .native_tools import save_pt, tensor_digest
from .random_streams import capture, restore
from .protocol import SEEDS
from .geometry_dynamics import STEP_FIELDS
from .task_scheduling import POLICY as EXECUTION_POLICY

DEFAULT_BATCH_SIZE = 50


class CandidateStreams:
    def __init__(self, states, slots):
        if len(states) != len(slots) or not states:
            raise ValueError("Incomplete candidate RNG layout")
        self.parts = [SlotStreams(state, slot) for state, slot in zip(states, slots)]

    @property
    def states(self):
        return [s.states for s in self.parts]

    @property
    def ledger(self):
        return [s.ledger for s in self.parts]

    def categorical(self, probabilities):
        if len(probabilities) != 10 * len(self.parts):
            raise ValueError("Candidate RNG row mismatch")
        return torch.cat(
            [
                s.categorical(probabilities[i * 10 : (i + 1) * 10])
                for i, s in enumerate(self.parts)
            ]
        )

    def normal(self, name, *args, **kwargs):
        if name == "randn_like":
            if len(args) != 1 or len(args[0]) != 10 * len(self.parts):
                raise ValueError("Candidate Gaussian row mismatch")
            return torch.cat(
                [
                    s.normal(name, args[0][i * 10 : (i + 1) * 10], **kwargs)
                    for i, s in enumerate(self.parts)
                ]
            )
        shape = (
            tuple(args[0])
            if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size))
            else tuple(args)
        )
        if name != "randn" or shape[0] != 10 * len(self.parts):
            raise ValueError("Unexpected candidate Gaussian call")
        return torch.cat(
            [s.normal(name, (10, *shape[1:]), **kwargs) for s in self.parts]
        )


def batch_initial(config, target, units):
    parts = [initial_shard(config, target, u["slots"]) for u in units]
    combined = {
        k: concatenate([p[k] for p in parts], [10] * len(parts))
        for k in ("prior", "pocket_target", "pocket_off", "target_com")
    }
    return dict(
        combined,
        population=10 * len(parts),
        stream_states=[p["stream_states"] for p in parts],
        candidate_slots=[u["slots"] for u in units],
    )


class CandidateControl(CandidateSafety, GeometryControl):
    def __init__(self, *args, units, physical_rows, **kwargs):
        self.units = units
        self.capacity = physical_rows
        self.failed_units = {}
        self.expanded = [expand_windows(u["scale"][EFFECTIVE],u["scale"]) for u in units]
        super().__init__(*args, **kwargs)

    def make_streams(self, states):
        return CandidateStreams(states, self.initial["candidate_slots"])

    def physical_rows(self):
        return 10

    def execute_blocks(self, function, starts, *, inputs=()):
        return [function(start) for start in starts]

    def step_scales(self, step, count):
        if count != 10 * len(self.units):
            raise ValueError("Candidate scale layout differs")
        return [
            values[step] if i not in self.failed_units else 0.0
            for i, values in enumerate(self.expanded)
            for _ in range(10)
        ]

    def checkpoint_metadata(self):
        return dict(
            **self.assignment_checkpoint(),
            candidate_binding=dict(
                condition_sha256=self.geometry_condition["condition_sha256"],
                units=[
                    dict(id=u["id"], slots=u["slots"], scale=u["scale"])
                    for u in self.units
                ],
                failed_units=self.failed_units,
                sampling_policy=self.sampling_policy,
                chemical_binding=getattr(self, "chemical_binding", None),
            ),
            **self.raw_output_checkpoint(),
        )

    def row_loss_weights(self, ids):
        return [self.units[i // 10]["scale"].get(WEIGHTS) for i in ids]

    def initialize(self, *args):
        self.restore_assignments()
        self.restore_raw_output()
        if self.resume:
            bound = self.resume["candidate_binding"]
            self.failed_units = {int(k): v for k, v in bound["failed_units"].items()}
            if bound != self.checkpoint_metadata()["candidate_binding"]:
                raise ValueError("Candidate resume binding changed")
        # GeometryControl checks the single-population metadata; this is a distinct layout.
        return sampler.Control.initialize(self, *args)

    def after(self, curr, step):
        result = sampler.Control.after(self, curr, step)
        self.rows[-1]["scale"] = self.step_scales(step, self.initial["population"])
        self.rows[-1]["candidate_rng_after_sha256"] = [
            tensor_digest(s.states) for s in self.random_streams.parts
        ]
        self.rows[-1]["failed_units"] = copy.deepcopy(self.failed_units)
        if self.absolute_state is not None:
            if not self.absolute_state.dynamic and step > 10 and any(self.rows[-1]["assignment_switch_fraction"]):
                raise ValueError("Frozen candidate assignment changed")
            self.absolute_state.save_cache()
        result = self.apply_prefix_inpainting(result, step)
        return self.apply_chemical_inpainting(result, step)

    def repair_diagnostics(self, key, fallback=None):
        for i, value in enumerate(self.rows[-1][key]):
            if math.isfinite(value):
                continue
            corrected = float(fallback[i]) if fallback is not None else None
            self.rows[-1][key][i] = (
                corrected
                if corrected is not None and math.isfinite(corrected)
                else None
            )
            self.rows[-1].setdefault("diagnostic_overflow", []).append(
                dict(
                    field=key,
                    row=i,
                    recovery=(
                        "float64_reduction"
                        if self.rows[-1][key][i] is not None
                        else "unavailable_nonfinite_source"
                    ),
                )
            )

    def normalize_forward_diagnostics(self, gradient):
        if any(not math.isfinite(x) for x in self.rows[-1]["gradient_norm"]):
            self.repair_diagnostics(
                "gradient_norm", gradient.double().flatten(1).norm(dim=1)
            )
        self.repair_diagnostics("coefficient")

    def normalize_step_diagnostics(self, native):
        # Do not classify a finite 1e30 displacement as NaN merely because a
        # float32 sum of squares overflowed. Sampling tensors stay unchanged.
        if any(not math.isfinite(x) for x in self.rows[-1]["max_atom_increment_A"]):
            self.repair_diagnostics(
                "max_atom_increment_A",
                (self.delta.double() * self.model.coord_scale).norm(dim=-1).amax(1),
            )
        if any(not math.isfinite(x) for x in self.rows[-1]["delta_native_ratio"]):
            self.repair_diagnostics(
                "delta_native_ratio",
                self.delta.double().flatten(1).norm(dim=1)
                / (native.double() - self.before.double())
                .flatten(1)
                .norm(dim=1)
                .clamp_min(1e-12),
            )

    def check_terminal(self, output):
        result = self.prepare_output(output, list(range(self.initial["population"])))
        value, *_ = self.condition_loss_for_rows(
            result["coords"],
            result["mask"].bool(),
            list(range(self.initial["population"])),
        )
        self.check_rows("terminal internal distance loss", value)
        return result


def split_products(output, control, source, unit, index):
    destination = unit["folder"]
    indices = list(range(index * 10, (index + 1) * 10))
    count = 10 * len(control.units)
    result = take(output, indices, count)
    save_pt(destination / "terminal.pt.gz", result)
    with (
        h5py.File(source / "trace.h5", "r") as before,
        h5py.File(destination / "trace.h5", "w") as after,
    ):
        for name in before:
            data = before[name]
            shape = (100, 10, *data.shape[2:])
            target = after.create_dataset(
                name,
                shape=shape,
                dtype=data.dtype,
                chunks=(1, *shape[1:]),
                compression="gzip",
                compression_opts=1,
            )
            for step in range(100):
                target[step] = data[step, indices]
    steps = copy.deepcopy(control.rows)
    for step in steps:
        for key in (
            "loss",
            "assignment_switch_fraction",
            "gradient_norm",
            "coefficient",
            "max_atom_increment_A",
            "delta_native_ratio",
        ):
            step[key] = step[key][index * 10 : (index + 1) * 10]
        for key in STEP_FIELDS:
            if key in step:
                step[key] = step[key][index * 10 : (index + 1) * 10]
        step["scale"] = expand_windows(unit["scale"][EFFECTIVE],unit["scale"])[step["step"]]
        step["sampling_rng_after_sha256"] = step.pop("candidate_rng_after_sha256")[
            index
        ]
        step.pop("failed_units", None)
        step.update(unit["scale"])
        if "diagnostic_overflow" in step:
            step["diagnostic_overflow"] = [
                dict(item, row=item["row"] - index * 10)
                for item in step["diagnostic_overflow"]
                if index * 10 <= item["row"] < (index + 1) * 10
            ]
    write(destination / "steps.json", steps)
    write(destination / "rng_ledger.json", control.random_streams.parts[index].ledger)
    if (source / "OUTPUT_POLICY.json").exists():
        write(destination / "OUTPUT_POLICY.json", read(source / "OUTPUT_POLICY.json"))
    if (source / "CHEMICAL_CONDITION.json").exists():
        frozen(
            destination / "CHEMICAL_CONDITION.json",
            read(source / "CHEMICAL_CONDITION.json"),
        )
    frozen(
        destination / "BATCH_PROVENANCE.json",
        dict(
            batch_directory=str(source),
            row_indices=indices,
            candidate_batch_sha256=sha(source / "candidates_batch.json"),
            execution_policy=EXECUTION_POLICY,
        ),
    )
    return result


def compile_units(config, teacher, condition, jobs, *, _compatibility=None):
    root = Path(config["root"])
    if _compatibility is None:
        _compatibility = compatibility(config, teacher, condition)
    binding = measurement_binding(config, teacher, condition, _compatibility)
    units = {}
    for job in jobs:
        count = job.get("count", 50)
        domain = job.get("domain")
        schedule = job["schedule"]
        if count not in (10, 20, 50):
            raise ValueError("Unsupported candidate population")
        if domain is None and job["stage"] != "calibration":
            raise ValueError("A frozen multiplier is required")
        if domain:
            validate_domain(
                domain,
                teacher["target"],
                condition["condition_sha256"],
                value_sha(binding),
            )
            validate_sampling_schedule(schedule, domain)
        scale = scale_binding(
            schedule,
            sampling_reference(domain),
            job.get("enabled", True),
            condition,
            job.get("loss_weights"),
            window_bounds=config.get("windows"),
        )
        for start in range(0, count // 5, 2):
            slots = [start, start + 1]
            key = value_sha(
                dict(compatibility=binding, **candidate_identity(scale), slots=slots)
            )
            folder = root / "sample_shards" / key
            run_id = folder.relative_to(root).as_posix()
            identity = dict(
                compatibility=binding,
                **candidate_identity(scale),
                population=10,
                logical_slots=slots,
            )
            fingerprint = value_sha(identity)
            entry = root / "slot_registry" / (fingerprint + ".json")
            if entry.exists():
                saved = read(entry)
                if saved != dict(source_group=run_id, identity=identity):
                    raise ValueError("Candidate registry identity differs")
                if (folder / "COMPLETE.json").exists():
                    validate_receipt(folder)
                    continue
                raise RuntimeError(
                    "Existing failed/interrupted candidate requires explicit audit: "
                    + run_id
                )
            units.setdefault(
                key,
                dict(
                    id=run_id,
                    folder=folder,
                    slots=slots,
                    scale=scale,
                    identity=identity,
                    fingerprint=fingerprint,
                    job=job,
                ),
            )
    return list(units.values())


def candidate_manifest(teacher, condition, units):
    return dict(
        policy=EXECUTION_POLICY,
        teacher_id=teacher["target"],
        condition_sha256=condition["condition_sha256"],
        network_microbatch_size=10,
        candidate_batch_size=10 * len(units),
        concurrent_network_units=1,
        configuration_groups_in_batch=1,
        candidates=[
            dict(
                candidate_id=u["fingerprint"] + f"/{seed}/{slot}",
                source_group=u["id"],
                logical_seed=seed,
                logical_slot=slot,
                **u["scale"],
            )
            for u in units
            for seed in SEEDS
            for slot in u["slots"]
        ],
    )


def _execute_job(model, config, teacher, condition, job, binding, sample_fn):
    """Finish a single configuration before reserving or sampling the next one."""
    root = Path(config["root"])
    units = compile_units(config, teacher, condition, [job], _compatibility=binding)
    failed = {}
    if units:
        size = 10 * len(units)
        attempt = root / "candidate_batches" / teacher["target"] / str(time.time_ns())
        attempt.mkdir(parents=True)
        frozen(attempt / "candidates_batch.json", candidate_manifest(teacher, condition, units))
        for unit in units:
            frozen(unit["folder"] / "manifest.json", dict(id=unit["id"], identity=unit["identity"]))
            reserve_group(root, unit["id"], job["stage"], 10, unit["identity"],
                          registry_name="slot_registry")
        external = capture()
        guard = copy.deepcopy(model.integrator._guard_warned)
        phase = "sampling"
        output = control = product = None
        try:
            factory = functools.partial(CandidateControl, condition=condition,
                scale_binding=units[0]["scale"], units=units, physical_rows=size)
            sampled = sample_fn(model, config, teacher, units[0]["scale"][EFFECTIVE],
                attempt, size, control_factory=factory, check_affinity=False,
                initial_override=batch_initial(config, teacher["target"], units))
            output, control = sampled[0], sampled[3]
            del sampled
            phase = "postprocessing"
            for index, unit in enumerate(units):
                if index in control.failed_units:
                    failed[unit["id"]] = control.failed_units[index]
                    write(unit["folder"] / "FAILED.json", dict(**control.failed_units[index],
                        attempts=10, source_batch=str(attempt),
                        failed_attempts_remain_charged=True, no_weaker_retry=True))
                    continue
                product = split_products(output, control, attempt, unit, index)
                execute_shard(model, config, teacher, condition, job["schedule"],
                    unit["folder"], job["stage"], unit["slots"], job.get("domain"),
                    job.get("enabled", True), _materialized=product,
                    _reservation=unit["fingerprint"], loss_weights=job.get("loss_weights"),
                    _compatibility=binding)
            frozen(attempt / "COMPLETE.json", dict(candidate_count=size,
                failed_units=control.failed_units, independent_validation=False,
                scheduling_policy=EXECUTION_POLICY))
        except Exception as error:
            write(attempt / "FAILED.json", dict(type=type(error).__name__, error=str(error),
                phase=phase, terminal_saved=(attempt / "terminal.pt.gz").exists(),
                no_automatic_retry=True, no_capacity_fallback=True))
            raise
        finally:
            restore(external)
            model.integrator._guard_warned = guard
            model._adaptive = None
            model.integrator._adaptive_observer = None
            output = control = product = None
    if failed:
        error = FloatingPointError("Nonfinite candidate units: " + ", ".join(failed))
        scale = scale_binding(job["schedule"], sampling_reference(job.get("domain")),
                              job.get("enabled", True), condition, job.get("loss_weights"), window_bounds=config.get("windows"))
        identity = dict(compatibility=binding, **candidate_identity(scale),
                        population=job.get("count", 50), logical_slots_per_seed=job.get("count", 50) // 5)
        run_id = Path(job["folder"]).relative_to(root).as_posix()
        frozen(Path(job["folder"]) / "manifest.json", dict(id=run_id, identity=identity))
        reserve_group(root, run_id, job["stage"], job.get("count", 50), identity, charge=False)
        write(Path(job["folder"]) / "FAILED.json", dict(error=str(error),
              source_units=list(failed), no_weaker_retry=True))
        if job["stage"] == "formal":
            raise error
        return error
    return execute(model, config, teacher, condition, job["schedule"], job["folder"],
        job["stage"], job.get("count", 50), job.get("domain"), job.get("enabled", True),
        loss_weights=job.get("loss_weights"), _compatibility=binding)


def execute_many(model, config, teacher, condition, jobs, *, sample_fn=None):
    """UG04 ordering: complete each configuration on the caller thread, without retries."""
    from .task_scheduling import configure
    config = configure(config)
    binding = compatibility(config, teacher, condition)
    results = []
    for job in jobs:
        results.append(_execute_job(model, config, teacher, condition, job, binding,
                                    sample_fn or sampler.sample))
    return results
