"""Teacher-local candidate scheduler; physical capacity is not a scientific setting.

Ten-row promotion shards remain the immutable accounting/storage unit. Multiple
shards and directions share one native integration call, with row-local scales
and independent native RNG replay. Only a confirmed device OOM retries capacity.
"""

import copy
import functools
import gc
import math
import time
import threading
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
)
from .geometry_reward import WEIGHTS
from .geometry_safety import CandidateSafety
from .geometry_protocol import EFFECTIVE, MULTIPLIER, expand_effective, validate_domain
from .io import frozen, read, write, value_sha, sha
from .native_tools import save_pt, tensor_digest
from .random_streams import capture, restore
from .protocol import SEEDS
from .geometry_dynamics import STEP_FIELDS

CAPACITIES = (100, 80, 60, 40, 20, 10)
EXECUTION_POLICY = "teacher-candidates-concurrent-fixed-shape-v2"


def lower_capacity(current):
    return next((n for n in CAPACITIES if n < current), None)


def device_oom(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError)
        and any(
            s in str(error).lower()
            for s in (
                "cuda out of memory",
                "hip out of memory",
                "cuda error: out of memory",
                "hip error: out of memory",
            )
        )
    )


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
        self.failure_lock = threading.RLock()
        self.expanded = [expand_effective(u["scale"][EFFECTIVE]) for u in units]
        super().__init__(*args, **kwargs)

    def make_streams(self, states):
        return CandidateStreams(states, self.initial["candidate_slots"])

    def physical_rows(self):
        return 10

    def execute_blocks(self, function, starts):
        from .chunking import concurrent_map

        return concurrent_map(function, starts, max(1, self.capacity // 10))

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
        return [self.units[i // 10]["scale"][WEIGHTS] for i in ids]

    def initialize(self, *args):
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
        step["scale"] = expand_effective(unit["scale"][EFFECTIVE])[step["step"]]
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


def compile_units(config, teacher, condition, jobs):
    root = Path(config["root"])
    binding = compatibility(config, teacher, condition)
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
            if any(
                not domain["normalized_minimum"] <= x <= domain["normalized_maximum"]
                for x in schedule
            ):
                raise ValueError("Outside measured domain")
        scale = scale_binding(
            schedule,
            domain[MULTIPLIER] if domain else None,
            job.get("enabled", True),
            condition,
            job.get("loss_weights"),
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
        concurrent_network_units=len(units),
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


def execute_many(model, config, teacher, condition, jobs, *, sample_fn=None):
    """Return one aggregate or FloatingPointError per job; never rerun numerical failures."""
    sample_fn = sample_fn or sampler.sample
    root = Path(config["root"])
    units = compile_units(config, teacher, condition, jobs)
    batch_root = root / "candidate_batches" / teacher["target"]
    batch_root.mkdir(parents=True, exist_ok=True)
    capacity_file = batch_root / "capacity.json"
    device_binding = dict(
        device=str(torch.cuda.get_device_name()),
        torch=str(torch.__version__),
        policy=EXECUTION_POLICY,
    )
    prior = read(capacity_file) if capacity_file.exists() else {}
    capacity = (
        prior.get("capacity", 100)
        if prior.get("device_binding") == device_binding
        else 100
    )
    if capacity not in CAPACITIES:
        raise ValueError("Corrupt teacher capacity")
    pending = list(units)
    failed = {}
    ordinal = 0
    # Reserve candidates once, before any inference; OOM retries never reserve again.
    for unit in units:
        frozen(
            unit["folder"] / "manifest.json",
            dict(id=unit["id"], identity=unit["identity"]),
        )
        reserve_group(
            root,
            unit["id"],
            unit["job"]["stage"],
            10,
            unit["identity"],
            registry_name="slot_registry",
        )
    while pending:
        selected = pending[: capacity // 10]
        size = 10 * len(selected)
        ordinal += 1
        attempt = batch_root / (str(time.time_ns()) + f"_{ordinal:03d}")
        attempt.mkdir()
        frozen(
            attempt / "candidates_batch.json",
            candidate_manifest(teacher, condition, selected),
        )
        external = capture()
        guard = copy.deepcopy(model.integrator._guard_warned)
        previous_rows = getattr(model, "_ug_physical_rows", 10)
        previous_workers = getattr(model, "_ug_parallel_workers", 1)
        oom = None
        fatal = None
        output = None
        control = None
        product = None
        try:
            model._ug_physical_rows = 10
            model._ug_parallel_workers = size // 10
            initial = batch_initial(config, teacher["target"], selected)
            factory = functools.partial(
                CandidateControl,
                condition=condition,
                scale_binding=selected[0]["scale"],
                units=selected,
                physical_rows=size,
            )
            sampled = sample_fn(
                model,
                config,
                teacher,
                selected[0]["scale"][EFFECTIVE],
                attempt,
                size,
                control_factory=factory,
                check_affinity=False,
                initial_override=initial,
            )
            output, control = sampled[0], sampled[3]
            del sampled  # Do not retain a previous batch's GPU pocket tensors during the next OOM probe.
            for i, unit in enumerate(selected):
                if i in control.failed_units:
                    failed[unit["id"]] = control.failed_units[i]
                    write(
                        unit["folder"] / "FAILED.json",
                        dict(
                            **control.failed_units[i],
                            attempts=10,
                            source_batch=str(attempt),
                            failed_attempts_remain_charged=True,
                            no_weaker_retry=True,
                        ),
                    )
                    continue
                product = split_products(output, control, attempt, unit, i)
                job = unit["job"]
                execute_shard(
                    model,
                    config,
                    teacher,
                    condition,
                    job["schedule"],
                    unit["folder"],
                    job["stage"],
                    unit["slots"],
                    job.get("domain"),
                    job.get("enabled", True),
                    _materialized=product,
                    _reservation=unit["fingerprint"],
                    loss_weights=job.get("loss_weights"),
                )
            frozen(
                attempt / "COMPLETE.json",
                dict(
                    candidate_count=size,
                    failed_units=control.failed_units,
                    capacity=size,
                    independent_validation=False,
                ),
            )
        except Exception as error:
            if device_oom(error):
                oom = dict(type=type(error).__name__, error=str(error))
            else:
                fatal = error
        finally:
            restore(external)
            model.integrator._guard_warned = guard
            model._ug_physical_rows = previous_rows
            model._ug_parallel_workers = previous_workers
            model._adaptive = None
            model.integrator._adaptive_observer = None
            output = None
            control = None
            product = None
        if fatal is not None:
            write(
                attempt / "FAILED.json",
                dict(
                    type=type(fatal).__name__, error=str(fatal), no_automatic_retry=True
                ),
            )
            raise fatal
        if oom is not None:
            # No completed products may be overwritten by a retry after postprocessing.
            if any((u["folder"] / "COMPLETE.json").exists() for u in selected):
                raise RuntimeError(
                    "OOM after partial materialization; preserve results and audit postprocessing"
                )
            next_size = lower_capacity(size)
            write(
                attempt / "OOM.json",
                dict(
                    **oom,
                    attempted_capacity=size,
                    next_capacity=next_size,
                    candidates_unchanged=True,
                    random_stream_replayed=True,
                    new_scientific_attempts=0,
                ),
            )
            gc.collect()
            torch.cuda.empty_cache()
            if next_size is None:
                raise RuntimeError(
                    "Teacher candidate inference exhausted OOM capacity ladder"
                )
            capacity = next_size
            write(
                capacity_file,
                dict(
                    capacity=capacity,
                    device_binding=device_binding,
                    reason="confirmed_device_oom",
                ),
            )
            continue
        pending = pending[len(selected) :]
    results = []
    binding = compatibility(config, teacher, condition)
    for job in jobs:
        scale = scale_binding(
            job["schedule"],
            job["domain"][MULTIPLIER] if job.get("domain") else None,
            job.get("enabled", True),
            condition,
            job.get("loss_weights"),
        )
        affected = [
            u
            for u in units
            if u["id"] in failed
            and candidate_identity(u["scale"]) == candidate_identity(scale)
            and u["slots"][1] < job.get("count", 50) // 5
        ]
        if affected:
            error = FloatingPointError(
                "Nonfinite candidate units: " + ", ".join(u["id"] for u in affected)
            )
            identity = dict(
                compatibility=binding,
                **candidate_identity(scale),
                population=job.get("count", 50),
                logical_slots_per_seed=job.get("count", 50) // 5,
            )
            run_id = Path(job["folder"]).relative_to(root).as_posix()
            frozen(
                Path(job["folder"]) / "manifest.json",
                dict(id=run_id, identity=identity),
            )
            reserve_group(
                root, run_id, job["stage"], job.get("count", 50), identity, charge=False
            )
            write(
                Path(job["folder"]) / "FAILED.json",
                dict(
                    error=str(error),
                    source_units=[u["id"] for u in affected],
                    no_weaker_retry=True,
                ),
            )
            results.append(error)
        else:
            results.append(
                execute(
                    model,
                    config,
                    teacher,
                    condition,
                    job["schedule"],
                    job["folder"],
                    job["stage"],
                    job.get("count", 50),
                    job.get("domain"),
                    job.get("enabled", True),
                    loss_weights=job.get("loss_weights"),
                )
            )
    return results
