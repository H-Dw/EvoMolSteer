"""Explicit, once-only transfer of saved sampling and its existing reservations.

Old roots and claims are read-only. A new root records aliases, not additional
attempts, for the same seed/slot/configuration identities. No sampler is called.
"""
import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .absolute_coordinate import POLICY
from .geometry_batch import compile_units, split_products
from .geometry_calibration import plan as calibration_plan
from .geometry_protocol import EFFECTIVE
from .geometry_sampling import compatibility, execute_shard, reserve_group
from .io import Budget, frozen, read, sha, value_sha, write
from .native_tools import load_pt
from .prepare import load_model
from .protocol import SEEDS

RECOVERY_POLICY = "audited-absolute-saved-sampling-transfer-v1"
STATIC_REFACTOR_SOURCES = frozenset({"absolute_coordinate", "geometry_sampling", "geometry_batch"})


def assert_sampling_equivalence(previous, current, changes):
    old = {k: v for k, v in previous.items() if k != "scientific_sources"}
    new = {k: v for k, v in current.items() if k != "scientific_sources"}
    if old != new or previous.get("policy") != POLICY:
        raise ValueError("Recovery sampling settings, teacher or condition changed")
    left, right = previous["scientific_sources"], current["scientific_sources"]
    if set(left) != set(right):
        raise ValueError("Recovery sampling dependency set changed")
    actual = {k: dict(previous=left[k], current=right[k]) for k in left if left[k] != right[k]}
    if actual != changes or not set(actual) <= STATIC_REFACTOR_SOURCES:
        raise ValueError("Recovery has unaudited sampling source changes")


def candidate_key(identity):
    return value_sha(dict(schedule=identity[EFFECTIVE], slots=identity["logical_slots"]))


def reservation_aliases(previous, previous_manifests, units):
    """A bijection preserves every charged reservation, including pending ones."""
    current = {candidate_key(u["identity"]): u for u in units}
    if len(current) != len(units) or len(previous) != len(units):
        raise ValueError("Recovery reservation population differs")
    result = []
    for old_id, row in previous.items():
        identity = previous_manifests[old_id]["identity"]
        if row != dict(stage="calibration", count=10, binding=value_sha(identity)):
            raise ValueError("Original reservation binding or charge changed")
        key = candidate_key(identity)
        if key not in current:
            raise ValueError("Original candidate is absent from the declared new grid")
        unit = current.pop(key)
        result.append(dict(previous_id=old_id, current_id=unit["id"], count=10,
                           previous_binding=row["binding"], current_binding=unit["fingerprint"]))
    if current:
        raise ValueError("Recovery would introduce additional reservations")
    return result


@dataclass
class RecoveryPlan:
    authorization: dict
    subjects: dict
    batch: Path
    output: dict
    steps: list
    rng: list
    units: list
    aliases: list
    saved: list

    def summary(self):
        return dict(policy=RECOVERY_POLICY, source_root=self.authorization["source_root"],
            source_batch=str(self.batch), target_root=self.authorization["target_root"],
            reserved_attempts=sum(a["count"] for a in self.aliases),
            saved_candidates=len(self.output["coords"]),
            pending_reserved_candidates=sum(a["count"] for a in self.aliases)-len(self.output["coords"]),
            reservation_aliases=self.aliases, new_scientific_attempts=0,
            previous_failures_and_claims_preserved=True)


def build_plan(config):
    settings = config.get("sampling_recovery", {})
    if set(settings) != {"authorization_path", "authorization_sha256"}:
        raise ValueError("Explicit saved-sampling authorization required")
    path = Path(settings["authorization_path"])
    if sha(path) != settings["authorization_sha256"]:
        raise ValueError("Recovery authorization changed")
    authorization = read(path)
    root, source = Path(config["root"]), Path(authorization["source_root"])
    without_recovery = {k: v for k, v in config.items() if k != "sampling_recovery"}
    if (authorization["policy"] != RECOVERY_POLICY or config["protocol"] != POLICY
            or root.resolve() == source.resolve()
            or str(root) != authorization["target_root"]
            or value_sha(without_recovery) != authorization["target_config_sha256"]):
        raise ValueError("Recovery authorization belongs to another experiment")
    for filename, digest in authorization["source_artifact_hashes"].items():
        if sha(source/filename) != digest:
            raise ValueError("Saved recovery artifact changed: " + filename)
    if sha(source/"attempts.json") != authorization["source_attempts_sha256"]:
        raise ValueError("Original scientific attempt ledger changed")
    proof = Path(authorization["equivalence_evidence"])
    if sha(proof) != authorization["equivalence_evidence_sha256"]:
        raise ValueError("Numerical equivalence evidence changed")
    checks = read(proof)["teacher_vjp"]
    if len(checks) != 2 or not all(c["loss_bitwise_equal"] and c["full_coordinate_vjp_bitwise_equal"]
            and c["sampling_rng_unchanged"] and not c["affinity_gradient_entered"] for c in checks):
        raise ValueError("Both real-model coordinate VJPs must be exactly equivalent")
    before = read(source/"SUBJECTS.json")
    after = read(root/"SUBJECTS.json")
    if before != after:
        raise ValueError("Recovery teacher, molecular input or frozen condition differs")
    subjects = {s["teacher"]["target"]: s for s in after}
    if {c["teacher_id"]: c["condition_sha256"] for c in checks} != {
            name: s["condition"]["condition_sha256"] for name, s in subjects.items()}:
        raise ValueError("Numerical equivalence evidence belongs to other conditions")
    ledger = read(source/"attempts.json")
    manifests = {key: read(source/key/"manifest.json") for key in ledger}
    targets = {v["identity"]["compatibility"]["teacher_id"] for v in manifests.values()}
    units, aliases = [], []
    for target in sorted(targets):
        teacher, condition = (subjects[target][key] for key in ("teacher", "condition"))
        binding = compatibility(config, teacher, condition)
        key = value_sha([target, condition["condition_sha256"], value_sha(binding)])
        out = root/"calibration/frozen"/key
        jobs = [dict(schedule=[scale]*4, folder=out/"coarse"/value_sha(scale),
                     stage="calibration", count=10)
                for scale in calibration_plan(config, target)["coarse_actual_scales"]]
        current = compile_units(config, teacher, condition, jobs, _compatibility=binding)
        selected = {k: v for k, v in ledger.items()
                    if manifests[k]["identity"]["compatibility"]["teacher_id"] == target}
        for old_id in selected:
            assert_sampling_equivalence(manifests[old_id]["identity"]["compatibility"],
                                        binding, authorization["source_changes"])
        aliases.extend(reservation_aliases(selected, manifests, current))
        units.extend(current)
    batch = source/authorization["source_batch"]
    if not batch.resolve().is_relative_to(source.resolve()):
        raise ValueError("Saved batch is outside the authorized source root")
    failure = read(batch/"FAILED.json")
    if failure.get("type") != "NameError" or failure.get("error") != "name 'h5py' is not defined":
        raise ValueError("Only the audited postprocessing NameError may be recovered")
    batch_manifest = read(batch/"candidates_batch.json")
    steps, candidates = read(batch/"steps.json"), batch_manifest["candidates"]
    output = load_pt(batch/"terminal.pt.gz")
    count = len(output["coords"])
    if (len(steps) != 100 or [v["step"] for v in steps] != list(range(100))
            or any(v.get("failed_units") for v in steps)
            or count != len(candidates) or count % 10 or not torch.isfinite(output["coords"]).all()):
        raise ValueError("Saved batch does not contain complete healthy raw trajectories")
    with h5py.File(batch/"trace.h5", "r") as trace:
        assignment = trace["assignment"][10:100]
        if (not (assignment == assignment[0:1]).all()
                or not np.array_equal(trace["endpoint_A"][99], output["coords"].numpy())):
            raise ValueError("Saved endpoint or frozen assignment differs from the original trace")
    indexed = {a["previous_id"]: a["current_id"] for a in aliases}
    by_id = {u["id"]: u for u in units}
    saved = []
    for i in range(count//10):
        rows = candidates[i*10:(i+1)*10]
        old_id = rows[0]["source_group"]
        if old_id not in indexed or any(r["source_group"] != old_id for r in rows):
            raise ValueError("Saved batch candidate identities do not match original reservations")
        unit = by_id[indexed[old_id]]
        expected = [(seed, slot) for seed in SEEDS for slot in unit["slots"]]
        if ([(r["logical_seed"], r["logical_slot"]) for r in rows] != expected
                or any(r[EFFECTIVE] != unit["scale"][EFFECTIVE] for r in rows)):
            raise ValueError("Saved seed, slot or effective guidance scale changed")
        saved.append(unit)
    if ({u["identity"]["compatibility"]["teacher_id"] for u in saved} != {batch_manifest["teacher_id"]}
            or batch_manifest["condition_sha256"] != subjects[batch_manifest["teacher_id"]]["condition"]["condition_sha256"]):
        raise ValueError("Saved batch teacher or condition header differs")
    if len({u["id"] for u in saved}) != len(saved):
        raise ValueError("Saved candidates would be imported more than once")
    rng = read(batch/"rng_ledger.json")
    if len(rng) != len(saved):
        raise ValueError("Saved RNG shard ledger differs")
    result = RecoveryPlan(authorization, subjects, batch, output, steps, rng, units, aliases, saved)
    if (result.summary()["reserved_attempts"] != authorization["expected_reserved_attempts"]
            or count != authorization["expected_saved_candidates"]):
        raise ValueError("Saved population or original charged budget changed")
    return result


def preflight(config):
    """No claim, attempt reservation, molecule building, model forward or network request."""
    return build_plan(config).summary()


def run(config, model_loader=load_model):
    from .absolute_runtime import require_acceptance
    root = Path(config["root"])
    if (root/"RECOVERY_CLAIM.json").exists():
        raise RuntimeError("Saved-sampling transfer already claimed; do not retry automatically")
    require_acceptance(config)
    plan = build_plan(config)
    frozen(root/"RECOVERY_CLAIM.json", dict(authorization_sha256=config["sampling_recovery"]["authorization_sha256"],
        code_sha256=sha(__file__), no_automatic_retry=True, **plan.summary()))
    # The alias ledger carries the existing 130 charges. Pending units deliberately
    # have no execution claim yet; their first future sampling is not a refill.
    for unit in plan.units:
        Budget(root).reserve(unit["id"], "calibration", 10, unit["fingerprint"])
    frozen(root/"RECOVERY_RESERVATIONS.json", plan.summary())
    control = SimpleNamespace(units=plan.saved, rows=copy.deepcopy(plan.steps),
        random_streams=SimpleNamespace(parts=[SimpleNamespace(ledger=v) for v in plan.rng]))
    completed = []
    for target in sorted({u["identity"]["compatibility"]["teacher_id"] for u in plan.saved}):
        subject = plan.subjects[target]
        model = model_loader(config, subject["teacher"])
        for index, unit in enumerate(plan.saved):
            if unit["identity"]["compatibility"]["teacher_id"] != target:
                continue
            frozen(unit["folder"]/"manifest.json", dict(id=unit["id"], identity=unit["identity"]))
            reserve_group(root, unit["id"], "calibration", 10, unit["identity"], registry_name="slot_registry")
            output = split_products(plan.output, control, plan.batch, unit, index)
            provenance = unit["folder"]/"BATCH_PROVENANCE.json"
            write(provenance, dict(read(provenance), sampling_recovery=dict(policy=RECOVERY_POLICY,
                authorization_sha256=config["sampling_recovery"]["authorization_sha256"],
                original_sampling_commit=plan.authorization["source_commit"],
                sampling_was_not_reexecuted=True, reservation_alias=True)))
            execute_shard(model, config, subject["teacher"], subject["condition"], unit["job"]["schedule"],
                unit["folder"], "calibration", unit["slots"], _materialized=output,
                _reservation=unit["fingerprint"], _compatibility=unit["identity"]["compatibility"])
            completed.append(dict(id=unit["id"], receipt_sha256=sha(unit["folder"]/"COMPLETE.json")))
        del model
        torch.cuda.empty_cache()
    result = dict(plan.summary(), materialized_receipts=completed,
                  only_saved_postprocessing_executed=True, original_root_unchanged=True)
    frozen(root/"RECOVERY_COMPLETE.json", result)
    return result
