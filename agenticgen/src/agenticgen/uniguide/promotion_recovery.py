"""Audited import of completed raw shards after an output-policy merge defect.

This creates explicitly attributed measurement views in a fresh experiment.
It never opens a model, integrates a candidate, or retries an LLM request.
"""

import ast
import copy
import shutil
from pathlib import Path

from .io import read, write, frozen, sha, value_sha, Budget
from .geometry_sampling import compatibility, validate_receipt, candidate_identity
from .linear_calibration import ensure_limits
from .protocol import SEEDS
from .geometry_shards import validate_slots

POLICY = "completed-shard-import-after-promotion-merge-fix-v1"
METADATA = {"manifest.json", "aggregate.json"}


def verify_module_equivalence(source, destination):
    """Only assembly is allowed to differ in the one changed sampling module."""
    before = ast.parse(Path(source).read_text(encoding="utf-8"))
    after = ast.parse(Path(destination).read_text(encoding="utf-8"))
    old = {
        n.name: ast.dump(n, include_attributes=False)
        for n in before.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    new = {
        n.name: ast.dump(n, include_attributes=False)
        for n in after.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    if set(new) - set(old) != {"merge_output_policies"} or set(old) - set(new):
        raise ValueError("Unexpected shard module additions or removals")
    for name in set(old) - {"combine_products"}:
        if old[name] != new[name]:
            raise ValueError(
                "Sampling or random-stream implementation changed: " + name
            )
    imports = lambda t: [
        ast.dump(n, include_attributes=False)
        for n in t.body
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef))
    ]
    if imports(before) != imports(after):
        raise ValueError("Shard module imports or constants changed")
    return dict(
        allowed_changed_function="combine_products",
        added_helper="merge_output_policies",
        unchanged_sampling_definitions=sorted(set(old) - {"combine_products"}),
        source_sha256=sha(source),
        destination_sha256=sha(destination),
    )


def verify_compatibility(old, new, source_package, destination_package):
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    changed = [
        k
        for k in set(left["scientific_sources"]) | set(right["scientific_sources"])
        if left["scientific_sources"].get(k) != right["scientific_sources"].get(k)
    ]
    if changed != ["geometry_shards"]:
        raise ValueError(
            "Recovery requires exactly one assembly-only source change: "
            + repr(changed)
        )
    proof = verify_module_equivalence(
        Path(source_package) / "geometry_shards.py",
        Path(destination_package) / "geometry_shards.py",
    )
    if (
        left["scientific_sources"]["geometry_shards"] != proof["source_sha256"]
        or right["scientific_sources"]["geometry_shards"] != proof["destination_sha256"]
    ):
        raise ValueError("Assembly proof is not bound to the measured source")
    left["scientific_sources"].pop("geometry_shards")
    right["scientific_sources"].pop("geometry_shards")
    if left != right:
        raise ValueError(
            "Condition, model, RNG, time grid, loss, windows or statistics differ"
        )
    return proof


def source_items(source, target, new_binding, source_package):
    source = Path(source).resolve()
    ledger = read(source / "attempts.json")
    if not ledger:
        raise ValueError("No source attempts")
    items, seen = [], set()
    for relative, reservation in ledger.items():
        folder = (source / relative).resolve()
        if not folder.is_relative_to(source) or not relative.startswith(
            "sample_shards/"
        ):
            raise ValueError("Only source shard reservations may be imported")
        if (
            reservation.get("stage") != "calibration"
            or reservation.get("count") != 10
            or reservation.get("teacher_id") != target
        ):
            raise ValueError(
                "Recovery cannot discard or reinterpret other charged attempts"
            )
        if not (folder / "COMPLETE.json").exists() or (folder / "FAILED.json").exists():
            raise ValueError(
                "Only completed raw shards may be reused; incomplete costs are not refunded"
            )
        aggregate = validate_receipt(folder)
        terminal_checksum = (folder / "terminal.pt.gz").with_suffix(".sha.json")
        if read(terminal_checksum)["sha256"] != sha(folder / "terminal.pt.gz"):
            raise ValueError("Source terminal checksum differs")
        manifest = read(folder / "manifest.json")
        identity = manifest["identity"]
        if reservation["binding"] != value_sha(identity) or aggregate["attempts"] != 10:
            raise ValueError("Source reservation and completed identity differ")
        if (
            identity["population"] != 10
            or identity["compatibility"]["teacher_id"] != target
            or candidate_identity(identity) != candidate_identity(aggregate)
        ):
            raise ValueError("Source teacher, population or actual strengths differ")
        proof = verify_compatibility(
            identity["compatibility"],
            new_binding,
            source_package,
            Path(__file__).parent,
        )
        slots = validate_slots(identity["logical_slots"])
        keys = {
            (r["logical_seed"], r["logical_slot"])
            for r in read(folder / "records.json")
        }
        if keys != {(seed, slot) for seed in SEEDS for slot in slots}:
            raise ValueError("Source seed/slot population changed")
        actual = candidate_identity(aggregate)
        uniqueness = value_sha(dict(actual=actual, slots=slots))
        if uniqueness in seen:
            raise ValueError("Repeated source population would double-count an attempt")
        seen.add(uniqueness)
        new_id = "sample_shards/" + value_sha(
            dict(compatibility=new_binding, **actual, slots=slots)
        )
        new_identity = dict(
            compatibility=new_binding, **actual, population=10, logical_slots=slots
        )
        items.append(
            dict(
                source=folder,
                source_id=relative,
                new_id=new_id,
                identity=new_identity,
                aggregate=aggregate,
                proof=proof,
                receipt=read(folder / "COMPLETE.json"),
            )
        )
    return items


def run(config):
    from .absolute_runtime import calibrated, require_acceptance

    require_acceptance(config)
    subjects = calibrated(config)
    if len(subjects) != 1:
        raise ValueError("One explicitly audited teacher per recovery")
    settings = config["postprocessing_recovery"]
    if set(settings) != {
        "source_root",
        "source_commit",
        "source_package",
        "preservation_manifest",
        "preservation_sha256",
    }:
        raise ValueError("Unknown postprocessing recovery authorization")
    root, source = (
        Path(config["root"]).resolve(),
        Path(settings["source_root"]).resolve(),
    )
    if root == source or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Recovery views require an independent result root")
    if (root / "attempts.json").exists() or (
        root / "PROMOTION_RECOVERY_CLAIM.json"
    ).exists():
        raise ValueError("Recovery is one-shot; preserve claims instead of retrying")
    preserved_path = Path(settings["preservation_manifest"])
    if sha(preserved_path) != settings["preservation_sha256"]:
        raise ValueError("Recovery preservation authorization changed")
    preserved = read(preserved_path)
    for relative, digest in preserved.items():
        path = source / relative
        if not path.resolve().is_relative_to(source) or sha(path) != digest:
            raise ValueError("Preserved source changed: " + relative)
    subject = subjects[0]
    binding = compatibility(config, subject["teacher"], subject["condition"])
    items = source_items(
        source, subject["teacher"]["target"], binding, settings["source_package"]
    )
    if 10 * len(items) > 350:
        raise ValueError(
            "Imported attempts exceed the unchanged teacher calibration budget"
        )
    ensure_limits(config, 1)
    frozen(
        root / "PROMOTION_RECOVERY_CLAIM.json",
        dict(
            policy=POLICY,
            settings=settings,
            source_ledger_sha256=sha(source / "attempts.json"),
            imported_attempts=10 * len(items),
            actual_new_sampling_attempts=0,
            full_acceptance_sha256=sha(root / "acceptance/COMPLETE.json"),
        ),
    )
    write(
        root / "BUDGET_SCOPE.json",
        dict(stage="calibration", teacher_id=subject["teacher"]["target"]),
    )
    aliases = []
    for item in items:
        old, destination = item["source"], root / item["new_id"]
        destination.mkdir(parents=True, exist_ok=False)
        for name in set(item["receipt"]["files"]) | {"terminal.pt.sha.json"}:
            if name not in METADATA:
                shutil.copy2(old / name, destination / name)
        for name in ("manifest.json", "aggregate.json", "COMPLETE.json"):
            shutil.copy2(old / name, destination / ("ORIGINAL_" + name))
        fingerprint = value_sha(item["identity"])
        frozen(
            destination / "manifest.json",
            dict(id=item["new_id"], identity=item["identity"]),
        )
        provenance = dict(
            policy=POLICY,
            source_root=str(source),
            source_id=item["source_id"],
            actual_sampling_commit=settings["source_commit"],
            original_complete_sha256=sha(old / "COMPLETE.json"),
            original_files_sha256=item["receipt"]["files"],
            sampling_equivalence_proof=item["proof"],
            ledger_alias_not_independent_measurement=True,
            new_sampling_attempts=0,
        )
        frozen(destination / "RECOVERY_PROVENANCE.json", provenance)
        row = dict(
            item["aggregate"],
            id=item["new_id"],
            fingerprint=fingerprint,
            compatibility_sha256=value_sha(binding),
            sampling_recovery_provenance=provenance,
        )
        write(destination / "aggregate.json", row)
        files = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
        frozen(
            destination / "COMPLETE.json",
            dict(
                files=files,
                imported_measurement=True,
                actual_sampling_commit=settings["source_commit"],
                not_a_new_generation=True,
            ),
        )
        frozen(
            root / "slot_registry" / (fingerprint + ".json"),
            dict(source_group=item["new_id"], identity=item["identity"]),
        )
        Budget(root).reserve(item["new_id"], "calibration", 10, fingerprint)
        aliases.append(
            dict(
                source_id=item["source_id"],
                alias_id=item["new_id"],
                attempts=10,
                source_complete_sha256=sha(old / "COMPLETE.json"),
                alias_complete_sha256=sha(destination / "COMPLETE.json"),
            )
        )
    for relative, digest in preserved.items():
        if sha(source / relative) != digest:
            raise ValueError("Source changed while importing: " + relative)
    result = dict(
        policy=POLICY,
        source_root=str(source),
        aliases=aliases,
        imported_attempts=10 * len(aliases),
        actual_new_sampling_attempts=0,
        old_and_new_ledgers_are_alias_views=True,
        source_files_unchanged=True,
        source_failures_and_claims_retained=True,
    )
    frozen(root / "PROMOTION_RECOVERY_COMPLETE.json", result)
    return result
