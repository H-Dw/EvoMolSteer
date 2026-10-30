"""Import exact sampling evidence; search policy changes never rewrite old receipts."""
from pathlib import Path
import shutil

from .io import read, frozen, sha, value_sha


def import_measurements(config, subjects):
    from .geometry_sampling import compatibility, validate_receipt
    source_setting = config.get("calibration", {}).get("measurement_source_root")
    if not source_setting:
        return {}
    root, source = Path(config["root"]).resolve(), Path(source_setting).resolve()
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Calibration measurement source must be independent")
    complete = root / "CALIBRATION_MEASUREMENTS_IMPORTED.json"
    if complete.exists():
        audit = read(complete)
        if audit["source_root"] != str(source):
            raise ValueError("Calibration source changed")
        for name, digest in audit["copied_files"].items():
            if sha(root / name) != digest or sha(source / name) != digest:
                raise ValueError("Imported sampling source changed")
        return read(root / "calibration/IMPORTED_COARSE.json")
    claim = root / "CALIBRATION_MEASUREMENT_IMPORT_CLAIM.json"
    if claim.exists():
        raise RuntimeError("Interrupted import claim; explicit audit required")
    existing = read(root / "attempts.json") if (root / "attempts.json").exists() else {}
    if existing:
        raise ValueError("Import measurements before new scientific reservations")
    if not (source / "acceptance/COMPLETE.json").exists():
        raise ValueError("Sampling source lacks completed acceptance")
    imported, copies, teachers_by_run = {}, {}, {}
    bindings = {s["teacher"]["target"]: compatibility(config, s["teacher"], s["condition"])
                for s in subjects}
    for registry in ("measurement_registry", "slot_registry"):
        for path in sorted((source / registry).glob("*.json")):
            record = read(path)
            identity = record["identity"]
            target = identity["compatibility"]["teacher_id"]
            if target not in bindings:
                continue
            if identity["compatibility"] != bindings[target]:
                raise ValueError("Sampling implementation, condition, model, RNG or statistics differ")
            group = (source / record["source_group"]).resolve()
            if not group.is_relative_to(source) or group == source:
                raise ValueError("Source group escapes calibration root")
            if (group / "COMPLETE.json").exists():
                validate_receipt(group)
            elif not (group / "FAILED.json").exists():
                raise ValueError("Unfinished sampling cannot be imported as a measurement")
            copies[path.relative_to(source).as_posix()] = path
            for product in group.rglob("*"):
                if product.is_file():
                    copies[product.relative_to(source).as_posix()] = product
            teachers_by_run[record["source_group"]] = target
    for path in sorted((source / "calibration/frozen").glob("*/coarse_results.json")):
        rows = read(path)
        for row in rows:
            target = row["teacher_id"]
            if target in bindings:
                if row["compatibility_sha256"] != value_sha(bindings[target]) or row["attempts"] != 10:
                    raise ValueError("Incompatible coarse population")
                imported.setdefault(target, []).append(row)
    ledger = {}
    for run_id, entry in read(source / "attempts.json").items():
        if run_id in teachers_by_run:
            if entry["stage"] != "calibration":
                raise ValueError("Only calibration measurements may be imported here")
            ledger[run_id] = dict(entry, teacher_id=teachers_by_run[run_id])
    if not ledger or sum(x["count"] for x in ledger.values()) > 700:
        raise ValueError("Missing or excessive source calibration budget")
    for target in bindings:
        if sum(x["count"] for x in ledger.values() if x["teacher_id"] == target) > 350:
            raise ValueError("Source exceeds teacher calibration budget")
    hashes = {name: sha(path) for name, path in copies.items()}
    frozen(claim, dict(source_root=str(source), source_attempts_sha256=sha(source / "attempts.json"),
                       copied_files=hashes, aliases_not_independent=True))
    for name, path in copies.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != hashes[name]:
            raise ValueError("Import would overwrite a different artifact")
        if not destination.exists():
            shutil.copy2(path, destination)
        if sha(destination) != hashes[name] or sha(path) != hashes[name]:
            raise ValueError("Measurement changed during import")
    frozen(root / "attempts.json", ledger)
    frozen(root / "calibration/IMPORTED_COARSE.json", imported)
    frozen(complete, dict(source_root=str(source), copied_files=hashes,
                         source_attempts_sha256=sha(source / "attempts.json"),
                         carried_attempts=sum(x["count"] for x in ledger.values()),
                         ledger_views_are_aliases=True, new_sampling_attempts=0,
                         failed_claims_preserved=True, new_acceptance_required=True))
    return imported
