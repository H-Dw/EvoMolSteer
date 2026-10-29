"""Prepare an explicitly authorized restart without overwriting failed evidence."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agenticgen.uniguide.geometry_runtime import prepare, validate_config
from agenticgen.uniguide.io import frozen, read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous-repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    previous = args.previous_root
    if args.root.exists() or args.config.exists():
        raise ValueError("Restart requires new result and configuration paths")
    occupied = [
        name
        for name in ("attempts.json", "calibration", "versions")
        if (previous / name).exists()
    ]
    if occupied:
        raise ValueError("Prior experiment has sampling state; do not reset its budget")
    preservation = read(previous / "monitoring/failure_preservation_20261029_1334.json")
    protected = read(previous / "SOURCE_PRESERVATION.json")
    protected.update(
        {
            str(previous / name): digest
            for name, digest in preservation["artifact_hashes"].items()
        }
    )
    protected.update(
        {
            str(args.previous_repo / name): digest
            for name, digest in preservation["source_hashes"].items()
        }
    )
    changed = [name for name, digest in protected.items() if sha(name) != digest]
    if changed:
        raise ValueError("Original failed evidence changed: " + str(changed))
    config = validate_config(
        dict(read(previous / "GEOMETRY_CONFIG.json"), root=str(args.root))
    )
    frozen(args.config, config)
    subjects = prepare(config)
    if subjects != read(previous / "SUBJECTS.json"):
        raise ValueError("Restart changed teachers, conditions or fixed mappings")
    frozen(args.root / "SOURCE_PRESERVATION.json", protected)
    frozen(args.root / "INPUT_AUDIT.json", read(previous / "INPUT_AUDIT.json"))
    audit = dict(
        timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        authorization="User requested graph repair, deployment, restart and scheduled inspection",
        previous_root=str(previous),
        previous_repo=str(args.previous_repo),
        root=str(args.root),
        config_sha256=sha(args.config),
        protected_file_count=len(protected),
        previous_evidence_unchanged=True,
        subjects_identical=True,
        previous_calibration_attempts=0,
        previous_formal_attempts=0,
        new_calibration_required=True,
        old_lambda_reused=False,
    )
    frozen(args.root / "RESTART_AUTHORIZATION.json", audit)
    import json

    print(json.dumps(audit, ensure_ascii=True))


if __name__ == "__main__":
    main()
