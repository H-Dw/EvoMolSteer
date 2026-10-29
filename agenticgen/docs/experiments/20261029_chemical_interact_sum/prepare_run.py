"""Prepare a new summed-loss campaign while preserving prior objective evidence."""

import argparse
import copy
import json
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

NEW_PROTOCOL = "template-relative-distance-guidance-v6-summed-error"
OLD_CONDITION_FIELDS = {
    "policy",
    "reward_policy",
    "condition_sha256",
    "distance_normalization",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configuration(previous, root):
    """Change only the declared objective, its skills and the independent root."""
    config = copy.deepcopy(previous)
    config.pop("guidance_reward", None)
    config["protocol"] = NEW_PROTOCOL
    config["root"] = str(root)
    config["agent_tools"]["skill_release"] = "2.3.0"
    if config["rounds"] != 5 or config["chemical_inpainting"] != {"enabled": True}:
        raise ValueError(
            "This authorization retains five rounds and chemical inpainting"
        )
    return config


def assert_same_subjects(previous, current):
    if len(previous) != len(current):
        raise ValueError("Teacher count changed")
    for old, new in zip(previous, current):
        if old["teacher"] != new["teacher"]:
            raise ValueError("Teacher identity or generation context changed")
        before = {
            k: v for k, v in old["condition"].items() if k not in OLD_CONDITION_FIELDS
        }
        after = {
            k: v for k, v in new["condition"].items() if k not in OLD_CONDITION_FIELDS
        }
        if before != after:
            raise ValueError(
                "Condition geometry, atom mapping, chemistry or target pairs changed"
            )
        if old["condition"]["condition_sha256"] == new["condition"]["condition_sha256"]:
            raise ValueError("New objective must have a new frozen condition identity")
        condition = new["condition"]
        controlled = set(condition["controlled_atom_ids"])
        if len(controlled) != 9 or controlled != {
            a["atom_id"] for a in condition["interact_atoms"]
        }:
            raise ValueError(
                "Nine identical condition and interaction ligand atoms required"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--previous-repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from agenticgen.uniguide.geometry_runtime import prepare, validate_config
    from agenticgen.uniguide.io import frozen, sha
    import fcntl

    if args.root.exists() or args.config.exists():
        raise ValueError("New objective requires unused result and configuration paths")
    previous = args.previous_root
    old_config = read(previous / "GEOMETRY_CONFIG.json")
    if read(previous / "FINALIZED.json")["status"] != "terminal_stopped_archived":
        raise ValueError(
            "Previous campaign must be stopped and archived before restart"
        )
    with ExitStack() as stack:
        for path in (
            Path(old_config["generation_context_root"]) / ".executor.lock",
            previous / ".executor.lock",
            previous / ".campaign.lock",
        ):
            stream = stack.enter_context(path.open("r+"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archived = read(previous / "monitoring/terminal_boundary_audit_20261029.json")
        changed = [
            name
            for name, digest in archived["artifact_sha256"].items()
            if sha(previous / name) != digest
        ]
        if changed:
            raise ValueError("Prior measurements changed: " + str(changed))
        protected = read(previous / "SOURCE_PRESERVATION.json")
        protected.update({str(p): sha(p) for p in previous.rglob("*") if p.is_file()})
        for path, digest in protected.items():
            if sha(path) != digest:
                raise ValueError("Prior protected source changed: " + path)
        acceptance = read(previous / "acceptance/COMPLETE.json")
        for name, digest in acceptance["source_binding"].items():
            path = (
                args.previous_repo
                / "agenticgen/scripts"
                / name.removeprefix("entrypoint/")
                if name.startswith("entrypoint/")
                else args.previous_repo / "agenticgen/src/agenticgen" / name
            )
            if sha(path) != digest:
                raise ValueError("Previous frozen implementation changed: " + name)
            protected[str(path)] = digest

        config = validate_config(configuration(old_config, args.root))
        frozen(args.config, config)
        subjects = prepare(config)
        assert_same_subjects(read(previous / "SUBJECTS.json"), subjects)
        frozen(args.root / "SOURCE_PRESERVATION.json", protected)
        frozen(
            args.root / "INPUT_SOURCE_AUDIT.json", read(previous / "INPUT_AUDIT.json")
        )
        source_inputs = copy.deepcopy(read(previous / "INPUT_AUDIT.json"))
        source_inputs["objective_protocol"] = NEW_PROTOCOL
        source_inputs["condition_hashes"] = {
            subject["teacher"]["target"]: subject["condition"]["condition_sha256"]
            for subject in subjects
        }
        frozen(args.root / "INPUT_AUDIT.json", source_inputs)
        prior_attempts = {}
        for row in read(previous / "attempts.json").values():
            stage = row["stage"]
            prior_attempts[stage] = prior_attempts.get(stage, 0) + row["count"]
        audit = dict(
            timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            authorization="User requested committing summed distance objective, deploying and restarting parameter exploration and the formal test",
            previous_root=str(previous),
            previous_repo=str(args.previous_repo),
            root=str(args.root),
            config_sha256=sha(args.config),
            protected_file_count=len(protected),
            previous_evidence_unchanged=True,
            previous_attempts_by_stage=prior_attempts,
            previous_attempts_preserved_in_original_ledger=True,
            new_objective_calibration_limit=700,
            new_objective_formal_limit=1500,
            historical_and_new_attempts_reported_separately=True,
            original_geometry_and_atom_mapping_unchanged=True,
            old_multiplier_reused=False,
            old_measurements_reused=False,
            new_condition_hashes=source_inputs["condition_hashes"],
        )
        frozen(args.root / "RESTART_AUTHORIZATION.json", audit)
    print(json.dumps(audit, ensure_ascii=True))


if __name__ == "__main__":
    main()
