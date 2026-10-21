"""Explicit teacher identities and completion accounting for the selected study."""
from .io import read, value_sha


TARGET = "5nn4_SC2_A_1017"
TEACHERS = [
    {"target": TARGET, "rank": rank, "teacher_id": f"{TARGET}__top{rank}", "node_id": node}
    for rank, node in [(1, "single_b002_final_005"), (2, "single_b004_final_009"),
                       (3, "single_b002_final_007")]
]


def select_templates(archive, selection):
    """Keep original ranks; reject missing, duplicated, substituted or empty teachers."""
    if not selection or len({s["teacher_id"] for s in selection}) != len(selection):
        raise ValueError("Teacher selection must contain unique explicit identities")
    chosen = []
    for spec in selection:
        matches = [t for t in archive if t["target"] == spec["target"] and t["rank"] == spec["rank"]]
        if len(matches) != 1:
            raise ValueError("Missing or duplicate selected teacher: " + spec["teacher_id"])
        t = matches[0]
        if t["teacher_id"] != spec["teacher_id"] or t["source_record"]["node_id"] != spec["node_id"]:
            raise ValueError("Selected teacher identity/ranking differs: " + spec["teacher_id"])
        if not t["core_atoms"]:
            raise ValueError("Selected teacher has an empty PLIP core: " + spec["teacher_id"])
        chosen.append(t)
    return chosen


def counts(config, templates):
    teachers = len(templates)
    batches = teachers * config["batches"]
    chains = batches * config["batch_size"]
    return {"teacher_count": teachers, "target_count": len({t["target"] for t in templates}),
            "initial_batches": len({t["target"] for t in templates}) * config["batches"],
            "teacher_batches": batches, "chains_per_version": chains,
            "decisions_per_version": chains * config["rounds"],
            "r0_results": chains,
            "maximum_results_all_versions": chains * (1 + len(config["versions"]) * config["rounds"])}


def progress(root, version, config, templates):
    """Count only selected, fully committed batches, never arbitrary COMPLETE files."""
    folder = root / "versions" / version
    expected = {f"{t['teacher_id']}/batch_{b:03d}" for t in templates for b in range(config["batches"])}
    observed = {p.parent.relative_to(folder).as_posix() for p in folder.glob("*/batch_*/COMPLETE.json")}
    completed, invalid = [], []
    for name in sorted(observed & expected):
        try:
            batch = folder / name
            marker = read(batch / "COMPLETE.json")
            chains = read(batch / "histories.json")
            if marker != {"chains": config["batch_size"], "rounds": config["rounds"]}:
                raise ValueError("Completion marker disagrees with the frozen budget")
            if len(chains) != config["batch_size"] or [c["slot"] for c in chains] != list(range(config["batch_size"])):
                raise ValueError("Incomplete or duplicated candidate slots")
            for c in chains:
                if [h["id"] for h in c["history"]] != [f"r{i:02d}" for i in range(config["rounds"] + 1)]:
                    raise ValueError("Not every candidate has all committed rounds")
                if c["best_id"] not in {h["id"] for h in c["history"]}:
                    raise ValueError("Incumbent is absent from history")
            if config.get("retention", {}).get("mode") == "top1_per_chain":
                receipt = read(batch / f"round_{config['rounds']:02d}" / "RETENTION_COMPLETE.json")
                if receipt["history_sha256"] != value_sha(chains):
                    raise ValueError("Final round retention has not committed")
            completed.append(name)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"batch": name, "error": str(exc)})
    return {**counts(config, templates), "completed_batches": len(completed),
            "completed": completed, "missing": sorted(expected - set(completed)),
            "invalid": invalid, "unexpected": sorted(observed - expected),
            "complete": set(completed) == expected and not invalid and not (observed - expected)}
