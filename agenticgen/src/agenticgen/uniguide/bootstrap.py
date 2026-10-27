"""Read-only initialization from the first measured multi-scale population.

The existing R0 molecular checkpoint and this measured action anchor are
different artifacts. Reusing a measurement never creates a new attempt.
"""
import copy
from itertools import combinations, permutations
from pathlib import Path

from .io import read, sha, value_sha
from .protocol import SEEDS, choose, lattice_index, validate_action
from .scale_domain import POLICY, teacher_rows


def bootstrap_history(root, teacher, domain):
    root = Path(root).resolve()
    if domain.get("policy") != POLICY or domain["teacher_sha256"] != value_sha(teacher):
        raise ValueError("Bootstrap requires the bound teacher-specific multi-scale domain")
    full = teacher_rows(read(root/"calibration/full_results.json"), teacher["target"], "full")
    if value_sha(full) != domain["source_rows_sha256"]:
        raise ValueError("First-round multi-scale evidence changed")
    sources = {r["id"]: r for r in full}
    ids = domain["bootstrap_sources"]
    if not ids or ids[0] != domain["initial_source"] or len(ids) != len(set(ids)):
        raise ValueError("Invalid bootstrap identities")
    history = []
    for key in ids:
        row = sources[key]
        folder = (root/key).resolve()
        expected_parent = (root/"calibration/full"/teacher["target"]).resolve()
        if folder.parent != expected_parent or not row["complete"]:
            raise ValueError("Bootstrap must reference this teacher's completed full-population group")
        manifest, receipt = read(folder/"manifest.json"), read(folder/"COMPLETE.json")
        if (manifest["id"] != key or manifest["teacher_sha256"] != value_sha(teacher)
                or manifest["count"] != 50 or manifest["schedule"] != row["schedule"]
                or receipt["manifest_sha256"] != sha(folder/"manifest.json")
                or receipt["records_sha256"] != sha(folder/"records.json")):
            raise ValueError("Bootstrap measurement binding failed")
        records = read(folder/"records.json")
        if (len(records) != 50 or len({r["node_id"] for r in records}) != 50
                or {(r["logical_seed"], r["logical_slot"]) for r in records}
                != {(seed, slot) for seed in SEEDS for slot in range(10)}):
            raise ValueError("Bootstrap population must preserve all 50 original seed/slot identities")
        aggregate = read(folder/"aggregate.json")
        expected = {k: v for k, v in row.items() if k not in ("target", "phase", "quality")}
        if value_sha(aggregate) != value_sha(expected):
            raise ValueError("Bootstrap aggregate disagrees with original multi-scale results")
        if any(not domain["min_index"] <= lattice_index(s) <= domain["max_index"] for s in aggregate["schedule"]):
            raise ValueError("Bootstrap anchor outside teacher domain")
        history.append(copy.deepcopy(aggregate))
    initial = sources[ids[0]]
    if value_sha(initial) != domain["initial_source_sha256"] or initial["schedule"] != domain["initial"]:
        raise ValueError("Initial scale is not the selected measured source")
    if len({tuple(r["schedule"]) for r in history}) != len(history):
        raise ValueError("Duplicate bootstrap schedule is not independent evidence")
    return history


def legal_options(anchor, history, domain, branch, limit=2):
    """Bounded deterministic feasibility check; does not ask the LLM to search."""
    seen = [r["schedule"] for r in history]
    anchors = {r["id"]: r for r in history}
    old = [lattice_index(s) for s in anchor["schedule"]]
    changes = ([(pair, (1, -1)) for pair in permutations(range(4), 2)] if branch == "redistribute"
        else [(indices, (1 if branch == "increase" else -1,)*len(indices))
              for count in (1, 2) for indices in combinations(range(4), count)])
    result = []
    for mode, step in (("fine", 1), ("coarse", 2)):
        for indices, signs in changes:
            schedule = list(anchor["schedule"])
            for index, sign in zip(indices, signs):
                schedule[index] = 2.**((old[index]+sign*step)/2)
            proposal = dict(anchor_id=anchor["id"], branch=branch, mode=mode, schedule=schedule)
            try:
                validate_action(proposal, anchors, domain, seen+[p["schedule"] for p in result])
            except ValueError:
                continue
            result.append(proposal)
            if len(result) == limit:
                return result
    return result


def action_anchor(history, domain, branch):
    preferred = next((r for r in reversed(history) if r.get("branch") == branch), history[0])
    if len(legal_options(preferred, history, domain, branch)) == 2:
        return preferred, "latest_branch_or_measured_initial"
    candidates = [r for r in history if len(legal_options(r, history, domain, branch)) == 2]
    if not candidates:
        raise ValueError("No two untested legal " + branch + " actions; no automatic expansion or resampling")
    return choose(candidates), "explicit_measured_anchor_for_branch_feasibility; initial_scale_unchanged"
