"""Teacher-local scale selection. This is a new protocol, not a legacy migration.

Bounds are empirical exploration limits, not a proof of numerical stability.
Only comparable full-population constant profiles set the initial domain.
"""
import copy
import math

from .io import value_sha
from .protocol import choose, lattice_index


POLICY = "teacher-local-scale-v1"
PHASE_COUNTS = {"coarse": 10, "fine": 20, "windows": 25, "full": 50}
PHASE_LIMITS = {"coarse": 26, "fine": 32, "windows": 24, "full": 20}
GROWTH_LIMIT = 10.0


def teacher_rows(rows, target, phase):
    selected = [r for r in rows if r["target"] == target and r["phase"] == phase]
    if any(r["attempts"] != PHASE_COUNTS[phase] for r in selected):
        raise ValueError("Do not compare different calibration populations")
    schedules = [tuple(r["schedule"]) for r in selected]
    if len(set(schedules)) != len(schedules):
        raise ValueError("Duplicate teacher/phase configuration is not independent evidence")
    for row in selected:
        if len(row["schedule"]) != 4 or any(not math.isfinite(s) or s <= 0 for s in row["schedule"]):
            raise ValueError("Invalid calibration schedule")
        if row.get("complete") and any(not math.isfinite(row[k]) for k in ("C", "gap", "best")):
            raise ValueError("Complete calibration metrics must be finite")
    return selected


def pareto(rows):
    return [r for r in rows if not any(o["C"] <= r["C"] and o["gap"] <= r["gap"] and
        (o["C"] < r["C"] or o["gap"] < r["gap"]) for o in rows)]


def displacement(row):
    windows = row.get("window_diagnostics", [])
    if len(windows) != 4:
        raise ValueError("Missing absolute displacement evidence; cannot infer a range")
    result = {}
    for key in ("increment_vector_rms_A", "max_atom_increment_A"):
        values = [w.get(key) for w in windows]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Unknown absolute displacement evidence")
        result[key] = max(values)
    return result


def derive_teacher_domain(rows, teacher):
    """Detect each teacher's own abrupt displacement growth before C/gap ranking.

    A >10x change beyond linear scale growth in either absolute diagnostic is
    a conservative engineering heuristic. It is NOT a confidence interval or
    a physical validity test. It never clips gradients or retries a sample.
    """
    target = teacher["target"]
    full = teacher_rows(rows, target, "full")
    constants = sorted((r for r in full if len(set(r["schedule"])) == 1),
                       key=lambda r: r["schedule"][0])
    if len(constants) < 3:
        raise ValueError("Need at least three full-population constants for " + target)
    curve, eligible, previous, boundary = [], [], None, None
    for row in constants:
        j = lattice_index(row["schedule"][0])
        item = {k: copy.deepcopy(row[k]) for k in ("id", "schedule", "attempts", "complete")}
        item["source_sha256"] = value_sha(row)
        item["index"] = j
        if not row.get("complete"):
            item.update(status="failed", error=row.get("error"))
            if boundary is None:
                boundary = dict(index=j, source=row["id"], reason="failed_full_constant")
        else:
            absolute = displacement(row)
            item.update({k: row[k] for k in ("C", "gap", "best")})
            item.update({k: row[k] for k in ("C_se", "built", "mean_pic50") if k in row})
            item["absolute_displacement"] = absolute
            growth = {}
            if previous is not None:
                scale_ratio = row["schedule"][0] / previous["schedule"][0]
                # No extrapolation across an unmeasured gap wider than an octave.
                if j - previous["index"] > 2 and boundary is None:
                    boundary = dict(index=j, source=row["id"], reason="sparse_scale_evidence")
                for key, value in absolute.items():
                    old = previous["absolute_displacement"][key]
                    growth[key] = value / max(old * scale_ratio, 1e-12)
                if max(growth.values()) > GROWTH_LIMIT and boundary is None:
                    boundary = dict(index=j, source=row["id"], reason="abrupt_absolute_displacement_growth",
                                    compared_to=previous["id"], normalized_growth=growth)
            item["normalized_growth"] = growth
            item["status"] = "outside_empirical_segment" if boundary else "eligible"
            if boundary is None:
                eligible.append(row)
            previous = item
        curve.append(item)
    if len(eligible) < 3:
        raise ValueError("Insufficient contiguous evidence below the risk boundary for " + target)
    frontier = pareto(eligible)
    low = max(lattice_index(eligible[0]["schedule"][0]),
              min(lattice_index(r["schedule"][0]) for r in frontier) - 2)
    high = min(lattice_index(eligible[-1]["schedule"][0]),
               max(lattice_index(r["schedule"][0]) for r in frontier) + 2)
    # An initial anchor needs at least one fine-grid move in BOTH directions.
    # Do not expand into the observed failure region to manufacture that room.
    interior = [r for r in eligible if low < lattice_index(r["schedule"][0]) < high]
    if not interior:
        raise ValueError("No measured interior initial scale; more calibration evidence is needed for " + target)
    winner = choose(interior)
    measured = {lattice_index(r["schedule"][0]) for r in eligible}
    return dict(policy=POLICY, target=target, node_id=teacher["node_id"], teacher_sha256=value_sha(teacher),
        initial=list(winner["schedule"]), initial_source=winner["id"],
        performance_best_source=choose(eligible)["id"],
        min_index=low, max_index=high, lower=2.**(low/2), upper=2.**(high/2),
        lattice="2^(j/2)", coarse_log2_step=1., fine_log2_step=.5,
        derivation="teacher-only full-50 constants; displacement segment; Pareto plus one octave capped by measured segment; C-band/gap selection among measured interior constants",
        selection_band_A=.02, normalized_displacement_growth_limit=GROWTH_LIMIT,
        risk_boundary=boundary, unmeasured_indices=[j for j in range(low, high+1) if j not in measured],
        stability_guarantee=False, constant_comparison=curve,
        source_rows_sha256=value_sha(full))


def derive_domains(rows, teachers):
    targets = [t["target"] for t in teachers]
    if len(targets) != 2 or len(set(targets)) != 2:
        raise ValueError("Exactly two distinct frozen teachers required")
    return dict(policy=POLICY, by_teacher={t["target"]: derive_teacher_domain(rows, t) for t in teachers},
                cross_teacher_pooling=False)


def resolve_domain(domain, teacher):
    if domain.get("policy") != POLICY or "by_teacher" not in domain:
        raise ValueError("Legacy shared domain requires its frozen implementation; no automatic migration")
    local = domain["by_teacher"].get(teacher["target"])
    if not local or local.get("target") != teacher["target"] or local.get("node_id") != teacher["node_id"] or local.get("teacher_sha256") != value_sha(teacher):
        raise ValueError("Teacher-specific domain identity mismatch")
    initial = [lattice_index(s) for s in local["initial"]]
    if len(initial) != 4 or len(set(initial)) != 1 or not local["min_index"] < initial[0] < local["max_index"]:
        raise ValueError("Initial scale must have fine-grid actions on both sides")
    return copy.deepcopy(local)
