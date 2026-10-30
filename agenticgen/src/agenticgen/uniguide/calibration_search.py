"""Deterministic, bounded search decisions; no model, RNG or molecule operations."""
from dataclasses import dataclass
import math

POLICY = "adaptive-bracket-population-v2"
EFFECTIVE = "effective_guidance_scale_schedule"
RANKING = "ranking_condition_coordinate_root_mean_square_deviation_angstrom"
RMSD = "condition_coordinate_root_mean_square_deviation_angstrom"
STATISTICS = "coordinate_loss_statistics"
GROWTH_LIMIT = 10.0
POPULATIONS = (10, 20, 50)


@dataclass(frozen=True)
class SearchLimits:
    maximum_attempts_per_teacher: int = 350
    maximum_new_screening_scales: int = 6
    maximum_bracket_probes: int = 4
    maximum_promoted_configurations: int = 6
    log2_resolution: float = .5


def strength(row):
    values = row[EFFECTIVE]
    if len(values) != 4 or len(set(values)) != 1 or values[0] <= 0 or not math.isfinite(values[0]):
        raise ValueError("Calibration requires positive finite constant profiles")
    return values[0]


def score(row):
    value = row.get(RANKING, row.get(RMSD, row.get("molecular_internal_pair_distance_root_mean_square_deviation_angstrom")))
    if value is None or not math.isfinite(value):
        raise ValueError("Missing finite calibration score")
    return value


def displacement(row):
    windows = row["window_diagnostics"]
    if len(windows) != 4:
        raise ValueError("Four measured displacement windows required")
    values = {key: max(w.get("unconstrained_" + key, w[key]) for w in windows)
              for key in ("increment_vector_rms_A", "max_atom_increment_A")}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Invalid raw displacement evidence")
    return values


def assess(rows):
    """One fidelity at a time; report the observed connected usable component."""
    if len({r["attempts"] for r in rows}) > 1:
        raise ValueError("Do not compare different calibration populations")
    ordered = sorted(rows, key=strength)
    if len({strength(r) for r in ordered}) != len(ordered):
        raise ValueError("Duplicate calibration configuration")
    curve, usable, previous, boundary = [], [], None, None
    for row in ordered:
        entry = dict(source_id=row["id"], effective_scale=strength(row), complete=row["complete"],
                     population=row["attempts"])
        reason = None
        if not row["complete"]:
            reason = "nonfinite_measurement"
        elif not row.get(STATISTICS, {}).get("ranking_eligible", True):
            reason = "statistics_ineligible"
        else:
            absolute = displacement(row)
            growth = {}
            if previous is not None:
                ratio = strength(row) / strength(previous)
                prior = displacement(previous)
                growth = {k: absolute[k] / max(prior[k] * ratio, 1e-12) for k in absolute}
                if max(growth.values()) > GROWTH_LIMIT:
                    reason = "abrupt_displacement_growth"
            entry.update(absolute_displacement=absolute, normalized_growth=growth)
            if reason is None and boundary is None:
                usable.append(row)
                previous = row
        if reason is not None and boundary is None:
            boundary = dict(source_id=row["id"], effective_scale=strength(row), reason=reason)
        entry.update(eligible=row in usable,
                     status=reason or ("usable" if row in usable else "outside_observed_component"))
        curve.append(entry)
    return usable, curve, boundary


def best(rows):
    if not rows:
        raise ValueError("No usable measured configuration")
    return min(rows, key=lambda r: (score(r), strength(r), r["id"]))


def next_probe(rows, limits=SearchLimits()):
    """Bisect a measured neighbour interval in log space; never expand the grid."""
    usable, curve, _ = assess(rows)
    if not usable:
        return None
    center = strength(best(usable))
    tested = sorted(strength(r) for r in rows)
    index = tested.index(center)
    intervals = []
    for neighbour in tested[max(0, index - 1):index] + tested[index + 1:index + 2]:
        width = abs(math.log2(neighbour / center))
        if width > limits.log2_resolution + 1e-9:
            # Prefer resolving the upper stability/quality boundary on equal widths.
            neighbour_state=next(r for r in curve if r["effective_scale"]==neighbour)
            risk_upper=neighbour>center and not neighbour_state["eligible"]
            intervals.append((risk_upper, width, neighbour > center, neighbour))
    if not intervals:
        return None
    _, _, _, neighbour = max(intervals)
    return math.sqrt(center * neighbour)


def phase_key(scale, center):
    return round(math.log2(scale / center) % 1, 8) % 1


def support_panel(scale, minimum, maximum):
    """Three contiguous integer-log2 levels, inside the observed usable bounds."""
    for offsets in ((-2, -1, 0), (-1, 0, 1), (0, 1, 2)):
        panel = [scale * 2.**i for i in offsets]
        if minimum * (1 - 1e-9) <= panel[0] and panel[-1] <= maximum * (1 + 1e-9):
            return panel
    return None


def panel_candidates(rows, center):
    usable, _, _ = assess(rows)
    if not usable:
        return []
    lower, upper = strength(usable[0]), strength(usable[-1])
    phases, panels = set(), []
    for row in sorted(usable, key=lambda r: (score(r), strength(r))):
        key = phase_key(strength(row), center)
        if key in phases:
            continue
        panel = support_panel(strength(row), lower, upper)
        if panel is not None:
            panels.append(panel)
            phases.add(key)
        if len(panels) == 2:
            break
    return panels


def select_panel(rows):
    usable, _, _ = assess(rows)
    by_scale = {round(math.log2(strength(r)), 9): r for r in usable}
    panels = {}
    for row in usable:
        exponent = round(math.log2(strength(row)), 9)
        for start in (exponent - 2, exponent - 1, exponent):
            keys = tuple(round(start + i, 9) for i in range(3))
            if all(k in by_scale for k in keys):
                panels[keys] = [by_scale[k] for k in keys]
    if not panels:
        raise SearchInsufficient("No three usable integer-log2 support levels")
    # Prefer the panel with the best measured score, then the weaker envelope.
    return min(panels.values(), key=lambda p: (score(best(p)), strength(p[-1])))


class SearchInsufficient(ValueError):
    """Expected teacher-local exhaustion or inadequate evidence, not engineering failure."""


def derive_domain(rows, teacher_id, condition_hash, protocol_hash):
    from .io import value_sha
    from .absolute_coordinate import POLICY as coordinate_policy
    full = [r for r in rows if r["phase"] == "full"]
    if any(r["teacher_id"] != teacher_id or r["condition_sha256"] != condition_hash or
           r["compatibility_sha256"] != protocol_hash or r["attempts"] != 50 for r in full):
        raise ValueError("Incompatible full-population calibration evidence")
    panel = select_panel(full)
    winner = best(panel)
    multiplier = strength(winner)
    usable, curve, boundary = assess(full)
    lower = round(math.log2(strength(panel[0]) / multiplier))
    upper = round(math.log2(strength(panel[-1]) / multiplier))
    body = dict(policy=coordinate_policy, teacher_id=teacher_id, condition_sha256=condition_hash,
                sampling_protocol_sha256=protocol_hash,
                calibration_key=value_sha([teacher_id, condition_hash, protocol_hash]),
                calibrated_guidance_multiplier=multiplier,
                normalized_minimum=lower, normalized_maximum=upper,
                initial_normalized_guidance_scale_schedule=[0] * 4,
                initial_source=winner["id"], bootstrap_sources=[r["id"] for r in panel],
                source_rows_sha256=value_sha(full), constant_comparison=curve,
                risk_boundary=boundary, stability_guarantee=False,
                calibration_search_policy=POLICY,
                domain_status="bounded_best" if winner in (panel[0], panel[-1]) else "interior_best",
                allowed_normalized_levels=list(range(lower, upper + 1)),
                confirmed_effective_levels=[strength(r) for r in panel],
                unmeasured_interpolation_allowed=False, global_optimum_claimed=False,
                ranking="minimum upper-tail-filtered condition coordinate RMSD, then smaller actual strength")
    return dict(body, domain_sha256=value_sha(body))
