"""Point-in-time computational summaries for Scientist evidence windows.

This module emits numeric observations only. It deliberately does not turn
descriptors into language hypotheses or draw conclusions about outcomes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_FEATURES = (
    "radius_gyration_A",
    "centroid_pocket_centroid_A",
    "nearest_pocket_A",
    "contact_pairs_4A",
    "pocket_contacts_4A",
    "planarity_fraction",
    "asphericity",
)
_REPRESENTATIONS = ("current", "forecast")
_TOL = 1e-6


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _load_source(source: Any) -> Mapping[str, Any]:
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise TypeError("source must be a Scientist packet mapping or JSON path")
    return source


def _teacher_and_background(source: Mapping[str, Any]) -> tuple[list, list]:
    teacher = source.get("teacher", source)
    if not isinstance(teacher, Mapping) or not isinstance(teacher.get("donors"), list):
        raise ValueError("source must contain teacher.donors")
    background = source.get("background")
    if not isinstance(background, list):
        raise ValueError("source must contain point-indexed background rows")
    return teacher["donors"], background


def _quantiles(row: Mapping[str, Any], key: str) -> tuple[float | None, float | None, float | None]:
    values = row.get("quantiles_P25_P50_P75", {}).get(key)
    if isinstance(values, Mapping):
        return tuple(_finite_number(values.get(k)) for k in ("P25", "P50", "P75"))
    if isinstance(values, (list, tuple)) and len(values) >= 3:
        return tuple(_finite_number(v) for v in values[:3])
    return None, None, None


def _band(value: float | None, p25: float | None, p75: float | None) -> str | None:
    if value is None or p25 is None or p75 is None:
        return None
    if value < p25:
        return "below"
    if value > p75:
        return "above"
    return "inside"


def _direction(start: float | None, end: float | None) -> str | None:
    if start is None or end is None:
        return None
    delta = end - start
    if delta > _TOL:
        return "up"
    if delta < -_TOL:
        return "down"
    return "flat"


def _numeric_summary(values: list[float | None], background_end: Mapping[str, Any], key: str) -> dict[str, Any]:
    start, mid, end = values[0], values[2], values[-1]
    directions = [_direction(a, b) for a, b in zip(values, values[1:])]
    active = [d for d in directions if d in ("up", "down")]
    p25, p50, p75 = _quantiles(background_end, key)
    deltas = [b - a for a, b in zip(values, values[1:]) if a is not None and b is not None]
    positive, negative = any(d > 1e-6 for d in deltas), any(d < -1e-6 for d in deltas)
    measured_pattern = ('mixed' if not deltas or positive and negative else
                        'increase' if positive else 'decrease' if negative else 'flat')
    return {
        "start": start,
        "mid": mid,
        "end": end,
        "net_delta": end - start if start is not None and end is not None else None,
        "net_direction": _direction(start, end),
        "adjacent_directions": directions,
        "sign_reversal": any(a != b for a, b in zip(active, active[1:])),
        "measured_pattern_1e6_tolerance": measured_pattern,
        "end_vs_background": _band(end, p25, p75),
    }


def _centroid_xyz_delta(observations: list[Mapping[str, Any]], prefix: str) -> list[float] | None:
    fields = [f"{prefix}_centroid_{axis}_A" for axis in "xyz"]
    start = [_finite_number(observations[0].get(k)) for k in fields]
    end = [_finite_number(observations[-1].get(k)) for k in fields]
    if any(v is None for v in start + end):
        return None
    return [float(b - a) for a, b in zip(start, end)]


def summarize_scientist_interval(source: Any, interval: int) -> dict[str, Any]:
    """Summarize the six donor/background states in one five-step window.

    ``source`` is the object returned by ``orchestrate.teacher_packet`` (or a
    compatible mapping with ``teacher.donors`` and quantiled ``background``).
    Only steps ``5*(interval-1)`` through ``5*interval`` are read. Each donor
    card contains start/mid/end values, numeric endpoint deltas, adjacent
    direction codes, reversal flags, and end-of-window background bands.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 10:
        raise ValueError("interval must be an integer from 1 through 10")
    lo, hi = 5 * (interval - 1), 5 * interval
    mid = lo + (hi - lo) // 2
    donors, background = _teacher_and_background(_load_source(source))
    if len(donors) != 30:
        raise ValueError(f"expected exactly 30 teacher donors, found {len(donors)}")

    needed_steps = list(range(lo, hi + 1))
    needed = set(needed_steps)
    background_by_step = {
        int(row["step"]): row for row in background
        if isinstance(row, Mapping) and row.get("step") in needed
    }
    if any(step not in background_by_step for step in needed_steps):
        raise ValueError(f"background is missing a required step in {lo}..{hi}")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("observations"), list)
           or len(row["observations"]) <= hi for row in donors):
        raise ValueError(f"each teacher donor must contain observations through step {hi}")

    cards = []
    donor_values: dict[tuple[int, str], list[float | None]] = {}
    for donor in donors:
        donor_id = donor.get("donor_id")
        if not isinstance(donor_id, str) or not donor_id:
            raise ValueError("teacher donors must have opaque donor_id values")
        states = donor["observations"][lo:hi + 1]
        # Materialize only the fixed feature set from the requested six states.
        monitored = {}
        for rep in _REPRESENTATIONS:
            for feature in _FEATURES:
                key = f"{rep}_{feature}"
                values = [_finite_number(state.get(key)) for state in states]
                monitored[key] = _numeric_summary(values, background_by_step[hi], key)
                donor_values.setdefault((lo, key), [])
                for j, value in enumerate(values):
                    donor_values.setdefault((lo + j, key), []).append(value)

        footprints = {}
        for rep in _REPRESENTATIONS:
            key = f"{rep}_residue_contact_footprint"
            footprints[rep] = {
                label: states[index].get(key)
                for label, index in (("start", 0), ("mid", mid - lo), ("end", hi - lo))
            }
        cards.append({
            "donor_id": donor_id,
            "interval": interval,
            "start_step": lo,
            "mid_step": mid,
            "end_step": hi,
            "features": monitored,
            "residue_contact_footprints": footprints,
            "centroid_xyz_delta_A": {
                rep: _centroid_xyz_delta(states, rep) for rep in _REPRESENTATIONS
            },
        })

    cohort_background = []
    for step in needed_steps:
        row = background_by_step[step]
        for rep in _REPRESENTATIONS:
            for feature in _FEATURES:
                key = f"{rep}_{feature}"
                p25, p50, p75 = _quantiles(row, key)
                values = [v for v in donor_values.get((step, key), []) if v is not None]
                cohort_background.append({
                    "step": step,
                    "feature": key,
                    "teacher_median": float(np.median(values)) if values else None,
                    "background_median": p50,
                    "background_P25": p25,
                    "background_P75": p75,
                    "teacher_count_below_background_P25": sum(v < p25 for v in values) if p25 is not None else None,
                    "teacher_count_inside_background_P25_P75": sum(p25 <= v <= p75 for v in values)
                    if p25 is not None and p75 is not None else None,
                    "teacher_count_above_background_P75": sum(v > p75 for v in values) if p75 is not None else None,
                    "teacher_count_nonnull": len(values),
                })

    return {
        "interval": interval,
        "start_step": lo,
        "mid_step": mid,
        "end_step": hi,
        "donors": cards,
        "cohort_background": cohort_background,
    }
