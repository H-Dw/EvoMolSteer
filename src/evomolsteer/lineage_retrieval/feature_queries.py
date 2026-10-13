"""Bounded source-only numeric feature summaries for Scientist evidence."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_REPRESENTATIONS = ("current", "forecast", "delta")
_DIRECTION_TOLERANCE = 1e-6
_BACKGROUND_COMPARISON_TOLERANCE = 1e-6
_FORBIDDEN_TOKENS = ("pic50", "affinity", "score", "rank", "probability", "weight", "node_id", "batch", "slot", "root")


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _source_obj(source: Any) -> Mapping[str, Any]:
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise TypeError("source must be a Scientist packet mapping or JSON path")
    return source


def _teacher_background(source: Mapping[str, Any]) -> tuple[list, list]:
    teacher = source.get("teacher", source)
    if not isinstance(teacher, Mapping) or not isinstance(teacher.get("donors"), list):
        raise ValueError("source must contain teacher.donors")
    background = source.get("background")
    if not isinstance(background, list):
        raise ValueError("source must contain background rows")
    if len(teacher["donors"]) != 30:
        raise ValueError(f"expected 30 teacher donors, found {len(teacher['donors'])}")
    return teacher["donors"], background


def _quantile_triplet(row: Mapping[str, Any] | None, feature: str) -> tuple[float | None, float | None, float | None]:
    if not isinstance(row, Mapping):
        return None, None, None
    quantiles = row.get("quantiles_P25_P50_P75")
    values = quantiles.get(feature) if isinstance(quantiles, Mapping) else None
    if isinstance(values, Mapping):
        triplet = [values.get(k) for k in ("P25", "P50", "P75")]
    elif isinstance(values, (list, tuple)) and len(values) >= 3:
        triplet = values[:3]
    else:
        return None, None, None
    return tuple(_finite_scalar(v) for v in triplet)


def _direction(a: float, b: float) -> str:
    delta = b - a
    if delta > _DIRECTION_TOLERANCE:
        return "up"
    if delta < -_DIRECTION_TOLERANCE:
        return "down"
    return "flat"


def _pattern(directions: Sequence[str]) -> str:
    up, down = "up" in directions, "down" in directions
    if up and down:
        return "mixed"
    if up:
        return "increase"
    if down:
        return "decrease"
    return "flat"


def _background_contrast(teacher: float | None, background: float | None) -> dict[str, Any]:
    """Compare numeric medians with an absolute tolerance, not a test of significance."""
    delta = teacher - background if teacher is not None and background is not None else None
    direction = None
    if delta is not None:
        direction = ("above" if delta > _BACKGROUND_COMPARISON_TOLERANCE else
                     "below" if delta < -_BACKGROUND_COMPARISON_TOLERANCE else "equal")
    return {
        "teacher_minus_background_P50": delta,
        "teacher_vs_background_P50": direction,
        "relative_to_background_exact_direction": direction,
        "relative_to_background_near_equal": abs(delta) <= _BACKGROUND_COMPARISON_TOLERANCE if delta is not None else None,
        "relative_to_background_tolerance": _BACKGROUND_COMPARISON_TOLERANCE,
        "relative_to_background_is_statistical_significance": False,
    }


def _validate_feature_names(features: Sequence[str]) -> list[str]:
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("features must be a sequence of 1 through 12 literal feature keys")
    names = list(features)
    if not 1 <= len(names) <= 12:
        raise ValueError("features must contain 1 through 12 keys")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("features must be literal string feature keys")
    if len(set(names)) != len(names):
        raise ValueError("features must not contain duplicate keys")
    for name in names:
        if not isinstance(name, str) or not name.startswith(tuple(rep + "_" for rep in _REPRESENTATIONS)):
            raise ValueError(f"feature must be an existing current_/forecast_/delta_ scalar key: {name!r}")
        if (any(token in name.lower() for token in _FORBIDDEN_TOKENS) or
                re.search(r"(^|_)id($|_)", name.lower())):
            raise ValueError(f"feature is not available to Scientist: {name!r}")
    return names


def scientist_feature_summary(source: Any, interval: int, features: Sequence[str]) -> dict[str, Any]:
    """Summarize requested scalar features across exactly one six-step window.

    The source follows ``orchestrate.teacher_packet``: 30 anonymous teacher
    donors and per-step background quantiles. No outcomes, candidate mappings,
    source paths, batch/slot identifiers, or natural-language judgments are
    returned. Only donor observations and background rows at the interval's
    six steps are read.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 10:
        raise ValueError("interval must be an integer from 1 through 10")
    names = _validate_feature_names(features)
    lo, hi = 5 * (interval - 1), 5 * interval
    mid = lo + (hi - lo) // 2
    donors, background = _teacher_background(_source_obj(source))

    # Retain only requested steps; future rows and their values are untouched.
    needed_steps = set(range(lo, hi + 1))
    background_by_step = {
        int(row["step"]): row for row in background
        if isinstance(row, Mapping) and row.get("step") in needed_steps
    }
    background_values: dict[tuple[int, str], tuple[float | None, float | None, float | None]] = {}
    for step in range(lo, hi + 1):
        row = background_by_step.get(step)
        for feature in names:
            background_values[(step, feature)] = _quantile_triplet(row, feature)

    values_by_feature_step: dict[tuple[str, int], list[float]] = {
        (feature, step): [] for feature in names for step in range(lo, hi + 1)
    }
    donor_profiles: dict[str, list[dict[str, Any]]] = {feature: [] for feature in names}
    donor_ids = set()
    for donor in donors:
        if not isinstance(donor, Mapping):
            raise ValueError("teacher donor must be an object")
        donor_id = donor.get("donor_id")
        if not isinstance(donor_id, str) or not donor_id.startswith("T") or donor_id in donor_ids:
            raise ValueError("teacher donor_id must be a unique opaque T identifier")
        donor_ids.add(donor_id)
        observations = donor.get("observations")
        if not isinstance(observations, list) or len(observations) <= hi:
            raise ValueError(f"donor {donor_id}: missing bounded observations through step {hi}")
        # No indexing occurs beyond hi. Missing, vector, bool, or nonfinite
        # values are rejected with bounded donor/step/feature diagnostics.
        for feature in names:
            values = []
            for step in range(lo, hi + 1):
                observation = observations[step]
                if not isinstance(observation, Mapping) or feature not in observation:
                    raise ValueError(f"donor {donor_id} step {step}: feature {feature!r} is missing")
                value = _finite_scalar(observation[feature])
                if value is None:
                    raise ValueError(f"donor {donor_id} step {step}: feature {feature!r} must be finite scalar numeric")
                values.append(value)
                values_by_feature_step[(feature, step)].append(value)
            directions = [_direction(a, b) for a, b in zip(values, values[1:])]
            donor_profiles[feature].append({
                "donor_id": donor_id,
                "start_step": lo,
                "mid_step": mid,
                "end_step": hi,
                "start": values[0],
                "mid": values[mid - lo],
                "end": values[-1],
                "net_delta": values[-1] - values[0],
                "net_direction": _direction(values[0], values[-1]),
                "adjacent_directions": directions,
                "pattern": _pattern(directions),
            })

    cohort = []
    for step in range(lo, hi + 1):
        for feature in names:
            values = values_by_feature_step[(feature, step)]
            p25, p50, p75 = background_values[(step, feature)]
            comparison = _background_contrast(float(np.median(values)) if values else None, p50)
            cohort.append({
                "step": step,
                "feature": feature,
                "teacher_median": float(np.median(values)),
                "background_median": p50,
                "background_P25": p25,
                "background_P75": p75,
                "teacher_count_below_background_P25": sum(v < p25 for v in values) if p25 is not None else None,
                "teacher_count_inside_background_P25_P75": sum(p25 <= v <= p75 for v in values)
                if p25 is not None and p75 is not None else None,
                "teacher_count_above_background_P75": sum(v > p75 for v in values) if p75 is not None else None,
                "teacher_count_nonnull": len(values),
                **comparison,
            })
    return {
        "interval": interval,
        "start_step": lo,
        "mid_step": mid,
        "end_step": hi,
        "features": [{"feature": feature, "donors": donor_profiles[feature]} for feature in names],
        "cohort_background": cohort,
    }
