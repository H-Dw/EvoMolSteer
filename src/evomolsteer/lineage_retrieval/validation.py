"""Data-only factual validation for progressive Scientist hypotheses."""
from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any


_CLOCK_FIELDS = {"interval", "start_step", "end_step", "step", "start", "end"}


def validate_public_hypothesis(hypothesis: dict[str, Any]) -> bool:
    """Keep source identities and measured numbers out of public language."""
    text = ' '.join([hypothesis['hypothesis'], hypothesis['minority_modes'],
        *hypothesis['support_conditions'], *hypothesis['counter_conditions'], *hypothesis['observable_cues']])
    if re.search(r'(?:(?<![A-Za-z0-9])T\d{1,3}(?![A-Za-z0-9])|single_b\d+|batch[_ ]?\d+|final[_ ]?\d+|(?<![A-Za-z0-9])C[0-9a-f]{8,}(?![A-Za-z0-9])|(?:分支|轨迹|候选|candidate|trajectory|branch)[_ #:-]*\d+)', text, re.I):
        raise ValueError('Public hypothesis contains donor or source identity; keep all provenance IDs in private description metadata only')
    for token in re.findall(r'\d+\.\d+', text):
        if float(token) not in {n / 100 for n in range(0, 51, 5)}:
            raise ValueError('Public hypothesis contains exact continuous measurements. Put numeric medians ONLY in private background_evidence and use qualitative macroscopic language in all public hypothesis fields.')
    return True


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _source_rows(source: dict[str, Any], step: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    teacher = source.get("teacher", {})
    donors = teacher.get("donors", []) if isinstance(teacher, dict) else []
    observations = []
    for donor in donors if isinstance(donors, list) else []:
        if not isinstance(donor, dict):
            continue
        for row in donor.get("observations", []) if isinstance(donor.get("observations", []), list) else []:
            if isinstance(row, dict) and row.get("step") == step:
                observations.append(row)
                break
    background = source.get("background", [])
    bgrow = next((row for row in background if isinstance(row, dict) and row.get("step") == step), None) if isinstance(background, list) else None
    return observations, bgrow


def _actual_medians(source: dict[str, Any], step: int, feature: str) -> tuple[float | None, float | None, str | None]:
    observations, bgrow = _source_rows(source, step)
    raw_teacher_values = [row.get(feature) for row in observations]
    supplied_teacher_values = [value for value in raw_teacher_values if value is not None]
    if any(isinstance(value, bool) or not _finite_number(value) for value in supplied_teacher_values):
        return None, None, "teacher source contains a boolean or nonfinite/non-numeric value"
    teacher_values = [float(value) for value in supplied_teacher_values]
    donor_count = len(source.get("teacher", {}).get("donors", [])) if isinstance(source.get("teacher"), dict) else 0
    if len(teacher_values) == donor_count and donor_count == 30:
        teacher_median: float | None = float(statistics.median(teacher_values))
    elif not teacher_values and len(observations) == donor_count:
        teacher_median = None
    else:
        teacher_median = None
    background_median: float | None = None
    bg_quantiles = bgrow.get("quantiles_P25_P50_P75", {}) if isinstance(bgrow, dict) else {}
    values = bg_quantiles.get(feature) if isinstance(bg_quantiles, dict) else None
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        p50 = values[1]
        if p50 is not None:
            if isinstance(p50, bool) or not _finite_number(p50):
                return teacher_median, None, "background P50 is boolean or nonfinite/non-numeric"
            background_median = float(p50)
    if teacher_values and len(teacher_values) != donor_count:
        return teacher_median, background_median, "teacher data are incomplete across the 30 donors"
    if not teacher_values and len(observations) != donor_count:
        return teacher_median, background_median, "source does not contain one teacher state for each donor at this step"
    return teacher_median, background_median, None


def _available_feature_keys(source: dict[str, Any], step: int) -> list[str]:
    observations, _ = _source_rows(source, step)
    if len(observations) != 30:
        return []
    common = set.intersection(*(set(row) for row in observations)) if observations else set()
    keys = []
    for key in sorted(common):
        if not key.startswith(("current_", "forecast_", "delta_")):
            continue
        if not all(_finite_number(row.get(key)) for row in observations):
            continue
        _, _, error = _actual_medians(source, step, key)
        if error:
            continue
        keys.append(key)
    return keys


def _repair_options(source: dict[str, Any], low: int, high: int, records: list[Any]) -> list[dict[str, Any]]:
    """Give only within-window, source-backed options relevant to the bad rows."""
    pairs: set[tuple[int, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        feature = record.get("feature")
        step = record.get("step")
        if isinstance(feature, str) and feature.startswith(("current_", "forecast_", "delta_")):
            if isinstance(step, int) and not isinstance(step, bool) and low <= step <= high:
                pairs.add((step, feature))
            else:
                pairs.update((candidate_step, feature) for candidate_step in range(low, high + 1)
                             if feature in _available_feature_keys(source, candidate_step))
        elif isinstance(step, int) and not isinstance(step, bool) and low <= step <= high:
            pairs.update((step, key) for key in _available_feature_keys(source, step))
        else:
            for boundary in (low, high):
                pairs.update((boundary, key) for key in _available_feature_keys(source, boundary))
    if not pairs:
        for boundary in (low, high):
            pairs.update((boundary, key) for key in _available_feature_keys(source, boundary))
    options = []
    for step, feature in sorted(pairs):
        if not low <= step <= high or not feature.startswith(("current_", "forecast_", "delta_")):
            continue
        teacher_median, background_median, error = _actual_medians(source, step, feature)
        if error or (teacher_median is None and background_median is None):
            continue
        options.append({"step": step, "feature": feature,
                        "teacher_median": teacher_median,
                        "background_median": background_median})
    return options


def _raise_evidence_error(message: str, source: dict[str, Any], low: int, high: int,
                          records: list[Any]) -> None:
    options = _repair_options(source, low, high, records)
    detail = json.dumps(options, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"{message}; allowed-time source-backed feature medians={detail}")


def validate_background_evidence(hypothesis: dict[str, Any], source: dict[str, Any]) -> bool:
    """Validate a Scientist hypothesis against teacher and same-time background data.

    Each ``background_evidence`` item has ``step``, ``feature``,
    ``teacher_median`` and ``background_median``. At least two distinct feature/
    step pairs are required, including both interval endpoints. Errors include
    source-backed repair options from this interval only.
    """
    if not isinstance(hypothesis, dict) or not isinstance(source, dict):
        raise TypeError("hypothesis and source must be JSON objects")
    interval = hypothesis.get("interval")
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 10:
        raise ValueError("hypothesis interval must be an integer from 1 through 10")
    low, high = 5 * (interval - 1), 5 * interval
    start_step, end_step = hypothesis.get("start_step"), hypothesis.get("end_step")
    if (isinstance(start_step, bool) or not isinstance(start_step, int) or start_step != low
            or isinstance(end_step, bool) or not isinstance(end_step, int) or end_step != high):
        records = hypothesis.get("background_evidence", [])
        _raise_evidence_error(f"hypothesis clock must be exactly {low}..{high}", source, low, high,
                              records if isinstance(records, list) else [])
    donors = source.get("teacher", {}).get("donors") if isinstance(source.get("teacher"), dict) else None
    if not isinstance(donors, list) or len(donors) != 30:
        raise ValueError("source must contain exactly 30 anonymous teacher donors")
    donor_ids = [row.get("donor_id") for row in donors if isinstance(row, dict)]
    if len(donor_ids) != 30 or len(set(donor_ids)) != 30:
        raise ValueError("source must contain 30 unique anonymous donor IDs")

    records = hypothesis.get("background_evidence")
    if not isinstance(records, list):
        records = []
    errors: list[str] = []
    accepted_pairs: set[tuple[int, str]] = set()
    accepted_steps: set[int] = set()
    for index, record in enumerate(records):
        error_count_before = len(errors)
        if not isinstance(record, dict):
            errors.append(f"background_evidence[{index}] must be an object")
            continue
        step, feature = record.get("step"), record.get("feature")
        if isinstance(step, bool) or not isinstance(step, int) or not low <= step <= high:
            errors.append(f"background_evidence[{index}].step must be within {low}..{high}")
            continue
        if not isinstance(feature, str) or not feature.startswith(("current_", "forecast_", "delta_")):
            errors.append(f"background_evidence[{index}].feature must be a current_/forecast_/delta_ key")
            continue
        actual_teacher, actual_background, source_error = _actual_medians(source, step, feature)
        if source_error:
            errors.append(f"background_evidence[{index}] {source_error}")
            continue
        supplied_teacher = record.get("teacher_median")
        supplied_background = record.get("background_median")
        if actual_teacher is None:
            if supplied_teacher is not None:
                errors.append(f"background_evidence[{index}].teacher_median must be null because source has no values")
            errors.append(f"background_evidence[{index}] requires numeric teacher values for all 30 donors")
        elif not _finite_number(supplied_teacher) or not math.isclose(float(supplied_teacher), actual_teacher, rel_tol=1e-5, abs_tol=1e-6):
            errors.append(f"background_evidence[{index}].teacher_median does not match the 30-donor median")
        if actual_background is None:
            if supplied_background is not None:
                errors.append(f"background_evidence[{index}].background_median must be null because source has no value")
        elif not _finite_number(supplied_background) or not math.isclose(float(supplied_background), actual_background, rel_tol=1e-5, abs_tol=1e-6):
            errors.append(f"background_evidence[{index}].background_median does not match the supplied P50")
        if len(errors) == error_count_before:
            accepted_pairs.add((step, feature))
            accepted_steps.add(step)

    if len(accepted_pairs) < 2:
        errors.append("at least two distinct valid feature/step evidence records are required")
    if low not in accepted_steps:
        errors.append(f"at least one valid evidence record is required at interval start step {low}")
    if high not in accepted_steps:
        errors.append(f"at least one valid evidence record is required at interval end step {high}")
    if errors:
        _raise_evidence_error("; ".join(errors), source, low, high, records)
    return True


def _public_bundle(hypothesis: dict[str, Any]) -> str:
    # Private numeric evidence varies with the clock; it must not disguise a
    # copied public hypothesis as a new interval-specific synthesis.
    public = {'hypothesis', 'support_conditions', 'counter_conditions', 'observable_cues', 'minority_modes'}
    bundle = {key: value for key, value in hypothesis.items() if key in public}
    text = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # Merely changing the interval label or its numeric clock does not make a
    # copied narrative an interval-specific scientific observation.
    return re.sub(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])", "<clock>", text)


def validate_hypothesis_specificity(hypotheses: list[dict[str, Any]]) -> bool:
    """Reject repeated complete public hypothesis bundles across intervals."""
    if not isinstance(hypotheses, list):
        raise TypeError("hypotheses must be a list")
    seen: dict[str, int] = {}
    repeated = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            raise ValueError(f"hypotheses[{index}] must be an object")
        key = _public_bundle(hypothesis)
        interval = hypothesis.get("interval", index + 1)
        if key in seen:
            repeated.extend((seen[key], interval))
        else:
            seen[key] = interval
    if repeated:
        intervals = sorted(set(repeated))
        raise ValueError(
            f"identical public hypothesis bundles repeat across intervals {intervals}; "
            "changing only numeric clock labels does not create a new hypothesis; describe the observed same-pattern continuation using concrete "
            "contact, shape, or location changes when present; do not invent distinctions when none are observed"
        )
    return True
