"""Literal, source-bound validation for per-donor Scientist descriptions."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_REPRESENTATIONS = ("current", "forecast")


def _load_source(source: Any) -> Mapping[str, Any]:
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise TypeError("source must be a teacher packet mapping or JSON path")
    return source


def _donors(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    teacher = source.get("teacher", source)
    if not isinstance(teacher, Mapping) or not isinstance(teacher.get("donors"), list):
        raise ValueError("source must contain teacher.donors")
    donors = teacher["donors"]
    if len(donors) != 30:
        raise ValueError(f"expected exactly 30 teacher donors, found {len(donors)}")
    out = {}
    for donor in donors:
        donor_id = donor.get("donor_id")
        if not isinstance(donor_id, str) or not donor_id:
            raise ValueError("teacher donor is missing an opaque donor_id")
        if donor_id in out:
            raise ValueError(f"duplicate teacher donor_id {donor_id}")
        out[donor_id] = donor
    return list(out.values())


def _description_rows(descriptions: Any, interval: int) -> list[Mapping[str, Any]]:
    if isinstance(descriptions, Mapping):
        descriptions = descriptions.get("interval_descriptions")
    if not isinstance(descriptions, list):
        raise ValueError("descriptions must be a list or Scientist answer mapping")
    rows = [row for row in descriptions if isinstance(row, Mapping) and row.get("interval") == interval]
    if len(rows) != 30:
        raise ValueError(f"interval {interval}: expected 30 donor descriptions, found {len(rows)}")
    return rows


def _numeric_tree(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_numeric_tree(v) for v in value)
    return False


def _nested_numbers(value: Any) -> list[float] | None:
    if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
        number = float(value)
        return [number] if math.isfinite(number) else None
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            sub = _nested_numbers(item)
            if sub is None:
                return None
            flattened.extend(sub)
        return flattened
    return None


def _sign_code(a: float, b: float) -> int:
    delta = b - a
    return 1 if delta > 1e-6 else (-1 if delta < -1e-6 else 0)


def _window_signature(donor: Mapping[str, Any], lo: int, hi: int) -> str:
    """Qualitative trend and contact-footprint signature for one bounded window."""
    observations = donor.get("observations")
    if not isinstance(observations, list) or len(observations) <= hi:
        raise ValueError(f"donor {donor.get('donor_id')}: source lacks observations through step {hi}")
    window = observations[lo:hi + 1]
    all_features = set().union(*(set(o) for o in window if isinstance(o, Mapping)))
    trend_codes = {}
    for key in sorted(all_features):
        if not key.startswith(("current_", "forecast_")) or key.endswith("residue_contact_footprint"):
            continue
        # Principal-axis sign is arbitrary and must not count as a qualitative
        # trajectory change.
        if key.endswith("principal_axis"):
            continue
        vectors = [_nested_numbers(o.get(key)) if isinstance(o, Mapping) else None for o in window]
        if not any(v is not None for v in vectors):
            continue
        shape = next((len(v) for v in vectors if v is not None), None)
        codes = []
        for left, right in zip(vectors, vectors[1:]):
            if left is None or right is None or len(left) != len(right) or len(left) != shape:
                codes.append(None)
            else:
                codes.append([_sign_code(a, b) for a, b in zip(left, right)])
        trend_codes[key] = codes

    footprints = []
    for observation in window:
        if not isinstance(observation, Mapping):
            raise ValueError(f"donor {donor.get('donor_id')}: observation is not an object")
        state = {}
        for rep in _REPRESENTATIONS:
            key = f"{rep}_residue_contact_footprint"
            if key in observation:
                value = observation[key]
                members = [value] if isinstance(value, str) else (value or [])
                state[key] = sorted(set(str(x) for x in members if x is not None))
        footprints.append(state)
    return json.dumps({"trends": trend_codes, "footprints": footprints}, sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _normalized_paragraph(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("each donor description must have nonempty text")
    text = unicodedata.normalize("NFKC", text).casefold()
    # Strip donor labels and clock spellings before comparing paragraph reuse.
    text = re.sub(r"(?<![A-Za-z0-9])T\d+(?![A-Za-z0-9])", "<donor>", text, flags=re.IGNORECASE)
    text = re.sub(r"第\s*\d+\s*步", "第#步", text)
    text = re.sub(r"步\s*\d+", "步#", text)
    text = re.sub(r"\b(?:step|t)\s*[-:#=]?\s*\d+\b", "step#", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", text)


def _check_evidence(row: Mapping[str, Any], donor: Mapping[str, Any], lo: int, hi: int) -> None:
    donor_id = str(row.get("donor_id"))
    observations = donor["observations"]
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError(f"donor {donor_id} interval {row.get('interval')}: evidence must be a list")
    distinct: dict[str, set[str]] = {rep: set() for rep in _REPRESENTATIONS}
    steps_by_rep: dict[str, set[int]] = {rep: set() for rep in _REPRESENTATIONS}
    for item in evidence:
        if not isinstance(item, Mapping):
            raise ValueError(f"donor {donor_id}: evidence item must be an object")
        feature = item.get("feature")
        step = item.get("step")
        if not isinstance(feature, str):
            raise ValueError(f"donor {donor_id}: evidence feature must be a literal key")
        if isinstance(step, bool) or not isinstance(step, int) or not lo <= step <= hi:
            raise ValueError(f"donor {donor_id} interval {row.get('interval')}: citation step {step!r} is outside {lo}..{hi}")
        rep = next((prefix for prefix in _REPRESENTATIONS if feature.startswith(prefix + "_")), None)
        if rep is None:
            raise ValueError(f"donor {donor_id} step {step}: feature {feature!r} is not a current_/forecast_ key")
        observation = observations[step]
        if feature not in observation:
            raise ValueError(f"donor {donor_id} step {step}: feature {feature!r} is absent from source observation")
        value = observation[feature]
        if _numeric_tree(value) is False and value is not None and not feature.endswith("residue_contact_footprint"):
            raise ValueError(f"donor {donor_id} step {step}: feature {feature!r} is not a numeric source feature")
        if "observed_value" in item:
            observed = item["observed_value"]
            if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                if isinstance(observed, bool) or not isinstance(observed, (int, float, np.number)) or not math.isclose(
                    float(observed), float(value), rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(f"donor {donor_id} step {step}: observed_value for {feature!r} disagrees with source")
            elif observed != value:
                raise ValueError(f"donor {donor_id} step {step}: observed_value for {feature!r} disagrees with source")
        distinct[rep].add(feature[len(rep) + 1:])
        steps_by_rep[rep].add(step)

    for rep in _REPRESENTATIONS:
        if len(distinct[rep]) < 3:
            raise ValueError(f"donor {donor_id} interval {row.get('interval')}: need at least 3 distinct {rep} feature types")
        if lo not in steps_by_rep[rep] or hi not in steps_by_rep[rep]:
            raise ValueError(f"donor {donor_id} interval {row.get('interval')}: {rep} evidence must cite both steps {lo} and {hi}")


def _pattern(values: list[float | None]) -> str:
    deltas = [b - a for a, b in zip(values, values[1:]) if a is not None and b is not None]
    if not deltas:
        return "mixed"
    positive = any(delta > 1e-6 for delta in deltas)
    negative = any(delta < -1e-6 for delta in deltas)
    if positive and negative:
        return "mixed"
    if positive:
        return "increase"
    if negative:
        return "decrease"
    return "flat"


def _check_measured_changes(row: Mapping[str, Any], donor: Mapping[str, Any], lo: int, mid: int, hi: int) -> None:
    donor_id = str(row.get("donor_id"))
    measured = row.get("measured_changes")
    if not isinstance(measured, list):
        raise ValueError(f"donor {donor_id} interval {row.get('interval')}: measured_changes must be an array")
    if len(measured) < 4:
        raise ValueError(f"donor {donor_id} interval {row.get('interval')}: measured_changes needs at least 4 scalar features")
    observations = donor["observations"]
    seen = set()
    representations = set()
    has_pocket_feature = False
    for item in measured:
        if not isinstance(item, Mapping):
            raise ValueError(f"donor {donor_id}: measured_changes item must be an object")
        feature = item.get("feature")
        if not isinstance(feature, str) or not feature.startswith(("current_", "forecast_")):
            raise ValueError(f"donor {donor_id}: measured_changes feature must be a current_/forecast_ key")
        if feature in seen:
            raise ValueError(f"donor {donor_id}: duplicate measured_changes feature {feature!r}")
        seen.add(feature)
        rep = "current" if feature.startswith("current_") else "forecast"
        representations.add(rep)
        if any(term in feature.lower() for term in ("pocket", "contact")):
            has_pocket_feature = True
        source_values = []
        for step in range(lo, hi + 1):
            observation = observations[step]
            if feature not in observation:
                raise ValueError(f"donor {donor_id} step {step}: measured feature {feature!r} is absent from source")
            value = observation[feature]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                                      or not math.isfinite(float(value))):
                raise ValueError(f"donor {donor_id} step {step}: measured feature {feature!r} is not scalar numeric/null")
            source_values.append(None if value is None else float(value))
        for label, step in (("start_value", lo), ("mid_value", mid), ("end_value", hi)):
            observed = item.get(label, "__missing__")
            expected = source_values[step - lo]
            if expected is None:
                if observed is not None:
                    raise ValueError(f"donor {donor_id} step {step}: {feature}.{label} must be null to match source")
            elif (isinstance(observed, bool) or not isinstance(observed, (int, float, np.number)) or
                  not math.isclose(float(observed), expected, rel_tol=1e-5, abs_tol=1e-6)):
                raise ValueError(f"donor {donor_id} step {step}: {feature}.{label} disagrees with source")
        actual_pattern = _pattern(source_values)
        if item.get("pattern") != actual_pattern:
            raise ValueError(f"donor {donor_id}: {feature}.pattern must be {actual_pattern!r} from source interval")
    if not {"current", "forecast"}.issubset(representations):
        raise ValueError(f"donor {donor_id} interval {row.get('interval')}: measured_changes needs current and forecast features")
    if not has_pocket_feature:
        raise ValueError(f"donor {donor_id} interval {row.get('interval')}: measured_changes needs a pocket/contact feature")


def validate_donor_descriptions(descriptions: Any, source: Any, interval: int) -> dict[str, Any]:
    """Validate one window's donor prose citations and identical-text reuse.

    Validation is literal: citations must match existing features and exact
    in-window steps, optional observed values must match source, and copied
    paragraphs are rejected only when their source-visible window states
    differ. No language interpretation, biological mechanism, outcome, or
    ranking is inferred here.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 10:
        raise ValueError("interval must be an integer from 1 through 10")
    lo, hi = 5 * (interval - 1), 5 * interval
    donor_rows = _donors(_load_source(source))
    by_id = {row["donor_id"]: row for row in donor_rows}
    descriptions_for_window = _description_rows(descriptions, interval)
    description_by_id: dict[str, Mapping[str, Any]] = {}
    for row in descriptions_for_window:
        donor_id = row.get("donor_id")
        if donor_id not in by_id:
            raise ValueError(f"interval {interval}: unknown donor_id {donor_id!r}")
        if donor_id in description_by_id:
            raise ValueError(f"interval {interval}: duplicate description for donor {donor_id}")
        description_by_id[donor_id] = row
        _check_evidence(row, by_id[donor_id], lo, hi)
        _check_measured_changes(row, by_id[donor_id], lo, lo + (hi - lo) // 2, hi)
    if set(description_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(description_by_id))
        raise ValueError(f"interval {interval}: missing donor descriptions {missing[:3]}")

    # Detect literal paragraph reuse after normalizing donor identifiers and
    # common clock spellings. Exact observation clones may legitimately share it.
    signatures = {donor_id: _window_signature(donor, lo, hi) for donor_id, donor in by_id.items()}
    text_owners: dict[str, list[str]] = {}
    for donor_id, row in description_by_id.items():
        text_owners.setdefault(_normalized_paragraph(row.get("text")), []).append(donor_id)
    clone_pairs = 0
    for text, owners in text_owners.items():
        for i, left in enumerate(owners):
            for right in owners[i + 1:]:
                if signatures[left] != signatures[right]:
                    raise ValueError(
                        f"interval {interval}: identical paragraph reused for donors {left} and {right}, "
                        "but their source-visible qualitative evolution differs"
                    )
                clone_pairs += 1
    return {
        "interval": interval,
        "start_step": lo,
        "end_step": hi,
        "donors_validated": len(description_by_id),
        "identical_paragraph_pairs_same_qualitative_evolution": clone_pairs,
        "literal_evidence_validated": True,
        "semantic_claims_interpreted": False,
    }
