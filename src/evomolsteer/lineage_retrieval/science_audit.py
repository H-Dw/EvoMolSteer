"""Source-only xhigh critique of one Scientist interval submission.

This module is intentionally standalone: callers may run it before committing a
Scientist interval. It sends no oracle membership, terminal labels, future
states, source paths, or rewritten scientific prose to the auditor.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from .summaries import summarize_scientist_interval


MODEL = "gpt-6.1-sol"
EFFORT = "xhigh"
# Optional injection point for deterministic unit tests. The production runner
# is imported lazily inside the audit function to avoid a module cycle.
run_job = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(_canonical(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _schema() -> dict[str, Any]:
    obj = lambda props, required=None: {"type": "object", "properties": props,
        "required": list(props) if required is None else required, "additionalProperties": False}
    text = {"type": "string"}
    boolean = {"type": "boolean"}
    return obj({
        "passes": boolean,
        "evidence_truth": boolean,
        "natural_language": boolean,
        "clause_assessment": obj({"respected": boolean, "assessment": text}),
        "unsupported_claims": {"type": "array", "items": text},
        "correction_needed": boolean,
    })


def _window_source(source: Mapping[str, Any], interval: int,
                   descriptions: list[Mapping[str, Any]], hypothesis: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    low, high = (interval - 1) * 5, interval * 5
    mid = low + (high - low) // 2
    teacher = source.get("teacher", {})
    donors = teacher.get("donors", []) if isinstance(teacher, Mapping) else []
    if not isinstance(donors, list) or len(donors) != 30:
        raise ValueError("source must contain exactly 30 teacher donors")
    by_id: dict[str, Mapping[str, Any]] = {}
    for donor in donors:
        donor_id = donor.get("donor_id") if isinstance(donor, Mapping) else None
        observations = donor.get("observations") if isinstance(donor, Mapping) else None
        if not isinstance(donor_id, str) or donor_id in by_id or not isinstance(observations, list):
            raise ValueError("source donors must have unique IDs and observation arrays")
        by_id[donor_id] = donor
    if not isinstance(descriptions, list) or len(descriptions) != 30:
        raise ValueError("submission must contain 30 private descriptions for this interval")
    desc_by_id = {}
    for row in descriptions:
        if not isinstance(row, Mapping) or row.get("interval") != interval:
            raise ValueError("every private description must belong to the audited interval")
        donor_id = row.get("donor_id")
        if donor_id not in by_id or donor_id in desc_by_id:
            raise ValueError("descriptions must name each source donor exactly once")
        desc_by_id[donor_id] = row
    if set(desc_by_id) != set(by_id):
        raise ValueError("descriptions must cover every source donor")

    core_keys = {
        f"{rep}_{feature}"
        for rep in ("current", "forecast")
        for feature in (
            "radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A",
            "pocket_contacts_4A", "contact_pairs_4A", "planarity_fraction",
            "asphericity", "residue_contact_footprint", "centroid_x_A", "centroid_y_A", "centroid_z_A",
        )
    }
    rows = []
    queried_features: set[str] = set()
    for donor_id in sorted(by_id):
        submission = desc_by_id[donor_id]
        observed = by_id[donor_id]["observations"]
        state_by_step = {int(row["step"]): row for row in observed
                         if isinstance(row, Mapping) and isinstance(row.get("step"), int)
                         and low <= row["step"] <= high}
        if set(state_by_step) != set(range(low, high + 1)):
            raise ValueError(f"source donor {donor_id} is missing one of the six allowed interval states")
        requested = set(core_keys)
        for claim in submission.get("measured_changes", []):
            if isinstance(claim, Mapping) and isinstance(claim.get("feature"), str):
                requested.add(claim["feature"])
        for claim in submission.get("evidence", []):
            if isinstance(claim, Mapping) and isinstance(claim.get("feature"), str):
                requested.add(claim["feature"])
        requested = {key for key in requested if key.startswith(("current_", "forecast_", "delta_"))}
        queried_features.update(requested)
        observed_rows = []
        for step in range(low, high + 1):
            state = state_by_step[step]
            observed_rows.append({"step": step, **{key: state.get(key) for key in sorted(requested) if key in state}})
        source_checked_changes = []
        for claim in submission.get("measured_changes", []):
            if not isinstance(claim, Mapping):
                continue
            feature = claim.get("feature")
            if not isinstance(feature, str):
                continue
            actual = {"start_value": state_by_step[low].get(feature),
                      "mid_value": state_by_step[mid].get(feature),
                      "end_value": state_by_step[high].get(feature)}
            source_checked_changes.append({"submission": dict(claim), "actual_source_values": actual,
                "feature_available_at_all_checkpoints": all(feature in state_by_step[s] for s in (low, mid, high))})
        checked_evidence = []
        for claim in submission.get("evidence", []):
            if not isinstance(claim, Mapping):
                checked_evidence.append({"submission": claim, "source_value": None, "source_match": False})
                continue
            step, feature = claim.get("step"), claim.get("feature")
            state = state_by_step.get(step) if isinstance(step, int) else None
            exists = bool(state is not None and isinstance(feature, str) and feature in state)
            checked_evidence.append({"submission": dict(claim), "source_value": state.get(feature) if exists else None,
                                     "source_match": exists})
        rows.append({"donor_id": donor_id, "text": submission.get("text", ""),
            "evidence": checked_evidence, "measured_changes": source_checked_changes,
            "six_current_observation_states": observed_rows})

    background_rows = source.get("background", [])
    if not isinstance(background_rows, list):
        raise ValueError("source background must be point-indexed")
    by_step = {row.get("step"): row for row in background_rows if isinstance(row, Mapping)
               and isinstance(row.get("step"), int) and low <= row["step"] <= high}
    if set(by_step) != set(range(low, high + 1)):
        raise ValueError("source background must provide all six audited interval steps")
    # Keep the actual numeric P25/P50/P75 values at these six cutoffs only.
    # The records contain no batch/slot/node identifiers or source locators.
    background = []
    bg_features = set(queried_features) | {
        f"{rep}_{feature}" for rep in ("current", "forecast")
        for feature in ("radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A",
                        "pocket_contacts_4A", "contact_pairs_4A", "planarity_fraction", "asphericity")
    }
    hypothesis_features = {r.get("feature") for r in hypothesis.get("background_evidence", [])
                           if isinstance(r, Mapping) and isinstance(r.get("feature"), str)
                           and r["feature"].startswith(("current_", "forecast_", "delta_"))}
    bg_features.update(hypothesis_features)
    for step in range(low, high + 1):
        raw = by_step[step].get("quantiles_P25_P50_P75", {})
        quantiles = {}
        if isinstance(raw, Mapping):
            for feature, values in raw.items():
                if feature not in bg_features or not isinstance(values, (list, tuple)) or len(values) < 3:
                    continue
                quantiles[feature] = [_number(value) for value in values[:3]]
        background.append({"step": step, "quantiles_P25_P50_P75": quantiles})

    summary = summarize_scientist_interval(source, interval)
    # Donor-level derived summaries duplicate the six raw observations below
    # and can make a valid source packet needlessly large. Keep the cohort
    # comparison, which is the part not reconstructible from one donor row.
    summary = {"interval": interval, "start_step": low, "mid_step": mid, "end_step": high,
        "cohort_background": summary["cohort_background"]}
    actual_hypothesis_evidence = []
    for claim in hypothesis.get("background_evidence", []):
        if not isinstance(claim, Mapping):
            actual_hypothesis_evidence.append({"submission": claim, "source_match": False})
            continue
        step, feature = claim.get("step"), claim.get("feature")
        source_row = by_step.get(step) if isinstance(step, int) else None
        quantiles = source_row.get("quantiles_P25_P50_P75", {}) if source_row else {}
        raw = quantiles.get(feature) if isinstance(quantiles, Mapping) else None
        values = raw if isinstance(raw, (list, tuple)) else []
        teacher_values = []
        for donor in donors:
            state = next((r for r in donor["observations"] if r.get("step") == step), None)
            val = _number(state.get(feature)) if state and isinstance(feature, str) else None
            if val is not None:
                teacher_values.append(val)
        actual_teacher = float(statistics.median(teacher_values)) if len(teacher_values) == 30 else None
        actual_bg = _number(values[1]) if len(values) >= 2 else None
        actual_hypothesis_evidence.append({"submission": dict(claim),
            "actual_teacher_median": actual_teacher, "actual_background_P50": actual_bg,
            "source_match": actual_teacher is not None and actual_bg is not None
                and _number(claim.get("teacher_median")) == actual_teacher
                and _number(claim.get("background_median")) == actual_bg})

    hypothesis_fields = ("interval", "start_step", "end_step", "hypothesis", "support_conditions",
                         "counter_conditions", "observable_cues", "minority_modes", "background_evidence")
    payload = {
        "role": "main-agent",
        "audit_kind": "scientist-source-only-science-critic",
        "scope": {"interval": interval, "allowed_steps": list(range(low, high + 1)),
                  "only_current_six_states": True, "source_future_states_included": False,
                  "terminal_scores_included": False},
        "scientist_prompt_and_active_clause": "",  # Filled by caller; kept separate from all source values.
        "current_hypothesis": {key: hypothesis[key] for key in hypothesis_fields if key in hypothesis},
        "computed_source_summaries": summary,
        "source_checked_background_quantiles": background,
        "source_checked_hypothesis_evidence": actual_hypothesis_evidence,
        "donor_descriptions": rows,
        "source_features_queried": sorted(queried_features),
    }
    return payload, sorted(queried_features)


AUDIT_PROMPT = """Act as a strict scientific evidence critic for one submitted Scientist interval. Return critique only; do not rewrite, complete, or suggest replacement Scientist prose. Do not access external sources.

Check every material statement in each Chinese private paragraph against that donor's six source states and the cohort/background values shown. Check the hypothesis against the same interval's six steps. Treat the attached Scientist prompt as the operative policy and check whether its active clause was concretely respected or applied in this interval; for a conditional clause, a grounded statement that its condition is absent can satisfy it. Do not invent a pattern to force clause application.

Use measured values and sign conventions carefully. Compare contact-pair counts, pocket-contact counts, atom counts, and residue footprints as distinct quantities. Distinguish nearest-pocket minimum distance from centroid-to-pocket distance. Whole-cloud centroid translation is supported only by fixed-frame centroid coordinates; a nearest-distance change alone cannot establish outward/inward translation. A sign flip in a principal axis is not a physical rotation. Compare the actual teacher cohort with same-time background medians/quantiles before claiming a teacher-specific mode or an equal endpoint. Look at intermediate states and reversals, not just interval endpoints.

Paragraphs must be ordinary qualitative Chinese prose, not feature-key lists, pattern-code repetition, or generic filler. Evidence and measured_changes are structured metadata and do not make unsupported prose true. Do not infer formed molecules, chemical validity, energies, hydrogen bonds, causal affinity effects, rankings, final outcomes, or prospective discovery from these computational observations. No terminal labels, oracle membership, node IDs, batches, slots, file paths, or observations after this interval are provided or in scope.

Set evidence_truth=false if a material factual claim conflicts with the supplied source or is unsupported. Set natural_language=false for non-prose or generic template prose that fails to communicate this donor/window's observed behavior. clause_assessment.respected should reflect observable grounded application of the active clause, not copied clause language. passes is true only if evidence_truth and natural_language and clause_assessment.respected are all true with no material unsupported claims. correction_needed must be true when a factual, language-quality, or in-scope clause issue should be repaired. Return only JSON matching the schema."""


def audit_scientist_submission(job_dir: str | Path, source: Mapping[str, Any], interval: int,
                               descriptions: list[Mapping[str, Any]], hypothesis: Mapping[str, Any],
                               prompt: str) -> dict[str, Any]:
    """Run or retrieve a cached xhigh source-only critique for one interval.

    The returned critic response is never replacement Scientist text. Each
    distinct source/submission/prompt/code input gets a separate Wxx/hash job.
    """
    if isinstance(interval, bool) or not isinstance(interval, int) or not 1 <= interval <= 10:
        raise ValueError("interval must be an integer from 1 through 10")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("the active Scientist prompt is required for clause review")
    if not isinstance(hypothesis, Mapping) or hypothesis.get("interval") != interval:
        raise ValueError("hypothesis must match the audited interval")
    if not isinstance(source, Mapping):
        raise TypeError("source must be a mapping")

    payload, _ = _window_source(source, interval, descriptions, hypothesis)
    payload["scientist_prompt_and_active_clause"] = prompt
    # Keep the complete current-window evidence and all 30 donor paragraphs.
    # The measured R00 packets are roughly 473–498 kB UTF-8; 800 kB leaves
    # room for feature variation while remaining below the app-server's 1 MB
    # content-string limit. This is a transport guard, not a reason to truncate.
    packet_bytes = _canonical(payload)
    if len(packet_bytes) > 800_000:
        raise ValueError(f"source-only audit packet exceeds 800 kB transport budget ({len(packet_bytes)} bytes)")

    source_sha = _json_sha(source)
    submission_sha = _json_sha({"descriptions": descriptions, "hypothesis": hypothesis})
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    schema = _schema()
    request_record = {"source_sha256": source_sha, "submission_sha256": submission_sha,
        "science_audit_code_sha256": code_sha, "prompt_sha256": _sha(prompt.encode("utf-8")),
        "audit_prompt_sha256": _sha(AUDIT_PROMPT.encode("utf-8")), "packet_sha256": _sha(packet_bytes),
        "schema_sha256": _json_sha(schema), "model": MODEL, "effort": EFFORT,
        "turn_timeout_seconds": 1200, "transport_attempt_limit": 1}
    request_sha = _json_sha(request_record)
    # The full SHA-256 remains in the binding. A short directory component
    # prevents nested native-worker event/receipt filenames exceeding MAX_PATH.
    # The exact binding check below fails closed on any prefix collision.
    audit_dir = Path(job_dir) / f"W{interval:02d}" / request_sha[:16]
    audit_dir.mkdir(parents=True, exist_ok=True)
    binding_path = audit_dir / "controller_binding.json"
    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != {**request_record, "request_sha256": request_sha}:
            raise ValueError("cached Scientist audit request binding differs; preserve this payload and review it")
    else:
        _write_json(binding_path, {**request_record, "request_sha256": request_sha})
    packet_path = audit_dir / "audit_packet.json"
    if packet_path.exists():
        cached_packet = json.loads(packet_path.read_text(encoding="utf-8"))
        if _json_sha(cached_packet) != _sha(packet_bytes):
            raise ValueError("cached Scientist audit packet differs from its request binding")
    if not packet_path.exists():
        _write_json(packet_path, payload)

    answer_path = audit_dir / "answer.json"
    receipt_path = audit_dir / "receipt.json"
    if answer_path.exists() and receipt_path.exists():
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") == "completed" and receipt.get("answer_sha256") == _json_sha(answer):
            return answer
        raise ValueError("cached Scientist audit output does not match its completed receipt")

    auditor_prompt = AUDIT_PROMPT + "\n\nScientist role prompt (including current active clause):\n" + prompt
    # Import locally to keep the runner's precommit callback free of a module
    # import cycle while still using the same isolated real-model worker.
    worker = run_job
    if worker is None:
        from .runner import run_job as worker
    result = worker(audit_dir / "call", "main-agent", auditor_prompt, payload, schema,
        model=MODEL, effort=EFFORT, timeout=1200, max_attempts=1)
    if not isinstance(result, dict) or not all(key in result for key in schema["required"]):
        raise ValueError("xhigh Scientist critic returned an incomplete audit object")
    _write_json(audit_dir / "answer.json", result)
    # run_job records a byte-level answer hash in call/receipt.json. This copy
    # makes the cache integrity check deterministic across canonical JSON I/O.
    call_receipt = audit_dir / "call" / "receipt.json"
    if call_receipt.exists():
        receipt = json.loads(call_receipt.read_text(encoding="utf-8"))
        receipt["critic_answer_sha256"] = _json_sha(result)
        _write_json(receipt_path, receipt)
    else:
        raise RuntimeError("xhigh Scientist critic did not write its call receipt")
    return result
