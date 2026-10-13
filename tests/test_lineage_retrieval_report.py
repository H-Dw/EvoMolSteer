from __future__ import annotations

import hashlib
import json

from evomolsteer.lineage_retrieval.report import (
    _functional_job_status,
    _selection_run_complete,
    _round_audit_status,
    report,
)


def _functional_job(job, clause="Q", request="request-1"):
    job.mkdir(parents=True, exist_ok=True)
    answer = job / "answer.json"
    answer.write_text('{"answer":"current"}', encoding="utf-8")
    digest = hashlib.sha256(json.dumps({"answer": "current"}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    (job / "controller_binding.json").write_text(json.dumps({"request_sha256": request}), encoding="utf-8")
    (job / "functional_audit.json").write_text(json.dumps({"clause_id": clause, "request_sha256": request,
        "prompt_contains_active_clause": True, "application_claim_not_a_causal_improvement": True}), encoding="utf-8")
    (job / "receipt.json").write_text(json.dumps({"status": "completed", "answer_sha256": digest}), encoding="utf-8")


def _semantic(**updates):
    return {"passes": True, "scientist_evidence_truth": True, "scientist_clause_applied": True,
        "selector_assessments": [{"step": step, "evidence_truth": True, "clause_applied": True}
                                  for step in range(5, 51, 5)], **updates}


def _complete_metrics():
    steps = list(range(5, 51, 5))
    return {"complete": True, "evaluated_steps": steps, "metrics": [{"step": step} for step in steps]}


def test_functional_audit_must_match_binding_and_receipt(tmp_path):
    job = tmp_path / "job"
    _functional_job(job)
    assert _functional_job_status(job) == (True, "")

    audit = json.loads((job / "functional_audit.json").read_text())
    audit["request_sha256"] = "stale-request"
    (job / "functional_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    valid, reason = _functional_job_status(job)
    assert not valid
    assert "does not match controller binding" in reason

    _functional_job(job, request="request-2")
    receipt = json.loads((job / "receipt.json").read_text())
    receipt["answer_sha256"] = "stale-answer"
    (job / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    valid, reason = _functional_job_status(job)
    assert not valid
    assert "current answer" in reason


def test_semantic_audit_requires_explicit_scientist_truth_and_application(tmp_path):
    study = tmp_path / "study"
    round_dir = study / "rounds" / "R00"
    source = round_dir / "scientist"
    _functional_job(source, clause="S")
    for step in range(5, 51, 5):
        _functional_job(round_dir / "selectors" / f"t{step:03d}")
    (round_dir / "_version.json").write_text(json.dumps({}), encoding="utf-8")
    (round_dir / "semantic_audit.json").write_text(json.dumps(_semantic(scientist_clause_applied=False)), encoding="utf-8")
    status = _round_audit_status(study, round_dir)
    assert not status["complete"]
    assert not status["semantic_complete"]

    legacy = _semantic()
    del legacy["scientist_evidence_truth"]
    (round_dir / "semantic_audit.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert not _round_audit_status(study, round_dir)["semantic_complete"]


def test_repeat_validation_requires_ten_bound_actual_cutoff_calls(tmp_path):
    run = tmp_path / "repeat"
    for step in range(5, 51, 5):
        _functional_job(run / "selectors" / f"t{step:03d}")
    metrics = _complete_metrics()
    assert _selection_run_complete(run, metrics) == (True, [])

    missing = run / "selectors" / "t050" / "answer.json"
    missing.unlink()
    valid, issues = _selection_run_complete(run, metrics)
    assert not valid
    assert any("t050" in issue for issue in issues)

    duplicate_cutoff = _complete_metrics()
    duplicate_cutoff["metrics"][-1]["step"] = 45
    valid, issues = _selection_run_complete(run, duplicate_cutoff)
    assert not valid
    assert any("ten unique expected cutoffs" in issue for issue in issues)


def test_report_surfaces_rejected_trials_without_scoring_them(tmp_path):
    study = tmp_path / "study"
    rejected = study / "rounds" / "R00" / "rejected" / "T01"
    rejected.mkdir(parents=True)
    manifest = {"reason": "xhigh rejected generic Scientist descriptions", "files": [
        {"source": "metrics.json", "sha256": "abc"}, {"source": "semantic_audit.json", "sha256": "def"}]}
    (rejected / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (study / "run_status.json").write_text(json.dumps({"status": "repairing"}), encoding="utf-8")

    result = report(study)
    assert result["status"] == "incomplete"
    assert len(result["rejected_trials"]) == 1
    assert result["rejected_trials"][0]["contains_rejected_metrics"] is True
    assert result["complete_rounds"] == []
    assert "rejected_trials.csv" in result["artifacts"]
    text = (study / "report.md").read_text(encoding="utf-8")
    assert "R00" in text and "xhigh rejected generic Scientist descriptions" in text
