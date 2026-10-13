from __future__ import annotations

import hashlib
import json
import statistics

import pytest

from evomolsteer.lineage_retrieval import science_audit


def _source():
    donors = []
    for donor_ix in range(30):
        observations = []
        for step in range(51):
            row = {"step": step, "time": step / 100}
            for representation, offset in (("current", 0.0), ("forecast", 0.2)):
                row.update({
                    f"{representation}_radius_gyration_A": 4.0 + donor_ix / 100 + step / 100 + offset,
                    f"{representation}_centroid_pocket_centroid_A": 12.0 + donor_ix / 10 + step / 20 + offset,
                    f"{representation}_nearest_pocket_A": 5.0 + donor_ix / 100 + step / 50 + offset,
                    f"{representation}_pocket_contacts_4A": 2 + donor_ix % 3 + step // 5,
                    f"{representation}_contact_pairs_4A": 4 + donor_ix % 4 + step,
                    f"{representation}_planarity_fraction": .1 + donor_ix / 1000,
                    f"{representation}_asphericity": 2.0 + donor_ix / 100,
                    f"{representation}_residue_contact_footprint": ["ASN117", "VAL116" if step % 2 else "ASP115"],
                    f"{representation}_centroid_x_A": 1.0 + step / 10 + donor_ix / 100,
                    f"{representation}_centroid_y_A": 2.0 + donor_ix / 100,
                    f"{representation}_centroid_z_A": 3.0,
                })
            observations.append(row)
        donors.append({"donor_id": f"T{donor_ix + 1:02d}", "observations": observations})
    bg_keys = [f"{rep}_{feature}" for rep in ("current", "forecast") for feature in (
        "radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A", "pocket_contacts_4A",
        "contact_pairs_4A", "planarity_fraction", "asphericity")]
    background = [{"step": step, "quantiles_P25_P50_P75": {key: [1.0, 2.0, 3.0] for key in bg_keys}}
                  for step in range(51)]
    return {"teacher": {"donors": donors}, "background": background}


def _submission(source, interval=4):
    low, high = (interval - 1) * 5, interval * 5
    descriptions = []
    for donor in source["teacher"]["donors"]:
        obs = {r["step"]: r for r in donor["observations"]}
        feature = "current_contact_pairs_4A"
        descriptions.append({"donor_id": donor["donor_id"], "interval": interval,
            "text": "该 donor 的当前接触对在本区间逐步增加，接触残基足迹在中段后发生变化。",
            "evidence": [{"step": high, "feature": feature, "claim": "接触对数在区间末端高于起点。"}],
            "measured_changes": [{"feature": feature, "start_value": obs[low][feature],
                "mid_value": obs[low + 2][feature], "end_value": obs[high][feature], "pattern": "increase"}]})
    bg_evidence = []
    for step in (low, high):
        values = [d["observations"][step]["current_contact_pairs_4A"] for d in source["teacher"]["donors"]]
        teacher_median = statistics.median(values)
        bg_evidence.append({"step": step, "feature": "current_contact_pairs_4A",
            "teacher_median": teacher_median, "background_median": 2.0,
            "evidence": "teacher median from the six-step source profile and contemporaneous background"})
    hypothesis = {"interval": interval, "start_step": low, "end_step": high,
        "hypothesis": "当前接触规模在本窗口增加，但群体背景也随生成时钟变化。",
        "support_conditions": ["当前接触对上升"], "counter_conditions": ["背景同步上升时差异有限"],
        "observable_cues": ["比较窗口端点接触数", "同时检查同时间背景"],
        "minority_modes": "部分残基足迹在窗口中段变化。", "background_evidence": bg_evidence}
    return descriptions, hypothesis


def _mock_result():
    return {"passes": False, "evidence_truth": False, "natural_language": True,
        "clause_assessment": {"respected": False, "assessment": "claim exceeds the observed comparison"},
        "unsupported_claims": ["The prose claims an outward translation without a centroid comparison."],
        "correction_needed": True}


def _fake_run_job(monkeypatch, calls):
    def fake(job_dir, role, prompt, packet, schema, model, effort, **kwargs):
        calls.append({"job_dir": job_dir, "role": role, "prompt": prompt,
            "packet": packet, "schema": schema, "model": model, "effort": effort,
            "kwargs": kwargs})
        result = _mock_result()
        job_dir.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        (job_dir / "answer.json").write_bytes(raw)
        receipt = {"status": "completed", "answer_sha256": hashlib.sha256(raw).hexdigest(),
                   "role": role, "requested_model": model, "effort": effort}
        (job_dir / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return result
    monkeypatch.setattr(science_audit, "run_job", fake)


def test_source_only_critic_receives_six_states_and_source_checked_context(tmp_path, monkeypatch):
    source = _source()
    descriptions, hypothesis = _submission(source)
    calls = []
    _fake_run_job(monkeypatch, calls)
    result = science_audit.audit_scientist_submission(tmp_path / "R00" / "scientist", source, 4,
        descriptions, hypothesis, "Scientist workflow. Active clause S04: compare contact evolution with same-time background.")

    assert result["passes"] is False
    assert result["correction_needed"] is True
    assert "replacement" not in json.dumps(result, ensure_ascii=False).lower()
    assert len(calls) == 1
    call = calls[0]
    assert call["role"] == "main-agent"
    assert call["model"] == "gpt-6.1-sol" and call["effort"] == "xhigh"
    packet = call["packet"]
    assert packet["scope"]["allowed_steps"] == [15, 16, 17, 18, 19, 20]
    assert len(packet["donor_descriptions"]) == 30
    assert len(packet["computed_source_summaries"]["cohort_background"]) > 0
    assert all([r["step"] for r in d["six_current_observation_states"]] == [15, 16, 17, 18, 19, 20]
               for d in packet["donor_descriptions"])
    assert len(packet["source_checked_background_quantiles"]) == 6
    assert all(e["feature_available_at_all_checkpoints"] for d in packet["donor_descriptions"]
               for e in d["measured_changes"])
    encoded = json.dumps(packet, ensure_ascii=False)
    assert "node_id" not in encoded and "batch" not in encoded and "slot" not in encoded
    assert "oracle" not in encoded.lower() and "C:/" not in encoded
    assert len(encoded.encode("utf-8")) < 250_000


def test_future_mutation_changes_binding_but_not_audited_packet(tmp_path, monkeypatch):
    source = _source()
    descriptions, hypothesis = _submission(source, interval=4)
    calls = []
    _fake_run_job(monkeypatch, calls)
    job = tmp_path / "R03" / "scientist"
    prompt = "Workflow with active clause S03."
    science_audit.audit_scientist_submission(job, source, 4, descriptions, hypothesis, prompt)
    first_packet = calls[0]["packet"]
    source["teacher"]["donors"][0]["observations"][45]["current_contact_pairs_4A"] += 800
    science_audit.audit_scientist_submission(job, source, 4, descriptions, hypothesis, prompt)
    assert len(calls) == 2  # full original-source hash differs, so no stale critic is reused
    assert calls[0]["packet"] == calls[1]["packet"]  # future state is absent from both model packets
    assert first_packet["scope"]["allowed_steps"][-1] == 20
    assert len(list((job / "W04").iterdir())) == 2


def test_same_submission_reuses_hash_bound_critic_and_detects_changed_binding(tmp_path, monkeypatch):
    source = _source()
    descriptions, hypothesis = _submission(source, interval=1)
    calls = []
    _fake_run_job(monkeypatch, calls)
    job = tmp_path / "R01" / "scientist"
    prompt = "Workflow with active clause S01."
    first = science_audit.audit_scientist_submission(job, source, 1, descriptions, hypothesis, prompt)
    second = science_audit.audit_scientist_submission(job, source, 1, descriptions, hypothesis, prompt)
    assert first == second
    assert len(calls) == 1
    binding = next((job / "W01").glob("*/controller_binding.json"))
    changed = json.loads(binding.read_text(encoding="utf-8"))
    changed["source_sha256"] = "tampered"
    binding.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="binding differs"):
        science_audit.audit_scientist_submission(job, source, 1, descriptions, hypothesis, prompt)


def test_rejects_wrong_interval_or_incomplete_donor_submission(tmp_path):
    source = _source()
    descriptions, hypothesis = _submission(source, interval=4)
    with pytest.raises(ValueError, match="30 private descriptions"):
        science_audit.audit_scientist_submission(tmp_path, source, 4, descriptions[:29], hypothesis, "active")
    with pytest.raises(ValueError, match="match the audited interval"):
        science_audit.audit_scientist_submission(tmp_path, source, 5, descriptions, hypothesis, "active")
