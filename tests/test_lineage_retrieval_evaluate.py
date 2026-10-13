from __future__ import annotations

import json
import hashlib

import pytest

from evomolsteer.lineage_retrieval.evaluate import coverage_at_k, evaluate_selection
from evomolsteer.lineage_retrieval.report import report


def write_functional_job(job, clause):
    job.mkdir(parents=True, exist_ok=True)
    answer_path = job / "answer.json"
    answer_path.write_text(json.dumps({"answer": "accepted"}), encoding="utf-8")
    answer_hash = hashlib.sha256(json.dumps({"answer": "accepted"}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    binding = {"request_sha256": f"request-{clause}"}
    (job / "controller_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    (job / "functional_audit.json").write_text(json.dumps({"clause_id": clause, "request_sha256": binding["request_sha256"],
        "prompt_contains_active_clause": True, "application_claim_not_a_causal_improvement": True}), encoding="utf-8")
    (job / "receipt.json").write_text(json.dumps({"status": "completed", "answer_sha256": answer_hash}), encoding="utf-8")


def fixture_pool():
    groups = [
        {"key": "g0", "batch": 0, "representative_slot": 2, "aliases": [2, 4], "targets": ["n1", "n2"], "observed_descendants": 8},
        {"key": "g1", "batch": 1, "representative_slot": 8, "aliases": [8], "targets": ["n3"], "observed_descendants": 1},
        {"key": "g2", "representative_slot": 9, "aliases": [9], "targets": [], "observed_descendants": 0},
        {"key": "g3", "representative_slot": 10, "aliases": [10], "targets": [], "observed_descendants": 0},
        {"key": "g4", "representative_slot": 11, "aliases": [11], "targets": [], "observed_descendants": 0},
        {"key": "g5", "representative_slot": 12, "aliases": [12], "targets": [], "observed_descendants": 0},
    ]
    targets = [
        {"node_id": "n1", "batch": 0, "slot": 3, "ancestor_slot": 4, "key": "g0"},
        {"node_id": "n2", "batch": 0, "slot": 7, "ancestor_slot": 2, "key": "g0"},
        {"node_id": "n3", "batch": 1, "slot": 2, "ancestor_slot": 8, "key": "g1"},
    ]
    ranking = [{"rank": i + 1, "candidate_id": f"C{i+1}", "confidence": .8, "evidence": "trend", "counterevidence": "unknown", "observations": []} for i in range(5)]
    return {"step": 5, "groups": groups, "targets": targets}, {"step": 5, "ranking": ranking}, {"anonymous_to_key": {f"C{i+1}": f"g{i}" for i in range(5)}}


def test_coverage_helper_allows_small_configurable_denominator():
    groups = [{"targets": ["a", "b"]}, {"targets": ["c"]}]
    assert coverage_at_k(groups, ["a", "b", "c", "d"], 1) == pytest.approx(.5)
    assert coverage_at_k(groups, ["a", "b", "c", "d"], 1, denominator=2) == 1


def test_clone_equivalence_gets_alias_credit_but_strict_representative_is_separate():
    pool, answer, mapping = fixture_pool()
    result = evaluate_selection(pool, answer, mapping)
    rows = {r["node_id"]: r for r in result["target_ranks"]}
    assert rows["n1"]["alias_matched"] is True
    assert rows["n1"]["strict_representative_match"] is False
    assert rows["n2"]["alias_matched"] is True
    assert rows["n2"]["strict_representative_match"] is True
    assert result["at_1"]["coverage"] == pytest.approx(2 / 3)
    assert result["at_1"]["unique_ancestor_recall"] == pytest.approx(1 / 2)
    assert result["at_1"]["alias_match_rate"] == pytest.approx(1 / 2)
    assert result["at_1"]["strict_representative_match_rate"] == pytest.approx(1 / 2)
    assert result["at_1"]["strict_representative_coverage"] == pytest.approx(1 / 3)
    assert result["at_1"]["strict_original_branch_recall"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "distinct"),
    ("unknown", "unknown anonymous"),
    ("rank", "ranks"),
    ("step", "does not match"),
])
def test_rejects_duplicate_unknown_rank_and_step_mismatches(mutation, message):
    pool, answer, mapping = fixture_pool()
    if mutation == "duplicate":
        answer["ranking"][1]["candidate_id"] = "C1"
    elif mutation == "unknown":
        answer["ranking"][0]["candidate_id"] = "missing"
    elif mutation == "rank":
        answer["ranking"][0]["rank"] = 2
    else:
        answer["step"] = 10
    with pytest.raises(ValueError, match=message):
        evaluate_selection(pool, answer, mapping)


def test_censored_candidates_are_not_given_affinity_labels():
    pool, answer, mapping = fixture_pool()
    # Select a branch with zero recorded descendants, as could happen at this cutoff.
    mapping["anonymous_to_key"]["C1"] = "g2"
    mapping["anonymous_to_key"]["C3"] = "g0"
    result = evaluate_selection(pool, answer, mapping)
    assert result["ranking"][0]["observed_descendants"] == 0
    assert result["ranking"][0]["target_count"] == 0
    assert "affinity" not in result["ranking"][0]


def test_same_ancestor_slot_in_different_batches_is_not_one_ancestor():
    pool, answer, mapping = fixture_pool()
    # g1 is in a different batch but carries the same numeric slot as g0.
    pool["groups"][1]["batch"] = 1
    pool["targets"][2]["batch"] = 1
    pool["targets"][2]["ancestor_slot"] = 2
    pool["groups"][1]["representative_slot"] = 2
    pool["groups"][1]["aliases"] = [2]
    result = evaluate_selection(pool, answer, mapping)
    assert result["unique_ancestor_count"] == 2
    assert result["unique_original_branch_count"] == 3
    assert result["target_ranks"][2]["batch"] == 1


def test_rejects_duplicate_target_membership_across_groups():
    pool, answer, mapping = fixture_pool()
    pool["groups"][1]["targets"].append("n1")
    with pytest.raises(ValueError, match="multiple groups"):
        evaluate_selection(pool, answer, mapping)


def test_report_marks_partial_study_and_does_not_fill_missing_steps(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    (study / "run_status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    result = report(study)
    assert result["status"] == "incomplete"
    assert result["evaluated_steps"] == []
    text = (study / "report.md").read_text(encoding="utf-8")
    assert "未完成或状态未知" in text
    assert "缺失的答案不会按零分填充" in text


def test_report_selects_round_by_macro_mean_and_requires_all_51(tmp_path):
    study = tmp_path / "study"
    (study / "rounds").mkdir(parents=True)
    (study / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    for i in range(51):
        value = .8 if i == 30 else (.7 if i == 1 else .2)
        results = []
        for step in range(5, 51, 5):
            results.append({"step": step, "target_ranks": [{"node_id": f"n{j}", "rank": None} for j in range(30)],
                            "ideal_weighted_coverage_ceiling": {"1": 1 / 30, "3": 3 / 30, "5": 5 / 30},
                            **{f"at_{k}": {"coverage": value, "precision": value, "hit": 1, "unique_ancestor_recall": value} for k in (1, 3, 5)}})
        metrics = {"complete": True, "evaluated_steps": list(range(5, 51, 5)), "metrics": results,
                   "macro": {f"coverage_at_{k}": value for k in (1, 3, 5)}}
        round_dir = study / "rounds" / f"R{i:02d}"
        round_dir.mkdir()
        (round_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (round_dir / "_version.json").write_text(json.dumps({"iteration": i, "scientist_outputs_frozen_from": "R01" if i > 25 else None}), encoding="utf-8")
        if i > 25:
            (round_dir / "scientist_frozen.json").write_text(json.dumps({"source_round": "R01"}), encoding="utf-8")
        for step in range(5, 51, 5):
            job = round_dir / "selectors" / f"t{step:03d}"
            write_functional_job(job, "Q")
        scientist_job = round_dir / "scientist"
        write_functional_job(scientist_job, "S")
        (round_dir / "semantic_audit.json").write_text(json.dumps({"passes": True, "scientist_evidence_truth": True,
            "scientist_clause_applied": True, "scientist_assessment": "grounded",
            "instruction_effect_assessment": "observable", "failure_analysis": "none",
            "selector_assessments": [{"step": s, "evidence_truth": True, "clause_applied": True, "evidence": "checked",
                                      "correction_needed": "none"} for s in range(5, 51, 5)]}), encoding="utf-8")
        if i == 1:
            descriptions = [{"donor_id": f"T{j:02d}", "interval": interval, "text": f"description {j} {interval}", "evidence": []}
                           for j in range(1, 31) for interval in range(1, 11)]
            scientist = round_dir / "scientist"
            scientist.mkdir(exist_ok=True)
            scientist_answer = {"interval_descriptions": descriptions}
            (scientist / "answer.json").write_text(json.dumps(scientist_answer), encoding="utf-8")
            receipt_path = scientist / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["answer_sha256"] = hashlib.sha256(json.dumps(scientist_answer, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            (round_dir / "hypotheses.json").write_text(json.dumps([{"interval": q, "start_step": 5 * (q - 1), "end_step": 5 * q,
                "hypothesis": f"hypothesis {q}", "minority_modes": "minority"} for q in range(1, 11)]), encoding="utf-8")
    oracle = study / "oracle"
    oracle.mkdir()
    (oracle / "teacher_map.json").write_text(json.dumps({f"T{j:02d}": {"node_id": f"n{j-1}", "batch": j % 2, "slot": j} for j in range(1, 31)}), encoding="utf-8")
    (oracle / "top30.csv").write_text("rank,node_id,batch,slot,pic50_on_rescore,mmff_relief_per_heavy\n" +
        "".join(f"{j+1},n{j},{j%2},{j},8.0,0.1\n" for j in range(30)), encoding="utf-8")
    controls = study / "controls" / "structure_only" / "R01"
    controls.mkdir(parents=True)
    (controls / "metrics.json").write_text(json.dumps({"complete": True, "metrics": [], "macro": {"coverage_at_1": .1, "coverage_at_3": .2, "coverage_at_5": .3}}), encoding="utf-8")
    confirmation = study / "confirmations" / "R01" / "affinity_visible" / "repeat2"
    confirmation.mkdir(parents=True)
    (confirmation / "metrics.json").write_text(json.dumps({"complete": True, "metrics": [], "macro": {"coverage_at_1": .1, "coverage_at_3": .2, "coverage_at_5": .3}}), encoding="utf-8")
    result = report(study)
    assert result["status"] == "complete"
    assert result["best"]["round"] == "R30"
    assert result["round_audit_status"]["R30"]["scientist_source_round"] == "R01"
    curves = (study / "iteration_curves.csv").read_text(encoding="utf-8-sig")
    assert "R01" in curves
    with (study / "private_interval_descriptions.csv").open(encoding="utf-8-sig") as stream:
        assert sum(1 for _ in stream) == 301
    with (study / "best_round_target_summary.csv").open(encoding="utf-8-sig") as stream:
        assert sum(1 for _ in stream) == 31
    assert "confirmation_repeats.csv" in result["artifacts"]
    text = (study / "report.md").read_text(encoding="utf-8")
    assert "Main-model xhigh semantic audit" in text
    assert "grounded" in text
    (study / "rounds" / "R50" / "selectors" / "t050" / "functional_audit.json").unlink()
    incomplete = report(study)
    assert incomplete["status"] == "incomplete"
    assert "R50" not in incomplete["complete_rounds"]


def test_report_rejects_failed_xhigh_evidence_audit_even_when_metrics_are_complete(tmp_path):
    study = tmp_path / "study"
    rounds = study / "rounds"
    rounds.mkdir(parents=True)
    (study / "run_status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    # A full 51-round metric skeleton ensures the missing requirement is specifically the evidence audit.
    for i in range(51):
        folder = rounds / f"R{i:02d}"
        folder.mkdir()
        metrics = {"complete": True, "evaluated_steps": list(range(5, 51, 5)), "metrics": [
            {"step": s, "target_ranks": [], **{f"at_{k}": {"coverage": .2} for k in (1, 3, 5)}} for s in range(5, 51, 5)],
            "macro": {f"coverage_at_{k}": .2 for k in (1, 3, 5)}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (folder / "_version.json").write_text(json.dumps({"scientist_outputs_frozen_from": "R00" if i > 25 else None}), encoding="utf-8")
        for step in range(5, 51, 5):
            job = folder / "selectors" / f"t{step:03d}"
            job.mkdir(parents=True)
            write_functional_job(job, "Q")
        scientist_job = folder / "scientist"
        write_functional_job(scientist_job, "S")
        semantic = {"passes": i != 50, "scientist_evidence_truth": True, "scientist_clause_applied": True, "selector_assessments": [
            {"step": s, "evidence_truth": True, "clause_applied": True} for s in range(5, 51, 5)]}
        (folder / "semantic_audit.json").write_text(json.dumps(semantic), encoding="utf-8")
    result = report(study)
    assert result["status"] == "incomplete"
    assert "R50" not in result["complete_rounds"]
    assert result["round_audit_status"]["R50"]["semantic_complete"] is False
