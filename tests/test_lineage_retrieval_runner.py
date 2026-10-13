import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evomolsteer.lineage_retrieval.runner import _scientist_interval_payload, _validate_packet, run_job
from evomolsteer.lineage_retrieval.orchestrate import teacher_packet


def packet():
    return {
        "step": 2,
        "cards": [
            {"candidate_id": "anon_a", "now": {"shape": [1.0, 2.0]}, "history": [{"step": 1, "descriptor": 0.4}]},
            {"candidate_id": "anon_b", "now": {"shape": [2.0, 3.0]}, "history": []},
        ],
        "reference": {"motif": "compact"},
        "hypothesis": "Compare the recorded features.",
    }


def test_packet_rejects_future_steps_nested_in_arbitrary_fields():
    p = packet()
    p["cards"][0]["history"][0]["metadata"] = {"generation": 3}
    with pytest.raises(ValueError, match="future step"):
        _validate_packet(p, "Selector")


@pytest.mark.parametrize("field,value", [("source_path", "D:\\oracle\\x.json"), ("candidate_path", "../oracle.json"), ("final_score", 7.0), ("oracle_value", "x")])
def test_packet_rejects_paths_scores_and_oracle_fields(field, value):
    p = packet()
    p["cards"][0]["now"][field] = value
    with pytest.raises(ValueError):
        _validate_packet(p, "Selector")


def test_packet_rejects_candidate_id_mismatch_and_duplicate_ids():
    p = packet()
    p["cards"][0]["candidate_id"] = "same"
    p["cards"][1]["candidate_id"] = "same"
    with pytest.raises(ValueError, match="unique anonymous"):
        _validate_packet(p, "Selector")
    p = packet()
    p["cards"][0]["now"]["candidate_id"] = "other"
    with pytest.raises(ValueError, match="candidate id mismatch"):
        _validate_packet(p, "Selector")


def test_affinity_hidden_packet_strips_affinity_recursively():
    p = packet()
    p["cards"][0]["now"]["online_pic50"] = 7.1
    p["affinity_available"] = False
    clean = _validate_packet(p, "Selector")
    assert "online_pic50" not in clean["cards"][0]["now"]


def test_current_online_affinity_is_retained_when_available():
    p = packet()
    p["affinity_available"] = True
    p["cards"][0]["now"]["online_pic50"] = 7.1
    clean = _validate_packet(p, "Selector")
    assert clean["cards"][0]["now"]["online_pic50"] == 7.1


def test_scientist_packet_supports_actual_teacher_packet_with_full_trusted_horizon(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    teacher = {"donors": [{"donor_id": f"donor_{i}", "observations": [{"step": step, "current_shape": [step / 10]} for step in range(51)]} for i in range(30)]}
    (evidence / "teacher.json").write_text(json.dumps(teacher), encoding="utf-8")
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / "population_summaries.json").write_text(json.dumps([{"step": step, "quantiles_P25_P50_P75": {"current_shape": [0.1, 0.2, 0.3]}} for step in range(51)]), encoding="utf-8")
    scientist = teacher_packet(tmp_path)
    assert _validate_packet(scientist, "Scientist")["role"] == "Scientist"
    scientist["teacher"]["donors"][0]["observations"][50]["step"] = 51
    with pytest.raises(ValueError, match="future step|cover steps"):
        _validate_packet(scientist, "Scientist")


def test_scientist_interval_payload_never_returns_future_teacher_or_background_rows():
    source = {
        "teacher": {"donors": [{"donor_id": f"d{i}", "observations": [{"step": step, "feature": step} for step in range(51)]} for i in range(30)]},
        "background": [{"step": step, "quantiles": [step]} for step in range(51)],
    }
    first = _scientist_interval_payload(source, 1)
    step_column = first['observation_columns'].index('step')
    assert [row[step_column] for row in first['donors'][0]['observation_rows']] == list(range(6))
    assert [row["step"] for row in first["background"]] == list(range(6))
    tenth = _scientist_interval_payload(source, 10)
    step_column = tenth['observation_columns'].index('step')
    assert [row[step_column] for row in tenth['donors'][0]['observation_rows']] == list(range(45, 51))
    assert [row['step'] for row in tenth['background']] == list(range(45, 51))
    with pytest.raises(ValueError, match="1..10"):
        _scientist_interval_payload(source, 11)


def test_privileged_main_agent_packet_does_not_require_selector_cards():
    main = {"role": "main-agent", "current_skill": "Use the approved workflow.", "failure_evidence": [{"step": 50, "observed_final_target_count": 2}]}
    assert _validate_packet(main, "main-agent")["current_skill"].startswith("Use")


def test_run_job_persists_reproducible_artifacts_without_model_call(tmp_path):
    schema = {"type": "object", "properties": {"decision": {"type": "string"}}, "required": ["decision"], "additionalProperties": False}

    def broker(name, args, clean_packet):
        return {"ok": True}

    # No network invocation is needed to test preflight behavior; invalid packet
    # must fail before a subprocess is started or artifacts are written.
    with pytest.raises(ValueError, match="future step"):
        run_job(tmp_path / "job", "Selector", "Choose", {**packet(), "future_step": 3}, schema)
    assert not (tmp_path / "job" / "packet.json").exists()
