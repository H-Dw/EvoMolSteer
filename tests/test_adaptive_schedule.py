import json

import httpx
import jsonschema
import pytest

from evomolsteer.adaptive_schedule.contracts import expanded, earliest, validate_action, step_allowances
from evomolsteer.adaptive_schedule.io import eligible, better
from evomolsteer.adaptive_schedule.analysis import distribution
from evomolsteer.deepseek_api import call_json


def history():
    return [{"id": "r00", "ticks": [5] * 8, "parent_id": None}]


def action(ticks=None):
    return {"parent_id": "r00", "ticks": ticks or [6] + [5] * 7, "evidence_ids": ["r00:t080"],
            "diagnosis": "late drift", "hypothesis": "test earlier strength", "expected_change": "lower local loss",
            "falsifier": "loss increases", "memory_update": "No earlier trials"}


def test_window_has_exact_native_steps_and_interval_edges():
    values = expanded(list(range(8)))
    assert len(values) == 100 and values[:10] == [0.] * 10 and values[90:] == [0.] * 10
    assert values[10:20] == [0.] * 10 and values[80:90] == [.7] * 10
    assert earliest([5] * 8, [5, 5, 4, 5, 5, 5, 5, 5]) == 30


@pytest.mark.parametrize("ticks", [[5.0] * 8, [True] * 8, [11] * 8, [5] * 7, [-1] * 8])
def test_invalid_amplitudes_cannot_enter_native_runtime(ticks):
    with pytest.raises(ValueError):
        expanded(ticks)


def test_evidence_and_duplicate_and_trust_region_are_enforced():
    h = history()
    assert validate_action(action(), h, ["r00:t080"]) == 10
    with pytest.raises(jsonschema.ValidationError):
        validate_action(action([5] * 8), h, ["r00:t080"])
    with pytest.raises(jsonschema.ValidationError):
        validate_action(action([7] + [5] * 7), h, ["r00:t080"])
    with pytest.raises(jsonschema.ValidationError):
        validate_action(action(), h, ["different"])
    with pytest.raises(jsonschema.ValidationError):
        validate_action(action([6, 6, 6, 5, 5, 5, 5, 5]), h, ["r00:t080"])


def test_acceleration_requires_actual_improving_parent_lineage():
    h = history() + [
        {"id": "r01", "ticks": [6] + [5] * 7, "parent_id": "r00", "improved_parent": True},
        {"id": "r02", "ticks": [7] + [5] * 7, "parent_id": "r01", "improved_parent": True},
    ]
    assert step_allowances(h, "r02") == {0: 1}
    a = dict(action([9] + [5] * 7), parent_id="r02")
    assert validate_action(a, h, ["r00:t080"]) == 10
    h[-1]["improved_parent"] = False
    with pytest.raises(jsonschema.ValidationError):
        validate_action(a, h, ["r00:t080"])


def record(score, smiles="C"):
    return {"build_success": True, "connected": True, "failure_reason": "",
            "pic50_on_rescore": score, "smiles": smiles}


def test_failure_is_not_a_zero_score_and_best_ties_keep_incumbent():
    assert not eligible(record(float("nan")))
    assert not eligible(dict(record(9), connected=False))
    assert not better(record(7), record(7))
    assert better(record(7), record(6))
    assert better(record(1), dict(record(9), build_success=False))


def test_top10_and_dedup_do_not_silently_change_denominator():
    rows = [record(float(i), str(i)) for i in range(100)]
    out = distribution(rows, 98)
    assert out["attempts"] == 100 and out["top10_mean"] == 94.5 and out["best"] == 99
    assert out["teacher_recovered_count"] == 2
    assert distribution(rows[:9], 5)["top10_mean"] is None
    duplicate = [record(float(i), "same") for i in range(100)]
    assert distribution(duplicate, 90, True)["top10_mean"] is None
    assert distribution(duplicate, 90, True)["best"] == 99


def test_thinking_request_is_opt_in_and_receipt_is_accurate(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    requests = []
    def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200, json={"model": "deepseek-flash", "system_fingerprint": "test-fp",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]})
    s = {"type": "object", "properties": {"ok": {"const": True}}, "required": ["ok"]}
    call_json([], s, tmp_path / "thinking.json", thinking=True, transport=httpx.MockTransport(respond))
    call_json([], s, tmp_path / "legacy.json", transport=httpx.MockTransport(respond))
    assert requests[0]["thinking"]["type"] == "enabled" and requests[0]["reasoning_effort"] == "high"
    assert "temperature" not in requests[0]
    assert requests[1]["thinking"]["type"] == "disabled" and requests[1]["temperature"] == 0
    receipt = json.loads((tmp_path / "thinking.receipt.json").read_text())
    assert receipt["thinking"] == "enabled" and receipt["system_fingerprint"] == "test-fp"
