"""Regression coverage for the recorded Analyst field failure and safe resume."""
import importlib
import json
from pathlib import Path

import httpx
import pytest

from evomolsteer.continuous.m2_skill_api import ANALYST_SCHEMA
from evomolsteer.deepseek_api import FORMAT_POLICY, LLMInterfaceError, call_json
from evomolsteer.io import digest, read_json, write_json


REPO = Path(__file__).resolve().parents[1]
HISTORICAL = REPO / "docs/experiments/steer_dependency_20261009/deepseek_flash_m2_20261011/calls/P_struct/chain_04/Analyst.attempt_1.invalid.txt"
SIMPLE = {"type": "object", "additionalProperties": False,
          "required": ["ok"], "properties": {"ok": {"const": True}}}


def response(content):
    return httpx.Response(200, json={"model": "deepseek-flash", "id": "mock",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}]})


def test_historical_extra_field_has_specific_neutral_repair(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    original = HISTORICAL.read_text(encoding="utf-8")
    repaired = json.loads(original)
    detail = repaired.pop("design_recommendation")
    repaired["hypotheses"].append({"preserved_design_detail": detail})
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert "allowed_root_properties" in body["messages"][-1]["content"]
            return response(original)
        repair = body["messages"][-1]["content"]
        contract = json.loads(repair.split("\n", 1)[1])
        assert contract["validation_errors"]["unexpected_properties"] == ["design_recommendation"]
        assert contract["validation_errors"]["schema_pointer"] == "/additionalProperties"
        assert "design_recommendation" not in contract["allowed_root_properties"]
        assert contract["response_schema"] == ANALYST_SCHEMA
        assert body["messages"][-2]["content"] == original
        assert "Do not add research advice" in repair
        return response(json.dumps(repaired))
    path = tmp_path / "Analyst.json"
    result = call_json([{"role": "user", "content": "Analyze supplied evidence"}], ANALYST_SCHEMA,
        path, transport=httpx.MockTransport(handler), input_store=tmp_path / "inputs")
    assert result == repaired
    assert result["findings"] == json.loads(original)["findings"]
    assert path.with_suffix(".attempt_1.invalid.txt").read_text(encoding="utf-8") == original
    receipt = read_json(path.with_suffix(".receipt.json"))
    assert receipt["format_policy"] == FORMAT_POLICY and len(receipt["interface_repairs"]) == 1
    assert len(requests) == 2


def test_network_retries_do_not_use_up_the_format_repair(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        return response('{"ok":true,"extra":1}' if len(calls) == 3 else '{"ok":true}')
    path = tmp_path / "a.json"
    assert call_json([], SIMPLE, path, transport=httpx.MockTransport(handler), attempts=3,
                     sleep=lambda _: None) == {"ok": True}
    assert len(calls) == 4
    assert read_json(path.with_suffix(".receipt.json"))["attempts"] == 4


def test_retry_failure_records_are_append_only_and_not_fake_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    path = tmp_path / "a.json"
    path.with_suffix(".attempt_1.invalid.txt").write_text("old failure 1")
    path.with_suffix(".attempt_2.invalid.txt").write_text("old failure 2")
    write_json(path.with_suffix(".repair.request.json"), {"historical": True})
    write_json(path.with_suffix(".repair_1.request.json"), {"earlier_v2_attempt": True})
    old_files = {p: digest(p) for p in tmp_path.iterdir()}
    requests = []
    def handler(request):
        requests.append(request)
        return response('{"ok":true,"extra":1}')
    with pytest.raises(LLMInterfaceError) as error:
        call_json([], SIMPLE, path, transport=httpx.MockTransport(handler))
    assert error.value.diagnostics["unexpected_properties"] == ["extra"]
    assert len(requests) == 2 and not path.exists()
    assert not path.with_suffix(".receipt.json").exists()
    assert all(digest(p) == sha for p, sha in old_files.items())
    assert path.with_suffix(".attempt_3.invalid.txt").exists()
    assert path.with_suffix(".attempt_4.invalid.txt").exists()
    assert path.with_suffix(".repair_2.request.json").exists()


def test_changed_instructions_and_corrupt_cache_do_not_overwrite_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    path = tmp_path / "a.json"
    transport = httpx.MockTransport(lambda _: response('{"ok":true}'))
    messages = [{"role": "user", "content": "original task"}]
    assert call_json(messages, SIMPLE, path, transport=transport) == {"ok": True}
    protected = {p: digest(p) for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="original request preserved"):
        call_json([{"role": "user", "content": "different task"}], SIMPLE, path, transport=transport)
    assert all(digest(p) == sha for p, sha in protected.items())
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    # Reading a verified cache needs no credentials or API request.
    assert call_json(messages, SIMPLE, path) == {"ok": True}
    write_json(path, {"ok": True, "tampered": True})
    with pytest.raises(ValueError, match="checksum"):
        call_json(messages, SIMPLE, path)


def test_v1_successful_receipt_remains_reusable(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    path = tmp_path / "a.json"
    messages = [{"role": "user", "content": "same scientific request"}]
    call_json(messages, SIMPLE, path, transport=httpx.MockTransport(lambda _: response('{"ok":true}')))
    receipt = read_json(path.with_suffix(".receipt.json"))
    for key in ["format_policy", "wire_request_sha256", "wire_request_path"]:
        receipt.pop(key)
    write_json(path.with_suffix(".receipt.json"), receipt)
    before = {p: digest(p) for p in tmp_path.iterdir() if p.is_file()}
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert call_json(messages, SIMPLE, path) == {"ok": True}
    assert all(digest(p) == sha for p, sha in before.items())


def test_nested_format_error_identifies_the_actual_object(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    schema = {"type": "object", "properties": {"findings": {"type": "array", "items": SIMPLE}}}
    with pytest.raises(LLMInterfaceError) as error:
        call_json([], schema, tmp_path / "a.json", transport=httpx.MockTransport(
            lambda _: response('{"findings":[{"ok":true,"extra":1}]}')))
    assert error.value.diagnostics["instance_pointer"] == "/findings/0"
    assert error.value.diagnostics["allowed_properties"] == ["ok"]


def test_supervisor_reports_partial_chains_without_incumbent_fallback(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    module = importlib.import_module("run_m2_skill_api_study")
    runner = module.Study(tmp_path, {"study": "study", "config_output": "config"})
    runner.state["controls"] = {"outcomes": {
        "gradient": {"summary": {"affinity_mean_all": 7.6}},
        "unguided": {"summary": {"affinity_mean_all": 7.4}}}}
    runner.state["chains"] = [{"condition": "P_struct", "effective_sha256": "same", "mean_pic50": 7.6}]
    def fail(*args, **kwargs):
        raise LLMInterfaceError({"validator": "additionalProperties", "unexpected_properties": ["extra"]},
                                tmp_path / "study/calls/P_struct/chain_04/Analyst.json")
    monkeypatch.setattr(module, "role_call", fail)
    with pytest.raises(LLMInterfaceError):
        runner.call_role({"condition_id": "P_struct"}, 4, "Analyst")
    assert len(runner.state["chains"]) == 1
    assert runner.state["status"] == "failed"
    report = read_json(tmp_path / "study/screen_summary.json")
    assert not report["conditions"][0]["condition_complete"]
    assert not report["interface_failures"][0]["incumbent_fallback"]
    assert not report["interface_failures"][0]["counted_as_valid_chain"]
