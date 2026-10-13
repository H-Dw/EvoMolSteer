"""Scientific finish feedback and feature-specific provenance; no real API/SSH."""
from pathlib import Path
import json

import httpx

import pytest

from evomolsteer.agent_workflow.engine import run_agent, WorkflowBudgetError
from evomolsteer.agent_workflow.contracts import canonical
from evomolsteer.io import read_json
from evomolsteer.temporal_guidance.evidence import (
    FINISH_POLICY, analyst_finish, designer_finish, temporal_sources)
from evomolsteer.temporal_guidance.workflow import TemporalEvidenceTools

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/experiments/compact_skills_verification_20261012/evidence"


def finish(refs):
    return {"action": "finish", "result": {"role": "Analyst", "findings": [],
            "hypotheses": [{"statement": "Test a temporal coordinate response", "evidence_refs": refs}],
            "limitations": ["Not an observed guidance benefit"]}}


def test_missing_profiles_does_not_disable_curves():
    tools = TemporalEvidenceTools(DATA)
    catalogue = tools.catalogue()
    assert catalogue["curves_available"] and not catalogue["stage_profiles_available"]
    assert "get_curve" in tools.schemas and "get_stage_profile" not in tools.schemas


def test_summary_and_curve_audit_row_cannot_substitute_for_observation():
    tools = TemporalEvidenceTools(DATA)
    for source in ["continuous:0", "curve_support"]:
        ref = tools.execute("get_evidence", {"source_ids": [source]})["observation"]["records"][0]["ref"]
        assert analyst_finish(finish([ref])["result"], tools)


def test_invalid_finish_gets_bounded_feedback_then_actual_read_and_replay(tmp_path):
    tools = TemporalEvidenceTools(DATA)
    feature = tools.catalogue()["curve_feature_names"][0]
    requests = []
    def caller(messages, schema, output, **kwargs):
        requests.append(messages)
        if len(requests) == 1:
            return finish([])
        if len(requests) == 2:
            assert "observed_curve_not_read" in messages[-1]["content"]
            return {"action": "tool", "tool": "get_curve", "arguments": {"feature": feature}}
        ref = next(r for r, i in tools.delivered.items() if i["kind"] == "curve")
        return finish([ref])
    result = run_agent("Analyst", tools, "Test delivery", {}, tmp_path, caller=caller,
                       max_steps=4, finish_validator=analyst_finish, finish_policy=FINISH_POLICY)
    assert len(requests) == 3 and "curve_support" in result["hypotheses"][0]["evidence_ids"]
    assert (tmp_path / "step_00.finish_rejected.json").is_file()
    assert read_json(tmp_path / "completed.json")["steps"] == 3
    assert read_json(tmp_path / "binding.json")["finish_policy"] == FINISH_POLICY


def test_repeated_invalid_finish_exhausts_budget_without_false_success(tmp_path):
    with pytest.raises(WorkflowBudgetError, match="budget exhausted"):
        run_agent("Analyst", TemporalEvidenceTools(DATA), "Test", {}, tmp_path,
                  caller=lambda *a, **kw: finish([]), max_steps=2,
                  finish_validator=analyst_finish, finish_policy=FINISH_POLICY)
    assert not (tmp_path / "completed.json").exists()
    assert len(list(tmp_path.glob("*.finish_rejected.json"))) == 2


def test_designer_cannot_use_unrelated_curve_and_field_pages_keep_identity():
    tools = TemporalEvidenceTools(DATA)
    names = tools.catalogue()["curve_feature_names"]
    a = tools.execute("get_curve", {"feature": names[0]})["observation"]["ref"]
    b = tools.execute("get_curve", {"feature": names[1]})["observation"]["ref"]
    sources = {"H001": sorted(temporal_sources(tools, [a]))}
    decision = {"decision": "edit", "hypothesis_refs": ["H001"], "evidence_refs": [b]}
    assert designer_finish(decision, tools, sources)
    page = tools.execute("read_field", {"ref": a, "path": ["times"], "offset": 0, "limit": 10})
    assert page["ok"]
    decision["evidence_refs"] = [page["observation"]["ref"]]
    assert designer_finish(decision, tools, sources) == []
    decision["hypothesis_refs"] = []
    assert designer_finish(decision, tools, sources)


def test_no_forced_edit_or_hypothesis_when_evidence_is_inadequate():
    tools = TemporalEvidenceTools(DATA)
    tools.execute("get_curve", {"feature": tools.catalogue()["curve_feature_names"][0]})
    assert analyst_finish({"hypotheses": []}, tools) == []
    assert designer_finish({"decision": "retain"}, tools, {}) == []


def test_http_wrapper_feedback_and_cache_replay_without_format_repair(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "mock-key")
    requests = []
    feature = TemporalEvidenceTools(DATA).catalogue()["curve_feature_names"][0]
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            action = finish([])
        elif len(requests) == 2:
            assert "finish_feedback" in body["messages"][-2]["content"]
            action = {"action": "tool", "tool": "get_curve", "arguments": {"feature": feature}}
        else:
            obs = json.loads(body["messages"][-2]["content"])["tool_observation"]
            action = finish([obs["observation"]["ref"]])
        return httpx.Response(200, json={"model": "deepseek-flash", "id": "mock", "usage": {},
            "choices": [{"finish_reason": "stop", "message": {"content": canonical(action)}}]})
    kwargs = {"max_steps": 4, "finish_validator": analyst_finish, "finish_policy": FINISH_POLICY,
              "transport": httpx.MockTransport(handler)}
    first = run_agent("Analyst", TemporalEvidenceTools(DATA), "Test", {}, tmp_path, **kwargs)
    assert len(requests) == 3
    assert read_json(tmp_path / "step_00.receipt.json")["interface_repairs"] == []
    assert run_agent("Analyst", TemporalEvidenceTools(DATA), "Test", {}, tmp_path, **kwargs) == first
    assert len(requests) == 3
