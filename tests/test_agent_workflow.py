import copy
import json
from pathlib import Path

import httpx
import jsonschema
import pytest

from evomolsteer.agent_workflow import EvidenceTools, run_agent
from evomolsteer.agent_workflow.contracts import strict_json, final_schema, validate, POLICY, canonical
from evomolsteer.agent_workflow.engine import WorkflowBudgetError, extract_result
from evomolsteer.agent_workflow.reviewer import source_cards, extract_review
from evomolsteer.agent_workflow.ablation_adapter import registered_choices
from evomolsteer.deepseek_api import call_json, LLMInterfaceError
from evomolsteer.io import read_json, digest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/experiments/m2_full_drop_ablation_20261012/evidence"


@pytest.fixture
def tools():
    return EvidenceTools(DATA)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":{"x":1,"x":2}}',
    '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '[1]', '{} {}',
    '```json\n{}\n```', 'Here: {}', '{"a":'])
def test_strict_extraction_rejects_ambiguous_json(raw):
    with pytest.raises(json.JSONDecodeError):
        strict_json(raw)


def test_strict_extraction_preserves_content():
    assert strict_json(' {"text":"中文{不是边界}","a":1.2} ') == {"text": "中文{不是边界}", "a": 1.2}


def test_unknown_old_citation_returns_no_handle(tools):
    before = tools.provenance()
    result = tools.execute("get_evidence", {"source_ids": ["regional_effects:579", "survival_effects:579"]})
    assert result["ok"] is False
    assert tools.provenance() == before


def test_all_rows_are_accessible_with_no_significance_filter(tools):
    seen = []
    offset = 0
    while True:
        result = tools.execute("query_evidence", {"tables": ["survival_effects"], "feature_contains": "",
            "sort_by": None, "descending": False, "absolute": False, "offset": offset, "limit": 24})["observation"]
        seen.extend(r["source"]["source_id"] for r in result["records"])
        if result["next_offset"] is None:
            break
        offset = result["next_offset"]
    assert seen == [r["evidence_id"] for r in tools.tables["survival_effects"]]
    assert len(seen) == 400


def test_nested_schema_disallows_manual_ids_and_wrong_roles(tools):
    delivered = tools.execute("get_evidence", {"source_ids": ["regional_effects:579"]})["observation"]["records"][0]
    result = {"role": "Analyst", "findings": [], "hypotheses": [
        {"statement": "claim", "evidence_refs": ["survival_effects:579"]}], "limitations": []}
    with pytest.raises(jsonschema.ValidationError):
        validate(result, final_schema("Analyst", tools.known))
    result["hypotheses"][0]["evidence_refs"] = [delivered["ref"]]
    validate(result, final_schema("Analyst", tools.known))
    result["design_recommendation"] = "retain"
    with pytest.raises(jsonschema.ValidationError):
        validate(result, final_schema("Analyst", tools.known))


def test_fourteen_valid_findings_are_not_truncated(tools):
    ref = tools.execute("get_evidence", {"source_ids": ["regional_effects:0"]})["observation"]["records"][0]["ref"]
    result = {"role": "Analyst", "findings": [{"claim": f"claim {i}", "evidence_refs": [ref]} for i in range(14)],
              "hypotheses": [], "limitations": []}
    validate(result, final_schema("Analyst", tools.known))
    extracted = extract_result("Analyst", result, tools)
    assert len(extracted["findings"]) == 14
    assert extracted["findings"][13]["id"] == "F014"
    assert extracted["findings"][0]["evidence_ids"] == ["regional_effects:0"]


def test_curve_keeps_empirical_nodes_fit_diagnostics_and_no_extrapolation(tools):
    feature = next(iter(tools.curves["functions"]))
    curve = tools.execute("get_curve", {"feature": feature})["observation"]
    assert curve["data"] == tools.curves["functions"][feature]
    assert curve["data"]["extrapolation_allowed"] is False
    assert len(curve["data"]["times"]) == 50
    assert "curve_support" in tools.resolve([curve["ref"]])


def test_metadata_container_paging_is_explicit(tools):
    source = tools.execute("get_evidence", {"source_ids": ["curve_support"]})["observation"]["records"][0]
    assert source["data"]["curve_fidelity_audit"]["container_summary_only"]
    page = tools.execute("read_field", {"ref": source["ref"], "path": ["curve_fidelity_audit"], "offset": 0, "limit": 2})
    assert page["ok"]
    assert len(page["observation"]["data"]) == 2


@pytest.mark.parametrize("name,args", [("shell", {"command": "whoami"}),
    ("get_curve", {"feature": "f", "path": "/etc/passwd"}),
    ("query_evidence", {"tables": ["outcomes"]})])
def test_unregistered_tools_and_extra_arguments_rejected(tools, name, args):
    with pytest.raises(jsonschema.ValidationError):
        tools.execute(name, args)


def test_registered_designer_choices_do_not_include_temperature(tools):
    allowed = {"teacher_neighbors": [2, 4, 8], "region_index": "bound"}
    choices = registered_choices(allowed, tools, {"findings": [{"evidence_ids": ["regional_effects:0"]}]})
    assert choices["C004"] == {"parameter": "region_index", "value": 0}
    assert all("temperature" not in c["parameter"] for c in choices.values())
    result = {"role": "Designer", "decision": "edit", "edit_choice": "C002", "evidence_refs": [],
              "hypothesis_refs": [], "rationale": "count", "failure_modes": []}
    validate(result, final_schema("Designer", [], choices))
    assert extract_result("Designer", result, tools, choices)["edit"] == {"parameter": "teacher_neighbors", "value": 4}
    result["decision"] = "retain"
    with pytest.raises(jsonschema.ValidationError):
        validate(result, final_schema("Designer", [], choices))


def test_reviewer_extracts_verbatim_text_and_rejects_wrong_chain():
    calls = [{"findings": [{"claim": "First actual statement."}]}, {"findings": [{"claim": "Second actual statement."}]}]
    cards = source_cards(calls)
    refs = {v["chain"]: k for k, v in cards.items()}
    rubrics = [{"module": "A1.1"}]
    raw = {"assessments": [{"chain": i, "module": "A1.1", "score": 2,
                            "support_ref": refs[i], "explanation": "specific"} for i in range(2)]}
    extracted = extract_review(raw, calls, rubrics, cards)
    assert extracted["assessments"][0]["quote"] == calls[0]["findings"][0]["claim"]
    raw["assessments"][0]["support_ref"] = refs[1]
    with pytest.raises(jsonschema.ValidationError):
        extract_review(raw, calls, rubrics, cards)


def test_real_transport_loop_requests_then_observes_then_extracts(tools, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            content = {"action": "tool", "tool": "get_evidence", "arguments": {"source_ids": ["regional_effects:0"]}}
        else:
            observation = json.loads(body["messages"][-2]["content"])["tool_observation"]
            ref = observation["observation"]["records"][0]["ref"]
            content = {"action": "finish", "result": {"role": "Analyst", "findings": [
                {"claim": "positive contrast", "evidence_refs": [ref]}], "hypotheses": [], "limitations": []}}
        return httpx.Response(200, json={"model": "deepseek-flash", "id": "test", "usage": {},
            "choices": [{"finish_reason": "stop", "message": {"content": canonical(content)}}]})
    transport = httpx.MockTransport(handler)
    result = run_agent("Analyst", tools, "Return JSON", {}, tmp_path, transport=transport)
    assert result["findings"][0]["evidence_ids"] == ["regional_effects:0"]
    assert len(requests) == 2
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["temperature"] == 0
    receipt = read_json(tmp_path / "step_01.receipt.json")
    assert digest(tmp_path / receipt["raw_response_path"]) == receipt["raw_response_sha256"]
    assert receipt["format_policy"] == POLICY
    # Resume replays immutable requests/observations from cache without HTTP.
    assert run_agent("Analyst", EvidenceTools(DATA), "Return JSON", {}, tmp_path, transport=transport) == result
    assert len(requests) == 2


def test_no_progress_loop_is_not_success(tools, tmp_path):
    def repeated(*args, **kwargs):
        return {"action": "tool", "tool": "get_evidence", "arguments": {"source_ids": ["regional_effects:0"]}}
    with pytest.raises(WorkflowBudgetError, match="Repeated"):
        run_agent("Analyst", tools, "Return JSON", {}, tmp_path, caller=repeated)
    assert not (tmp_path / "Analyst.json").exists()


def test_interface_revision_rejects_stale_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-not-a-real-key")
    with pytest.raises(ValueError, match="versioned"):
        call_json([], {}, tmp_path / "x.json", response_parser=strict_json)
