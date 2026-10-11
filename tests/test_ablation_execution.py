"""Nested source integrity and continuation without false causal conclusions."""
import importlib
import json
from pathlib import Path

import httpx
import jsonschema
import pytest

from evomolsteer.ablation.agents import validate_evidence_citations,validate_review
from evomolsteer.ablation.execution import ExecutionBudgetExceeded,ModuleExecutionError
from evomolsteer.deepseek_api import call_json
from evomolsteer.io import read_json,write_json,digest

REPO=Path(__file__).resolve().parents[1]


def wire(value):
    return httpx.Response(200,json={"model":"deepseek-flash","id":"mock-contract",
        "choices":[{"finish_reason":"stop","message":{"content":json.dumps(value)}}]})


def test_nested_hypothesis_citations_use_all_recorded_validation_errors():
    value={"findings":[],"hypotheses":[{"evidence_ids":["source:999","source:998"]}]}
    with pytest.raises(jsonschema.ValidationError) as exc:
        validate_evidence_citations(value,{"source:0"})
    assert list(exc.value.absolute_path)==["hypotheses",0,"evidence_ids",0]
    assert len(exc.value.context)==1
    assert exc.value.instance=="source:999" and exc.value.context[0].instance=="source:998"


def test_source_repair_occurs_before_success_cache(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","test-only")
    schema={"type":"object","properties":{"hypotheses":{"type":"array","items":{"type":"object"}}}}
    invalid={"hypotheses":[{"evidence_ids":["source:999"],"claim":"Original observation"}]}
    valid={"hypotheses":[{"evidence_ids":["source:0"],"claim":"Original observation"}]}
    path=tmp_path/"Analyst.json";calls=[]
    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls)==1:return wire(invalid)
        assert not path.exists() and not path.with_suffix(".receipt.json").exists()
        assert "source:999" in calls[-1]["messages"][-1]["content"]
        return wire(valid)
    result=call_json([],schema,path,transport=httpx.MockTransport(handler),
        response_validator=lambda value:validate_evidence_citations(value,{"source:0"}))
    assert result==valid and len(calls)==2
    assert json.loads(path.with_suffix(".attempt_1.invalid.txt").read_text())==invalid
    assert read_json(path.with_suffix(".receipt.json"))["interface_repairs"]


def test_invalid_legacy_semantic_cache_is_preserved_and_not_retried(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","test-only")
    path=tmp_path/"Analyst.json";value={"hypotheses":[{"evidence_ids":["bad"]}]}
    schema={"type":"object"}
    call_json([],schema,path,transport=httpx.MockTransport(lambda r:wire(value)))
    before={p:digest(p) for p in tmp_path.iterdir()}
    def forbidden(r):raise AssertionError("Cached semantic failure must be skipped, not repeatedly regenerated")
    with pytest.raises(jsonschema.ValidationError):
        call_json([],schema,path,transport=httpx.MockTransport(forbidden),
            response_validator=lambda x:validate_evidence_citations(x,{"source:0"}))
    assert all(digest(p)==sha for p,sha in before.items())


def test_reviewer_positive_score_requires_exact_supporting_quote():
    calls=[{"claim":"Exact coordinate effect"} for _ in range(6)]
    mods=[{"id":"A1.1"}]
    result={"assessments":[{"chain":i,"module":"A1.1","score":2,"quote":"Exact coordinate effect"} for i in range(6)]}
    validate_review(result,calls,mods)
    result["assessments"][0]["quote"]="Paraphrased coordinate effect"
    with pytest.raises(jsonschema.ValidationError,match="quote"):
        validate_review(result,calls,mods)
    result["assessments"][0]["quote"]=""
    with pytest.raises(jsonschema.ValidationError,match="quote"):
        validate_review(result,calls,mods)
    result["assessments"][0]["score"]=0
    validate_review(result,calls,mods)


@pytest.fixture
def runner(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"scripts"))
    mod=importlib.import_module("run_scientific_ablation_study")
    value=mod.ScientificStudy(tmp_path,{"study":"study","config_output":"config"})
    value.state["pruning"]=[];value.state["conditions"]={}
    value.modules=[{"id":m,"role":"Analyst","lane":"G"} for m in ["A1.1","A1.2"]]
    commits=[]
    monkeypatch.setattr(value,"commit",lambda message:commits.append(message))
    monkeypatch.setattr(value,"write_report",lambda:None)
    value.test_commits=commits
    return value


def test_failed_module_is_skipped_and_next_runs_without_removing_instruction(runner,monkeypatch):
    seen=[]
    def step(phase,role,active,incumbent,module,*args,**kwargs):
        seen.append(module)
        if module=="A1.1":raise ModuleExecutionError("case","role_response",ValueError("Unknown citation"))
        return active
    monkeypatch.setattr(runner,"_sequential_step",step)
    active=["A1.1","A1.2"]
    assert runner.sequential("A","Analyst",active,{})==active
    assert seen==active
    record=runner.state["skipped_modules"][0]
    assert record["module"]=="A1.1" and not record["instruction_removed"]
    assert not record["causal_claim_supported"] and not record["performance_comparison_valid"]
    assert read_json(runner.study/"skipped_modules/A_A1.1.json")==record
    seen.clear();runner.sequential("A","Analyst",active,{})
    assert seen==["A1.2"] and len(runner.state["skipped_modules"])==1


def test_previous_failed_condition_skips_without_reading_poisoned_cache(runner,monkeypatch):
    runner.state["failure_before_skip_resume"]={"phase":"A","task":"A_A1.1_hash_alternative",
        "error":"Unknown scientific evidence citation","active_role":"Analyst","active_chain":2}
    seen=[]
    monkeypatch.setattr(runner,"_sequential_step",lambda phase,role,active,inc,module,*a,**kw:seen.append(module) or active)
    runner.sequential("A","Analyst",["A1.1","A1.2"],{})
    assert seen==["A1.2"] and runner.state["skipped_modules"][0]["chain"]==2


@pytest.mark.parametrize("error",[ExecutionBudgetExceeded("budget"),RuntimeError("Remote exit 1: git fetch unavailable")])
def test_shared_infrastructure_or_budget_does_not_become_a_skill_failure(runner,monkeypatch,error):
    def fail(*a,**kw):raise error
    monkeypatch.setattr(runner,"_sequential_step",fail)
    with pytest.raises(type(error)):
        runner.sequential("A","Analyst",["A1.1","A1.2"],{})
    assert not runner.state["skipped_modules"]


def test_failed_flowr_program_does_not_block_the_next_module(runner,monkeypatch):
    seen=[]
    def step(phase,role,active,inc,module,*a,**kw):
        seen.append(module)
        if module=="A1.1":raise RuntimeError("FLOWR failed: invalid numerical update")
        return active
    monkeypatch.setattr(runner,"_sequential_step",step)
    runner.sequential("A","Analyst",["A1.1","A1.2"],{})
    assert seen==["A1.1","A1.2"] and runner.state["skipped_modules"][0]["stage"]=="FLOWR"


def test_final_context_does_not_retest_a_skipped_module(runner,monkeypatch):
    runner.state["skipped_modules"]=[{"module":"A1.1"}];seen=[]
    monkeypatch.setattr(runner,"_final_context_step",lambda a,d,inc,frozen,module,step:seen.append(module))
    runner.final_context(["A1.1","A1.2"],[],{},[])
    assert seen==["A1.2"]
    assert runner.state["final_context"][0]["classification"]=="SKIPPED_EXECUTION_ERROR"


def test_cached_failed_review_cannot_be_used_as_confirmed_function(runner):
    case="full"
    runner.state["conditions"][case]={"functional":None}
    write_json(runner.study/"reviews"/(case+".unconfirmed.json"),{"error":"Reviewer quote not in output"})
    with pytest.raises(ModuleExecutionError) as exc:
        runner.condition(case,"Analyst",["A1.1"],{})
    assert exc.value.stage=="functional_review"


def test_handoff_fallback_uses_actual_pool_and_reports_its_original_instructions(runner,monkeypatch):
    from evomolsteer.ablation import agents
    monkeypatch.setattr(agents,"full_input_packet",lambda p:{})
    write_json(runner.study/"evidence/index.json",{"ids":[]})
    desired=[];original=["A1.1","A1.2"];rows=[]
    for i in range(6):
        p=runner.study/"calls/complete"/f"chain_{i:02d}"/"Analyst.json"
        write_json(p,{"role":"Analyst","findings":[]})
        write_json(p.with_suffix(".receipt.json"),{"backend":"real_api","response_sha256":digest(p)})
        write_json(p.with_suffix(".delivery.json"),{"active_modules":original,"alternative":None})
        rows.append({"response":p.with_name("Designer.json").relative_to(runner.repo).as_posix()})
    runner.state["conditions"]["complete"]={"role":"Analyst","case":"complete","valid":True,"active":original,"rows":rows}
    def fail(active):raise ModuleExecutionError("selected","role_response",ValueError("Invalid selected response"))
    monkeypatch.setattr(runner,"_freeze_analyst",fail)
    result=runner.freeze_analyst(desired)
    pool=read_json(runner.study/"selected_Analyst_pool.json")
    assert len(result)==6 and pool["active_modules"]==original
    assert runner.state["frozen_analyst_active"]==original
    assert pool["fallback"] and pool["not_outputs_of_failed_selected_instructions"]
