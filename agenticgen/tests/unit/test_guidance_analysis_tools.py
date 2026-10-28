import json
from pathlib import Path
import pytest
import httpx
from guidance_tool_fixture import build
from agenticgen.uniguide.analysis_tools import EvidenceTools,EvidenceIntegrityError,metric_summary,paired_summary,CONTROL_DISTANCE
from agenticgen.uniguide.io import read,write,sha
from agenticgen.uniguide.tool_runtime import configure,assets,bind_experiment
from agenticgen.uniguide.geometry_council import advise
from agenticgen.providers.deepseek_tools import call_with_tools


def test_robust_center_preserves_tail_and_missing_not_zero():
    out=metric_summary([.25]*49+[12.])
    assert out["arithmetic_mean"]==pytest.approx(.485) and out["symmetric_ten_percent_trimmed_mean"]==.25
    assert out["maximum"]==12 and out["expected"]==50
    out=metric_summary([None,0.,float('nan')])
    assert out["status"]=="incomplete_descriptive_only" and out["measured"]==1


def test_evidence_receipts_pairing_tail_steps_cache_and_integrity(tmp_path):
    history,domain=build(tmp_path)
    registry=EvidenceTools(tmp_path,history,domain,tmp_path/"analysis")
    initial=registry.precompute();assert len(initial)==3
    result=registry.execute("compare_paired_runs",dict(reference_id="fixture/0",candidate_id="fixture/1"))
    assert result["seed_clusters"]==5 and result["paired_slots"]==50
    assert result["metrics"][CONTROL_DISTANCE]["seed_mean_changes"][0]["paired_slots"]==10
    assert result["metrics"][CONTROL_DISTANCE]["trimmed_mean_change"]<0
    assert result["metrics"][CONTROL_DISTANCE]["arithmetic_mean_change"]>0
    assert registry.execute("compare_paired_runs",dict(reference_id="fixture/0",candidate_id="fixture/1"))==result
    out=registry.execute("inspect_interval_evidence",dict(run_id="fixture/1"))
    assert out["tail_step_diagnostics"][0]["largest_increment_steps"][0]["step"]==80
    source=tmp_path/"fixture/0/records.json";write(source,[])
    with pytest.raises(EvidenceIntegrityError,match="changed"):registry.execute("summarize_population",dict(run_id="fixture/0"))
    with pytest.raises(EvidenceIntegrityError):registry.validate_citations([initial[0]["evidence_id"]])


def test_displacement_tail_includes_normal_geometry_and_unconstrained_steps(tmp_path):
    h,d=build(tmp_path);path=tmp_path/"fixture/1/steps.json";steps=read(path)
    steps[12]["max_atom_increment_A"][0]=50.
    steps[18]["unconstrained_max_atom_increment_A"]=[0.]*50
    steps[18]["unconstrained_max_atom_increment_A"][1]=100.
    write(path,steps);complete=tmp_path/"fixture/1/COMPLETE.json";receipt=read(complete)
    receipt["files"]["steps.json"]=sha(path);write(complete,receipt)
    tools=EvidenceTools(tmp_path,h,d,tmp_path/"analysis")
    out=tools.execute("inspect_interval_evidence",dict(run_id="fixture/1"))
    rows={r["node_id"]:r for r in out["tail_step_diagnostics"]}
    assert rows["fixture_0"]["largest_increment_steps"][0]["step"]==12
    assert rows["fixture_1"]["largest_unconstrained_increment_steps"][0]["step"]==18


def test_paired_missing_slots_and_duplicate_identity_are_unavailable(tmp_path):
    build(tmp_path);a=read(tmp_path/"fixture/0/records.json");b=read(tmp_path/"fixture/1/records.json")
    assert paired_summary(a,b[:-1])["status"]=="unavailable"
    assert paired_summary(a,b[:-1]+[b[0]])["status"]=="unavailable"


def test_action_audit_does_not_generate_or_accept_repeated_configuration(tmp_path):
    h,d=build(tmp_path);tools=EvidenceTools(tmp_path,h,d,tmp_path/"analysis")
    action=dict(anchor_id="fixture/0",branch="increase",updates=[dict(window_id=0,step=1)])
    out=tools.execute("audit_action_batch",dict(actions=[action]))
    assert out["actions"][0]["status"]=="rejected" and out["new_molecular_attempts"]==0
    assert not (tmp_path/"attempts.json").exists()
    action["updates"][0]["window_id"]=2
    valid=tools.execute("audit_action_batch",dict(actions=[action]))["actions"][0]
    assert valid["resolved_effective_guidance_scale_schedule"]==[2.,2.,4.,2.]
    assert valid["normalized_zero_disables_guidance"] is False


def test_new_defaults_and_legacy_freeze(tmp_path):
    config=configure(dict(root=str(tmp_path)))
    assert config["agent_tools"]["max_model_calls_per_round"]==30
    assert len(assets(config))==4
    binding=bind_experiment(config);assert bind_experiment(config)==binding
    legacy=dict(root=str(tmp_path));write(tmp_path/"GEOMETRY_CONFIG.json",legacy)
    assert configure(legacy)==legacy


def test_review_preserves_experiment_skills_and_rejects_rsi_patches(tmp_path):
    from agenticgen.uniguide import geometry_runtime as runtime
    from agenticgen.uniguide.geometry_protocol import POLICY
    declared=read(Path(__file__).resolve().parents[2]/"configs/uniguide_geometry.json")
    config=runtime.validate_config(dict(declared,root=str(tmp_path/"experiment"),
        generation_context_root=str(tmp_path/"source"),protocol=POLICY,rounds=2,conditions=[]))
    root=Path(config["root"]);folder=root/"versions/GD01";skills=folder/"skills"
    skills.mkdir(parents=True)
    for name,path in assets(config).items():(skills/name).write_bytes(path.read_bytes())
    bind_experiment(config)
    request=folder/"AWAITING_GPT_REVIEW.json"
    write(request,dict(strategy_contexts=["fixture-context"],next_version="GD02"))
    review=dict(version="GD01",review_request_sha256=sha(request),
        read_strategy_contexts=["fixture-context"],patches=[])
    path=tmp_path/"review.json";write(path,review)
    runtime.accept_review(config,"GD01",path)
    assert read(folder/"REVIEW_ACCEPTED.json")["review_scope"].startswith("results only")
    write(path,dict(review,patches=[dict(module="analysis.md")]))
    with pytest.raises(ValueError,match="frozen skills"):runtime.accept_review(config,"GD01",path)


def test_release_payload_and_previous_release_unchanged():
    import yaml
    folder=Path(__file__).resolve().parents[2]/"src/agenticgen/resources/skill_releases/guidance"
    manifest=read(folder/"2.1.0/RELEASE.json")
    for name,digest in manifest["files_sha256"].items():assert sha(folder/"2.1.0"/name)==digest
    for name,digest in manifest["old_files_sha256"].items():assert sha(folder/"2.0.0"/name)==digest
    for path in (folder/"2.1.0/skills").glob("*/SKILL.md"):
        text=path.read_text(encoding="utf-8")
        front=yaml.safe_load(text.split("---",2)[1])
        assert front["name"]==path.parent.name and front["description"]
        assert "\ufffd" not in text


def test_full_council_uses_tools_then_original_role_json_and_shared_ledger(tmp_path,monkeypatch):
    h,d=build(tmp_path);config=configure(dict(root=str(tmp_path),llm=dict(thinking=True,attempts=1)))
    skills=tmp_path/"skills";skills.mkdir()
    for n,p in assets(config).items():(skills/n).write_bytes(p.read_bytes())
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real");seen=[]
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        context=json.loads(body["messages"][1]["content"])
        used=any(m['role']=='tool' for m in body['messages'])
        if not used:
            message=dict(role="assistant",content=None,reasoning_content="opaque",tool_calls=[dict(id="pair",type="function",
                function=dict(name="compare_paired_runs",arguments=json.dumps(dict(reference_id="fixture/0",candidate_id="fixture/1"))))])
            reason="tool_calls"
        else:
            ev=json.loads(next(m['content'] for m in reversed(body['messages']) if m['role']=='tool'))['evidence_id']
            if 'branch' in context:
                result=dict(analysis=ev,mechanism_uncertainty="Synthetic fixture; not molecular evidence",proposals=[dict(p,id=str(i),
                    anchor_reason="measured fixture",evidence=ev,prediction="hypothesis",falsifier="contrary measured change") for i,p in enumerate(context['feasible_examples'])])
            else:result=dict(analysis=ev,choice_evidence=ev,selected={b:b+':0' for b in ['increase','decrease','redistribute']})
            message=dict(role="assistant",content=json.dumps(result),reasoning_content="opaque");reason="stop"
        return httpx.Response(200,json=dict(model="deepseek-flash",usage=dict(completion_tokens=12),choices=[dict(message=message,finish_reason=reason)]))
    def provider(*args,**kwargs):return call_with_tools(*args,**kwargs,transport=httpx.MockTransport(handle))
    chosen=advise(config,h,d,tmp_path/"round",skills,provider=provider)
    assert len(chosen)==3 and len(seen)==8
    assert len(list((tmp_path/"round/model_budget").glob('call_*.json')))==8
    assert not (tmp_path/"attempts.json").exists()
    before=len(seen);assert advise(config,h,d,tmp_path/"round",skills,provider=provider)==chosen
    assert len(seen)==before
