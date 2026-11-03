"""2.14 complete-population evaluation, one-role transport and recorded paths."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import h5py
import httpx
import numpy as np
import pytest
import torch

from test_dynamic_matching_jump import dynamic_condition
from test_absolute_coordinate_protocol import condition
from test_diverse_box_search import action
from agenticgen.uniguide import condition_evaluation as fitted
from agenticgen.uniguide import path_search, path_calibration, linear_scale as scale
from agenticgen.uniguide import linear_runtime, linear_council, checkpoint_learning
from agenticgen.uniguide.absolute_runtime import validate_config
from agenticgen.uniguide.absolute_coordinate import RMSD, MAX_ERROR
from agenticgen.uniguide.coordinate_statistics import aggregate_fields, RANKING, ranking_eligible
from agenticgen.uniguide.geometry_protocol import choose, primary_score
from agenticgen.uniguide.io import read, write, sha, value_sha, Budget


def config(root):
    folder = Path(__file__).resolve().parents[2]/"configs"
    c = read(folder/"uniguide_absolute_coordinate.json")
    c.update(read(folder/"uniguide_early_path_search.json"))
    c.update(root=str(root), generation_context_root=str(root.parent/"frozen"))
    return validate_config(c)


def executor_fixture(root, monkeypatch, fail=None):
    from agenticgen.uniguide import geometry_sampling
    calls = []
    def binding(c, t, k):
        return dict(teacher_id=t["target"], condition_sha256=k["condition_sha256"],
                    condition_evaluation=c[fitted.CONFIG], windows=c["windows"])
    monkeypatch.setattr(geometry_sampling, "compatibility", binding)
    monkeypatch.setattr(linear_runtime, "compatibility", binding)
    def executor(model, c, t, k, schedule, folder, stage, count=50, domain=None):
        actual = scale.effective(schedule, domain) if domain else schedule
        identity = dict(compatibility=binding(c,t,k), **{scale.EFFECTIVE:actual}, population=count)
        fingerprint = value_sha(identity); identifier = folder.relative_to(root).as_posix()
        Budget(root).reserve(identifier, stage, count, fingerprint)
        calls.append(dict(actual=actual, stage=stage, count=count))
        write(root/"measurement_registry"/(fingerprint+".json"),dict(identity=identity,source_group=identifier))
        if fail and fail(actual, stage):
            write(folder/"FAILED.json",dict(error="injected nonfinite",no_retry=True))
            raise FloatingPointError("injected nonfinite")
        mean = .1 + sum(actual)/10000
        records = [dict(node_id=f"{seed}_{slot}", logical_seed=seed, logical_slot=slot,
                        build_success=False, **{RMSD:2-mean, MAX_ERROR:3., fitted.METRIC:mean, fitted.POSE:2-mean})
                   for seed in range(5) for slot in range(10)]
        row = dict(id=identifier, teacher_id=t["target"], condition_sha256=k["condition_sha256"],
                   complete=True, attempts=count, built=0, connected=0, unique_smiles=0,
                   fingerprint=fingerprint, compatibility_sha256=value_sha(identity["compatibility"]),
                   **{scale.EFFECTIVE:actual, RMSD:2-mean, MAX_ERROR:3.}, **aggregate_fields(records,c),
                   window_diagnostics=[dict(increment_vector_rms_A=1.,max_atom_increment_A=1.) for _ in actual])
        write(folder/"records.json",records); write(folder/"aggregate.json",row)
        write(folder/"manifest.json",dict(identity=identity)); write(folder/"steps.json",[])
        write(folder/"COMPLETE.json",dict(files={n:sha(folder/n) for n in ["records.json","aggregate.json","manifest.json","steps.json"]}))
        return row
    return executor, calls


def setup(root, monkeypatch, fail=None):
    c = config(root)
    s = dict(teacher=dict(target="6pyy_H7S_A_402"), condition=dict(condition_sha256="fixture"))
    executor, calls = executor_fixture(root,monkeypatch,fail)
    domains = path_calibration.run(c,[s],lambda *args: None,executor)
    d = domains[s["teacher"]["target"]]
    h, failures = linear_runtime.history(c,s["teacher"],s["condition"],d)
    return c,s,d,h,failures,executor,calls


def five(c,h,d,round_id=1,learning=None):
    best = choose(h)[scale.EFFECTIVE]
    profiles=[]
    for i in [0,1,2,3]:
        p=list(best);p[i]=d["search_grids"][i][-round_id-1];profiles.append(p)
    p=list(best);p[4:]=[400-10*round_id,300-10*round_id];profiles.append(p)
    return [action(c,h,d,p,f"r{round_id}_a{i}",learning) for i,p in enumerate(profiles)]


def test_rigid_fit_uses_only_fixed_condition_points():
    target=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3]],dtype=float)
    rotation=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    x=target@rotation+np.array([9,-4,12])
    assert fitted.fit(x,target)["rmsd_angstrom"] < 1e-12
    assert np.linalg.det(fitted.fit(x,target)["rotation"]) == pytest.approx(1.)
    reflected=target.copy();reflected[:,0]*=-1
    assert fitted.fit(reflected,target)["rmsd_angstrom"] > .1
    assert fitted.fit(np.ones((4,3)),np.ones((4,3))*3)["rmsd_angstrom"] == 0
    assert fitted.fit(target[:1],x[:1])["rmsd_angstrom"] == 0
    with pytest.raises(FloatingPointError):fitted.fit([[np.nan,0,0]],[[0,0,0]])


def test_terminal_condition_evaluation_preserves_input_and_includes_build_failures(dynamic_condition,tmp_path):
    from agenticgen.uniguide.geometry_sampling import terminal_records
    x=torch.tensor([[[0.,0,0],[0,20,0],[1.4,0,0]]],dtype=torch.float64).repeat(50,1,1)
    out=dict(coords=x,mask=torch.ones(50,3,dtype=torch.bool))
    with h5py.File(tmp_path/"trace.h5","w") as trace:trace["assignment"]=np.tile([0,2],(100,50,1))
    model=SimpleNamespace(_generate_mols=lambda output,sanitise=True:[None]*50)
    before=x.clone();rng=torch.get_rng_state().clone()
    rows=terminal_records(model,out,dynamic_condition,tmp_path,50,config={fitted.CONFIG:dict(policy=fitted.POLICY)})
    assert len(rows)==50 and not any(r["build_success"] for r in rows)
    assert all(r[fitted.METRIC] < 1e-12 for r in rows)
    assignment=torch.tensor([[0,2]]*50)
    baseline=fitted.evaluate(x,dynamic_condition,assignment)[0]
    x[:,1]=1e6
    assert fitted.evaluate(x,dynamic_condition,assignment)[0] == baseline
    x[:,1]=before[:,1]
    assert torch.equal(before,x) and torch.equal(rng,torch.get_rng_state())
    assert read(tmp_path/"COORDINATE_ASSIGNMENT.json")[fitted.CONFIG]["policy"]==fitted.POLICY


def scored(identifier,values,pose):
    records=[{fitted.METRIC:float(v)} for v in values]
    return dict(id=identifier,complete=True,attempts=len(values),**{RMSD:pose,scale.EFFECTIVE:[50]*6},
                **aggregate_fields(records,{fitted.CONFIG:dict(policy=fitted.POLICY)}))


def test_unfiltered_mean_ranking_no_fallback_and_no_mixed_protocols():
    a=scored("a",[0.]*49+[50.],.01);b=scored("b",[.5]*50,2.)
    assert primary_score(a)==1. and choose([a,b])["id"]=="b"
    assert checkpoint_learning.checkpoint([a],b)["best_compatible_source_id"]=="b"
    assert not ranking_eligible(scored("partial",[.01]*49,.01))
    for key in [fitted.MEAN,RANKING]:
        broken=copy.deepcopy(a);broken.pop(key)
        with pytest.raises(ValueError):primary_score(broken)
        assert not ranking_eligible(broken)
    with pytest.raises(ValueError):choose([a,dict(id="legacy",complete=True,**{RMSD:.01})])
    with pytest.raises(FloatingPointError):scored("bad",[.1]*49+[float("nan")],.1)
    with pytest.raises(FloatingPointError):fitted.summarize([.1]*49+[None])
    with pytest.raises(FloatingPointError):fitted.summarize([.1]*49+[float('inf')])
    b['stability_status']='large_finite_displacement_growth'
    assert checkpoint_learning.checkpoint([b],a)['best_compatible_source_id']=='b'


def test_deterministic_calibration_has_ten_full_profiles_and_frozen_probe_background(tmp_path,monkeypatch):
    c,s,d,h,failed,executor,calls=setup(tmp_path,monkeypatch)
    assert len(calls)==10 and all(r["count"]==50 for r in calls)
    assert sum(r["count"] for r in calls)==500 and not failed
    fixed=read(tmp_path/"calibration/path/6pyy_H7S_A_402/NEIGHBOUR_REFERENCE.json")["actual_schedule"]
    for r in calls[2:]:
        changed=[i for i,(a,b) in enumerate(zip(r["actual"],fixed)) if a!=b]
        assert len(changed)==1 and changed[0]<4
    assert d["initial_source"]==choose(h)["id"]
    assert all(r[fitted.MEAN] == pytest.approx(2-r[RMSD]) for r in h)
    assert "box_exclusion_policy" not in d and "selection_sha256" not in d
    assert read(tmp_path/"ATTEMPT_LIMITS.json")["per_teacher_stage_caps"]==dict(calibration=500,formal=1250)
    path_calibration.run(c,[s],lambda *args: pytest.fail("replay loaded a model"),executor)
    assert len(calls)==10


def test_failed_background_is_charged_and_remaining_background_survives(tmp_path,monkeypatch):
    baseline=path_calibration.BACKGROUNDS["6pyy_H7S_A_402"][0]
    c,s,d,h,failed,executor,calls=setup(tmp_path,monkeypatch,lambda p,stage:p==baseline)
    assert len(calls)==10 and len(h)==9 and len(failed)==1
    assert sum(r["count"] for r in calls)==500


def test_neighbour_boundary_uses_two_unique_legal_doses():
    grids=[[0,1,2,5,10],*[list(range(0,401,10)) for _ in range(5)]]
    probes=path_calibration.neighbours([0,400,20,30,50,50],grids)
    assert [p[0] for _,p in probes[:2]] == [1,2]
    assert [p[1] for _,p in probes[2:4]] == [390,380]


def test_parent_relative_directions_and_independent_middle_boxes(tmp_path,monkeypatch):
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch)
    proposals=five(c,h,d)
    report=path_search.audit_early(proposals,h,d,[],250)
    assert report["observed_early_directions"]==4
    parent=choose(h);a=action(c,h,d,[300,*parent[scale.EFFECTIVE][1:]],"a")
    b=action(c,h,d,[320,*parent[scale.EFFECTIVE][1:]],"b")
    with pytest.raises(ValueError,match="distinct"):path_search.audit_early([a,b],h,d,[],250)
    assert path_search.audit_early([a],h,d,[],50)["reduction_reason"]=="remaining_budget_below_100"
    assert path_search.audit_early([],h,d,[],0)["required_early_directions"]==0
    assert proposals[2]["target_scale_schedule"][2]!=proposals[2]["target_scale_schedule"][3]


def test_path_transitions_and_nonwinner_anchor_on_backtrack(tmp_path,monkeypatch):
    from agenticgen.uniguide.linear_admission import audit_actions
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch)
    best=choose(h)
    rounds=[dict(iteration=i,groups=[],search_focus=dict(window_id=3,operation="refine",source_id=best["id"]),
        checkpoint=dict(status="measured",performance=1.,input_incumbent_performance=.5)) for i in [1,2]]
    ctx=checkpoint_learning.context(c,h,rounds,d)
    assert ctx["rebase_from_best_checkpoint"]
    assert len(ctx["search_path_context"]["highlighted_prefixes"])==3
    for index,op in [(1,"backtrack"),(3,"refine"),(4,"advance")]:
        path_search.validate_focus(dict(window_id=index,operation=op),ctx["search_path_context"])
    with pytest.raises(ValueError):path_search.validate_focus(dict(window_id=1,operation="advance"),ctx["search_path_context"])
    a=five(c,h,d,learning=ctx)[0];a["anchor_id"]=next(r["id"] for r in h if r["id"]!=best["id"])
    assert audit_actions(c,h,d,[a],learning=ctx)["actions"][0]["admission"]["executable_now"]


@pytest.mark.parametrize('repair_success',[True,False])
def test_strict_transport_repairs_proposals_and_never_calls_synthesis(tmp_path,monkeypatch,repair_success):
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.providers.deepseek_tools import call_with_tools
    from agenticgen.providers.strict_tools import FINAL_TOOL
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch);proposals=five(c,h,d);seen=[]
    monkeypatch.setenv("DEEPSEEK_API_KEY","unit-not-a-key")
    skills=next(iter(assets(c).values())).parent
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        packet=json.loads(body["messages"][1]["content"])
        assert packet["final_submission_contract"]["payload_field"]=="proposals"
        assert "not mechanism, ablation" not in body["messages"][0]["content"]
        assert "Do not design ablations" not in json.dumps(packet)
        answer=dict(decision="propose",proposals=proposals,reason="Measured early opportunities.",
                    search_focus=dict(window_id=0,operation="start",source_id=h[0]["id"]))
        if len(seen)==1 or not repair_success:answer["proposals"]=proposals[-1:]
        return httpx.Response(200,json=dict(model="deepseek-flash",usage=dict(completion_tokens=20),choices=[dict(
            finish_reason="tool_calls",message=dict(role="assistant",content=None,reasoning_content="opaque",tool_calls=[dict(
                id="final"+str(len(seen)),type="function",function=dict(name=FINAL_TOOL,arguments=json.dumps(answer)))]))]))
    def invoke():
        return linear_council.advise(c,h,d,tmp_path/"advice",skills,provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    if not repair_success:
        from agenticgen.providers.deepseek_tools import SessionStopped
        with pytest.raises(SessionStopped,match='repair allowance exhausted'):invoke()
        assert len(seen)==2 and (tmp_path/'advice/proposal.session/FAILED.json').exists()
        assert not (tmp_path/'advice/DECISION.json').exists()
        assert not list((tmp_path/'advice').glob('synthesis*'))
        return
    result=invoke()
    assert result["selected"]==[a["id"] for a in proposals] and result["actions"]==proposals
    assert result["execution_source"]=="deterministic_all_proposals" and len(seen)==2
    assert "For proposals, supply proposals explicitly" in json.dumps(seen[1])
    assert not list((tmp_path/"advice").glob("synthesis*"))
    assert read(tmp_path/"advice/proposal.receipt.json")["interface_repairs"]==1
    assert path_search.verified_evidence_receipts(tmp_path/'advice',h,d)


def test_evidence_tools_use_fitted_unfiltered_mean(tmp_path,monkeypatch):
    from agenticgen.uniguide.analysis_tools import EvidenceTools
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch)
    registry=EvidenceTools(tmp_path,h,d,tmp_path/"tools")
    row=registry.execute("summarize_population",dict(run_id=h[0]["id"]))
    assert row["metrics"][fitted.METRIC]["arithmetic_mean"]==h[0][fitted.MEAN]
    assert "unfiltered" in row["ranking"]
    paired=registry.execute("compare_paired_runs",dict(reference_id=h[0]["id"],candidate_id=h[1]["id"]))
    delta=paired["metrics"][fitted.METRIC]
    assert delta["unfiltered_mean_change"]==pytest.approx(h[1][fitted.MEAN]-h[0][fitted.MEAN])
    assert "trimmed_mean_change" not in delta


def test_six_windows_new_assets_and_old_release_bindings(tmp_path):
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.uniguide.guidance_windows import expand
    from agenticgen.uniguide.geometry_sampling import scale_binding
    from agenticgen.uniguide.box_search import early_shape
    c=config(tmp_path);actual=[1,20,30,40,50,300]
    expanded=expand(actual,scale_binding(actual,window_bounds=c["windows"],early_shape=early_shape(c)))
    assert expanded[0]==expanded[99]==0 and expanded[51:99]==[300]*48
    assert set(assets(c))=={"analysis.md","proposal.md","computation.md"}
    root=Path(__file__).resolve().parents[2]/"src/agenticgen/resources/skill_releases/guidance"
    for version in ["2.12.0","2.13.0","2.14.0"]:
        for name,digest in read(root/version/"RELEASE.json")["files_sha256"].items():assert sha(root/version/name)==digest
    bad=copy.deepcopy(c);bad["box_exclusion"]={"enabled":True}
    with pytest.raises(ValueError):validate_config(bad)
    bad=copy.deepcopy(c);bad["windows"]=[*bad["windows"][:-1],[51,75],[76,99]]
    with pytest.raises(ValueError):validate_config(bad)
    from agenticgen.uniguide.geometry_reuse import select_initial, reuse
    with pytest.raises(ValueError,match='deterministic'):select_initial(c,{})
    with pytest.raises(ValueError,match='fresh'):reuse(c,{},{})


@pytest.mark.parametrize('numerical_failure',[False,True])
def test_five_rounds_execute_all_and_rank_the_retained_measurement(tmp_path,monkeypatch,numerical_failure):
    from agenticgen.uniguide import absolute_runtime
    fail=(lambda p,stage:stage=='formal' and p[0]==390) if numerical_failure else None
    c,s,d,h,_,executor,calls=setup(tmp_path,monkeypatch,fail)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda cfg:[s])
    monkeypatch.setattr(linear_runtime,'source_binding',lambda:{'fixture':'condition-path'})
    def advisor(cfg,history,domain,folder,skills,failures=(),final=False):
        learning=cfg['guidance_checkpoint_context']
        previous=learning['search_path_context']['previous_focus']
        focus=dict(window_id=0,operation='refine' if previous else 'start',source_id=choose(history)['id'])
        actions=[] if final else five(cfg,history,domain,int(folder.name[-3:]),learning)
        write(folder/'proposal.json',dict(proposals=actions,search_focus=focus))
        return dict(decision='hold' if final else 'propose',actions=actions,search_focus=focus,
                    reason='Measured complete profiles; fixture only.')
    request=linear_runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)
    result=request['current'];formal=[r for r in calls if r['stage']=='formal']
    assert len(formal)==25 and sum(r['count'] for r in formal)==1250
    assert result['complete']==(not numerical_failure)
    assert result['measured_or_reused_groups']==(24 if numerical_failure else 25)
    assert all(r['requested_actions']==5 and r['hypothesis_execution']['all_hypotheses_attempted'] for r in result['rounds'])
    summary=result['teachers'][s['teacher']['target']]
    assert primary_score(summary['ranking_measurement'])==choose(h)[fitted.MEAN]
    assert summary[fitted.MEAN]==choose(h)[fitted.MEAN]
    assert request['best_version_by_teacher'][s['teacher']['target']]=='GD01'
    assert request['next_version'] is None
    assert len(request['strategy_contexts'])==1 and 'proposal' in request['strategy_contexts'][0]
    assert not list((tmp_path/'versions/GD01').rglob('synthesis*'))
    assert (tmp_path/'versions/GD01/skills/references/search-policy.md').is_file()
    assert linear_runtime.run_version(c,'GD01',None,None,None,None)==request
    assert len(calls)==35


def test_failed_calibration_has_no_root_and_never_substitutes_pose_ranking(tmp_path,monkeypatch):
    c=config(tmp_path);s=dict(teacher=dict(target='6pyy_H7S_A_402'),condition=dict(condition_sha256='fixture'))
    executor,calls=executor_fixture(tmp_path,monkeypatch)
    def missing_fit(*args,**kwargs):
        row=executor(*args,**kwargs)
        for k in [fitted.CONFIG,fitted.MEAN,fitted.METRIC,RANKING,'coordinate_loss_statistics']:row.pop(k)
        return row
    assert path_calibration.run(c,[s],lambda *a:None,missing_fit)=={}
    assert len(calls)==2 and sum(r['count'] for r in calls)==100
    state=read(tmp_path/'calibration/path/6pyy_H7S_A_402/INSUFFICIENT_EVIDENCE.json')
    assert state['status']=='no_valid_root' and len(state['failures'])==2


def test_native_batch_numerical_failure_becomes_a_charged_failed_population(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_batch, geometry_sampling, linear_calibration
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch)
    job=dict(schedule=[390,*choose(h)[scale.EFFECTIVE][1:]],folder=tmp_path/'formal_failed',stage='formal',domain=d)
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='formal',teacher_id=s['teacher']['target']))
    def batch_failure(*args,**kwargs):
        Budget(tmp_path).reserve('slot_failed','formal',50,'fixture')
        raise FloatingPointError('injected native batch error')
    monkeypatch.setattr(geometry_batch,'execute_many',batch_failure)
    result=linear_calibration.evaluate(None,c,s,job,geometry_sampling.execute)
    assert isinstance(result,FloatingPointError)
    updated,failures=linear_runtime.history(c,s['teacher'],s['condition'],d)
    assert len(updated)==10 and len(failures)==1
    assert failures[0][scale.EFFECTIVE]==job['schedule']
    assert sum(r['count'] for r in read(tmp_path/'attempts.json').values() if r['stage']=='formal')==50
    with pytest.raises(ValueError,match='failed'):
        scale.resolve_action(action(c,h,d,job['schedule'],'repeat'),{r['id']:r for r in h},d,h,failures)


def test_preexecution_audit_rejects_bad_identity_evidence_and_cumulative_budget(tmp_path,monkeypatch):
    from agenticgen.uniguide.linear_admission import audit_actions
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch);proposals=five(c,h,d)
    for field,value in [('anchor_id','missing'),('evidence_ids',['invented']),('target_scale_schedule',[20]*7)]:
        invalid=copy.deepcopy(proposals);invalid[0][field]=value
        assert not audit_actions(c,h,d,invalid,portfolio=True)['batch_admission']['executable_now']
    invalid=copy.deepcopy(proposals);invalid[1]['id']=invalid[0]['id']
    assert not audit_actions(c,h,d,invalid,portfolio=True)['batch_admission']['executable_now']
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='formal',teacher_id=s['teacher']['target']))
    Budget(tmp_path).reserve('already_used','formal',1100,'fixture')
    report=audit_actions(c,h,d,proposals,portfolio=True)
    assert not report['batch_admission']['executable_now'] and report['budget_state']['remaining_formal_attempts']==150


def test_exact_legal_shortage_and_failed_round_backtracking(tmp_path,monkeypatch):
    import itertools
    c,s,d,h,_,_,_=setup(tmp_path,monkeypatch);parent=choose(h);p=parent[scale.EFFECTIVE]
    small=copy.deepcopy(d);small['search_grids']=[[v,v+10] for v in p]
    last=[p[0]+10,*p[1:]]
    failures=[{scale.EFFECTIVE:list(a)} for a in itertools.product(*small['search_grids']) if list(a)!=last]
    a=action(c,h,d,last,'last')
    report=path_search.audit_early([a],[parent],small,failures,250)
    assert report['available_profile_count']==1 and report['required_early_directions']==1
    assert report['reduction_reason']=='insufficient_legal_early_candidates'
    rounds=[dict(iteration=i,groups=[],checkpoint=checkpoint_learning.checkpoint([],parent,baseline=parent),
        search_focus=dict(window_id=2,operation='refine',source_id=parent['id'])) for i in [1,2]]
    assert checkpoint_learning.context(c,h,rounds,d)['rebase_from_best_checkpoint']


def test_native_shards_combine_all_fifty_condition_fits(tmp_path,monkeypatch,dynamic_condition):
    from test_uniguide_geometry_batch import setup_scheduler
    from agenticgen.uniguide import geometry_batch, geometry_sampling
    from agenticgen.uniguide.native_tools import save_pt
    _,teacher,_,model,calls,original_sample=setup_scheduler(tmp_path,monkeypatch)
    c=config(tmp_path)
    binding=dict(teacher_id=teacher['target'],condition_sha256=dynamic_condition['condition_sha256'],
                 condition_evaluation=c[fitted.CONFIG])
    for module in (geometry_batch,geometry_sampling):monkeypatch.setattr(module,'compatibility',lambda *a:binding)
    def sample(*args,**kwargs):
        output,_,_,control=original_sample(*args,**kwargs)
        folder,count=args[4:6]
        xyz=torch.tensor([[100.,30,0],[0,0,0],[1.4,0,0]],dtype=torch.float64)[None].repeat(count,1,1)
        xyz+=torch.arange(count,dtype=torch.float64)[:,None,None]/10+3
        output=dict(coords=xyz,mask=torch.ones(count,3,dtype=torch.bool))
        save_pt(folder/'terminal.pt.gz',output)
        with h5py.File(folder/'trace.h5','w') as trace:
            trace['guidance_delta_A']=np.zeros((100,count,3,3))
            trace['assignment']=np.tile([1,2],(100,count,1))
        return output,None,None,control
    job=dict(schedule=[50]*6,folder=tmp_path/'population',stage='calibration',count=50)
    row=geometry_batch.execute_many(model,c,teacher,dynamic_condition,[job],sample_fn=sample)[0]
    assert primary_score(row)<1e-12 and row[RMSD]>1 and row[fitted.POSE]==row[RMSD]
    assert row['attempts']==50 and row['built']==0 and row['coordinate_loss_statistics']['retained_count']==50
    records=read(tmp_path/'population/records.json')
    assert len(records)==50 and all(r[fitted.METRIC]<1e-12 for r in records)
    assert calls==[50] and sum(r['count'] for r in read(tmp_path/'attempts.json').values())==50
