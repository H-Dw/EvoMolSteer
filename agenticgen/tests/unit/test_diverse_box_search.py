"""Expanded search: real strict transport and disk budgets, no API/GPU sampling."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import jsonschema
import pytest
import torch

from agenticgen.uniguide import box_search, box_gates, linear_scale as scale
from agenticgen.uniguide import linear_calibration as calibration, linear_runtime as runtime
from agenticgen.uniguide import checkpoint_learning, profile_diversity, linear_council
from agenticgen.uniguide.absolute_runtime import validate_config
from agenticgen.uniguide.absolute_coordinate import RMSD,MAX_ERROR
from agenticgen.uniguide.coordinate_statistics import aggregate_fields,RANKING
from agenticgen.uniguide.io import read,write,sha,value_sha,Budget
from agenticgen.uniguide.tool_runtime import assets


def config(root,shape='constant'):
    folder=Path(__file__).resolve().parents[2]/'configs'
    c=read(folder/'uniguide_absolute_coordinate.json')
    c.update(read(folder/'uniguide_diverse_box_search.json'))
    c['box_search']['early_guidance_shape']=shape
    c.update(root=str(root),generation_context_root=str(root.parent/'frozen'))
    return validate_config(c)


def install_executor(root,monkeypatch,score=None,fail=None):
    from agenticgen.uniguide import geometry_sampling
    calls=[];measured={}
    def binding(c,t,k):
        return dict(teacher_id=t['target'],condition_sha256=k['condition_sha256'],
            guidance_box_mask=box_gates.mask(c),early_guidance_shape=box_search.early_shape(c))
    monkeypatch.setattr(geometry_sampling,'compatibility',binding)
    monkeypatch.setattr(runtime,'compatibility',binding)
    def executor(model,c,t,k,schedule,folder,stage,count=50,domain=None):
        actual=scale.effective(schedule,domain) if domain else schedule
        key=value_sha([actual,box_gates.mask(c)])
        delta=count-measured.get(key,0);assert delta>0
        Budget(root).reserve(folder.relative_to(root).as_posix(),stage,delta,key)
        measured[key]=count;calls.append(dict(actual=actual,count=count,stage=stage,delta=delta))
        identity=dict(compatibility=binding(c,t,k),**{scale.EFFECTIVE:actual},population=count)
        fingerprint=value_sha(identity)
        write(root/'measurement_registry'/(fingerprint+'.json'),dict(identity=identity,source_group=folder.relative_to(root).as_posix()))
        if fail and fail(actual,stage):
            write(folder/'FAILED.json',dict(error='injected nonfinite fixture',no_retry=True))
            raise FloatingPointError('injected nonfinite fixture')
        error=score(actual,box_gates.mask(c),count) if score else .3+sum(actual)/100000
        records=[dict(node_id=f'{seed}_{slot}',logical_seed=seed,logical_slot=slot,**{RMSD:error,MAX_ERROR:error*2})
            for seed in range(5) for slot in range(count//5)]
        row=dict(id=folder.relative_to(root).as_posix(),teacher_id=t['target'],condition_sha256=k['condition_sha256'],
            complete=True,attempts=count,built=count,connected=count,unique_smiles=1,
            compatibility_sha256=value_sha(identity['compatibility']),fingerprint=fingerprint,
            **{scale.EFFECTIVE:actual,RMSD:error,MAX_ERROR:error*2},**aggregate_fields(records,c),
            window_diagnostics=[dict(increment_vector_rms_A=1,max_atom_increment_A=1) for _ in range(6)])
        write(folder/'records.json',records);write(folder/'aggregate.json',row)
        write(folder/'manifest.json',dict(identity=identity));write(folder/'steps.json',[])
        write(folder/'COMPLETE.json',dict(files={n:sha(folder/n) for n in ['records.json','aggregate.json','manifest.json','steps.json']}))
        return row
    return executor,calls,binding


def prepared(tmp_path,monkeypatch,fail=None):
    c=config(tmp_path)
    executor,calls,binding=install_executor(tmp_path,monkeypatch,fail=fail)
    s=dict(teacher=dict(target='6pyy_H7S_A_402'),condition=dict(condition_sha256='condition_fixture'))
    calibration.ensure_limits(c,1);write(tmp_path/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id=s['teacher']['target']))
    base=executor(None,c,s['teacher'],s['condition'],[50,100,40,40,30,50],tmp_path/'baseline','calibration')
    protocol=value_sha(binding(c,s['teacher'],s['condition']))
    body=dict(scale_policy=scale.OPEN_POLICY,teacher_id=s['teacher']['target'],condition_sha256='condition_fixture',
        sampling_protocol_sha256=protocol,calibration_key=value_sha([s['teacher']['target'],'condition_fixture',protocol]),
        control_settings=c['guidance_control'],calibrated_guidance_reference_scale=0,guidance_scale_grid_unit=1,
        initial_source=base['id'],initial_guidance_scale_tick_schedule=base[scale.EFFECTIVE],
        guidance_windows=c['windows'],bootstrap_sources=[base['id']],confirmed_effective_levels=sorted(set(base[scale.EFFECTIVE])),
        confirmed_profiles=[dict(source_id=base['id'],actual_schedule=base[scale.EFFECTIVE])],
        source_rows_sha256=value_sha([base]),evidence_revision=0,
        exploration_policy=c['guidance_exploration'],initial_calibration_performance=base[RANKING],
        box_exclusion_policy=box_search.POLICY,guidance_box_mask=[True]*6,selection_sha256='fixture',
        search_grids=box_search.grids(c),early_guidance_shape=box_search.early_shape(c),execute_all_proposed=True)
    d=scale.seal(body);scale.validate_domain(d)
    dest=tmp_path/'calibration/frozen'/d['calibration_key'];write(dest/'domain.json',d)
    write(dest/'COMPLETE.json',dict(domain_sha256=d['domain_sha256'],measurement_receipts={base['id']:sha(tmp_path/'baseline/COMPLETE.json')}))
    write(tmp_path/'calibration/domains.json',{s['teacher']['target']:d})
    return c,s,d,base,executor,calls


def action(c,h,d,actual,identifier,learning=None):
    ctx=learning or checkpoint_learning.context(c,h,[],domain=d)
    return dict(id=identifier,anchor_id=ctx['best_checkpoint_id'],target_scale_schedule=actual,
        exploration_tier=box_search.classify(actual,ctx['best_checkpoint_actual_schedule'],d),
        improvement_reason='Explore an unmeasured complete combination.',uncertainty='Actual improvement is unknown.',
        evidence_ids=[d['initial_source']],optimization_target=dict(calibration_source_id=d['initial_source'],
            incumbent_source_id=ctx['best_checkpoint_id'],reviewed_path_sha256=ctx['best_path_sha256'],
            purpose='seek_potential_global_improvement',path_relation='redirect_from_reviewed_best' if ctx['rebase_from_best_checkpoint'] else 'new_joint_region'))


def five(c,h,d,iteration=1,learning=None):
    # Different changed-window sets, including one small-dose bracket and a distant joint profile.
    doses=[0,1,2,5,10,20,30,40,60,70]
    vectors=[[v,100,40,40,30,50] for v in doses[2*(iteration-1):2*iteration]]
    vectors += [[50,100+10*iteration,40,40,30,50],
                [50,100,40+10*iteration,40,30,50],
                [50,100,40,40,200+10*iteration,250+10*iteration]]
    return [action(c,h,d,v,f'round{iteration}_branch{i}',learning) for i,v in enumerate(vectors)]


def test_expanded_grids_reach_both_400_windows_and_keep_integer_low_doses(tmp_path,monkeypatch):
    c,s,d,b,_,_=prepared(tmp_path,monkeypatch)
    assert c['guidance_control']['maximum_selected']==c['guidance_control']['maximum_proposals']==5
    assert all(v in d['search_grids'][0] for v in [0,1,2,5,10,400])
    assert 1 not in d['search_grids'][1]
    assert scale.effective(scale.encode([1,100,40,40,400,400],d),d)==[1,100,40,40,400,400]
    a=action(c,[b],d,[1,100,40,40,400,400],'beyond_old_domain')
    assert scale.resolve_action(a,{b['id']:b},d,[b])[scale.EFFECTIVE][-2:]==[400,400]
    for invalid in ([1,1,40,40,30,50],[1,100,40,40,410,400]):
        with pytest.raises(ValueError):scale.encode(invalid,d)
    assert read(tmp_path/'ATTEMPT_LIMITS.json')['per_teacher_stage_caps']['formal']==1250
    assert len(assets(c))==5


def test_teacher_grid_cannot_silently_drop_required_preexperiment_profiles(tmp_path):
    c=config(tmp_path)
    c['box_search']['teacher_window_grids']['3w54_RNB_A_504']=[list(g) for g in c['box_search']['window_grids']]
    c['box_search']['teacher_window_grids']['3w54_RNB_A_504'][4].remove(200)
    with pytest.raises(ValueError,match='window grid'):validate_config(c)


def test_coverage_bank_and_real_first_dose_brackets_are_not_near_duplicate_errors(tmp_path,monkeypatch):
    c,s,d,b,_,_=prepared(tmp_path,monkeypatch)
    bank=profile_diversity.bank([b],d,b[scale.EFFECTIVE])
    assert any(any(v>200 for v in r['target_scale_schedule']) for r in bank)
    assert any(r['search_opportunity']=='early_dose' for r in bank)
    actions=five(c,[b],d)
    profile_diversity.require_portfolio(actions,d,b[scale.EFFECTIVE],bank,1250)
    with pytest.raises(ValueError):profile_diversity.require_portfolio(actions+[actions[0]],d,b[scale.EFFECTIVE],bank,1250)
    with pytest.raises(ValueError):profile_diversity.require_portfolio([actions[0],actions[0]],d,b[scale.EFFECTIVE],bank,1250)


def test_actual_native_strict_council_retains_all_five_and_repairs_downselection_once(tmp_path,monkeypatch):
    from agenticgen.providers.deepseek_tools import call_with_tools
    from agenticgen.providers.strict_tools import FINAL_TOOL
    c,s,d,b,_,_=prepared(tmp_path,monkeypatch)
    proposed=five(c,[b],d);seen=[];synthesis_calls=[]
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-not-a-key')
    skills=tmp_path/'skills';skills.mkdir()
    for name,path in assets(c).items():(skills/name).write_bytes(path.read_bytes())
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        assert all(t['function']['strict'] for t in body['tools'])
        ctx=json.loads(body['messages'][1]['content'])
        assert ctx['execution_contract']['maximum_selected']==5
        assert ctx['window_evidence']['coverage'][0]['unmeasured_levels_at_current_background']
        if ctx['final_submission_contract']['payload_field']=='proposals':
            answer=dict(decision='propose',proposals=proposed,reason='Five different potentially better complete directions.')
        else:
            synthesis_calls.append(1)
            answer=dict(decision='propose',selected=[a['id'] for a in proposed[:3 if len(synthesis_calls)==1 else 5]],reason='Execute the hypotheses.')
        return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=20),choices=[dict(
            finish_reason='tool_calls',message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(
                id='final_'+str(len(seen)),type='function',function=dict(name=FINAL_TOOL,arguments=json.dumps(answer)))]))]))
    ledger=(tmp_path/'attempts.json').read_bytes()
    result=linear_council.advise(c,[b],d,tmp_path/'advice',skills,provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    assert len(result['actions'])==5 and set(result['selected'])=={a['id'] for a in proposed}
    assert len(seen)==3 and read(tmp_path/'advice/synthesis.receipt.json')['interface_repairs']==1
    assert (tmp_path/'attempts.json').read_bytes()==ledger


@pytest.mark.parametrize('numerical_failure',[False,True])
def test_five_rounds_execute_all_25_hypotheses_charge_1250_and_preserve_failures(tmp_path,monkeypatch,numerical_failure):
    from agenticgen.uniguide import absolute_runtime
    fail=(lambda a,stage:stage=='formal' and a[0]==0) if numerical_failure else None
    c,s,d,b,executor,calls=prepared(tmp_path,monkeypatch,fail=fail)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda config:[s])
    monkeypatch.setattr(runtime,'source_binding',lambda:{'fixture':'actual-disk-contract'})
    def advisor(cfg,h,domain,folder,skills,failures=(),final=False):
        actions=[] if final else five(cfg,h,domain,int(folder.name[-3:]),cfg['guidance_checkpoint_context'])
        for role in ('proposal','synthesis'):write(folder/(role+'.json'),dict(decision='hold' if final else 'propose',reason='Offline fixture.'))
        return dict(decision='hold' if final else 'propose',actions=actions,reason='Execute all different hypotheses.')
    result=runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)['current']
    formal=[r for r in calls if r['stage']=='formal']
    assert len(formal)==25 and sum(r['delta'] for r in formal)==1250
    assert all(r['requested_actions']==5 and r['hypothesis_execution']['all_hypotheses_attempted'] for r in result['rounds'])
    assert result['complete']==(not numerical_failure)
    assert result['measured_or_reused_groups']==(24 if numerical_failure else 25)
    assert (tmp_path/'versions/GD01'/s['teacher']['target']/'iteration_001/action_001/HYPOTHESIS_FAILED.json').exists()==numerical_failure
    with pytest.raises(ValueError,match='GD01'):runtime.run_version(c,'GD02',lambda *a:None,executor,advisor,lambda *a:None)


def test_v2_preexperiment_adds_weak_and_high_probes_without_exceeding_500(tmp_path,monkeypatch):
    from agenticgen.uniguide import box_exclusion
    c=config(tmp_path);target='6pyy_H7S_A_402'
    executor,calls,binding=install_executor(tmp_path,monkeypatch,score=lambda a,m,n:.5-.01*sum(m)+(.01 if a[0]>5 else 0))
    s=dict(teacher=dict(target=target),condition=dict(condition_sha256='condition_fixture'))
    calibration.ensure_limits(c,1)
    def provider(messages,schema,out,**kw):
        packet=json.loads(messages[1]['content']);data=packet['evidence']
        answer=dict(excluded_box_ids=[],evidence_ids=[data['evidence_id']],reason='Missing or contradictory negative proof; keep the boxes.')
        if packet['phase']=='final':answer['confirmed_profile_id']=min(data['confirmed_profiles'],key=lambda n:data['confirmed_profiles'][n][RANKING])
        kw['response_validator'](answer);write(out,answer);return answer
    selection,rows=box_exclusion.Search(c,s,None,executor,provider).run()
    assert selection['consumed_attempts']==500 and sum(r['delta'] for r in calls)==500
    assert any(r['actual'][0]==1 for r in calls)
    assert any(max(r['actual'])>200 for r in calls)
    assert selection['initial_actual_schedule'][0]<=5 and selection['guidance_box_mask']==[True]*6
    assert all(r['attempts']==50 for r in rows)


def test_early_ramp_scales_native_endpoints_chemistry_and_batch_agree(tmp_path):
    from agenticgen.uniguide.guidance_windows import expand
    from agenticgen.uniguide.geometry_sampling import GeometryControl,scale_binding
    from agenticgen.uniguide.geometry_batch import CandidateControl
    from agenticgen.uniguide.geometry_dynamics import policy
    c=config(tmp_path,'linear_ramp');shape=box_search.early_shape(c)
    b=scale_binding([50]*6,window_bounds=c['windows'],early_shape=shape)
    actual=expand([50]*6,b)
    assert actual[:11]==[0,5,10,15,20,25,30,35,40,45,50] and actual[99]==0
    unit=object.__new__(GeometryControl);unit.schedule=actual;unit.absolute_config=c
    batched=object.__new__(CandidateControl);batched.absolute_config=c;batched.units=[{},{}];batched.failed_units={};batched.expanded=[actual,actual]
    for step in range(100):assert batched.step_scales(step,20)==unit.step_scales(step,10)*2
    p=policy(c);assert p['early_guidance_shape']==shape
    assert p['chemical_inpainting']['step_bounds']==[1,98]
    assert policy(config(tmp_path/'constant'))!=p


def test_release_payload_and_previous_release_remain_byte_bound():
    base=Path(__file__).resolve().parents[2]/'src/agenticgen/resources/skill_releases/guidance'
    root=base/'2.12.0';manifest=read(root/'RELEASE.json')
    for name,digest in manifest['files_sha256'].items():assert sha(root/name)==digest
    for name,digest in manifest['old_files_sha256'].items():assert sha(base/'2.11.0'/name)==digest


def test_readonly_audit_tool_uses_five_branch_cumulative_admission(tmp_path,monkeypatch):
    from agenticgen.uniguide.analysis_tools import EvidenceTools
    c,s,d,b,_,_=prepared(tmp_path,monkeypatch)
    registry=EvidenceTools(tmp_path,[b],d,tmp_path/'calculations')
    registry.action_config=c
    registry.action_learning=checkpoint_learning.context(c,[b],[],domain=d)
    proposals=five(c,[b],d)
    before=(tmp_path/'attempts.json').read_bytes()
    result=registry.execute('audit_action_batch',{'actions':proposals})
    assert result['evaluation_scope']=='selected_portfolio'
    assert result['batch_admission']['formal_new_attempts']==250
    assert result['batch_admission']['executable_now']
    assert (tmp_path/'attempts.json').read_bytes()==before
    definition=next(r['function'] for r in registry.definitions if r['function']['name']=='audit_action_batch')
    assert definition['parameters']['properties']['actions']['maxItems']==5


@pytest.mark.parametrize('wrong_shape',[False,True])
def test_full_early_shape_acceptance_catches_constant_used_instead_of_ramp(tmp_path,monkeypatch,wrong_shape):
    import h5py
    import numpy as np
    from agenticgen.uniguide import early_shape_acceptance
    c=config(tmp_path,'linear_ramp')
    def sample(model,cfg,teacher,schedule,folder,count,control_factory,**kwargs):
        assert control_factory.keywords['scale_binding']['early_guidance_shape']['policy']=='linear_ramp'
        actual=[0,*([50]*10 if wrong_shape else list(range(5,51,5))),*[50]*88,0]
        rows=[dict(step=i,applied_guidance_scales=[v]*10,chemical_inpainting_applied=[1<=i<=98]*10) for i,v in enumerate(actual)]
        folder.mkdir(parents=True,exist_ok=True);write(folder/'steps.json',rows)
        with h5py.File(folder/'trace.h5','w') as trace:
            for name in ['input_gradient','guidance_delta_A','native_proposal','guided_proposal']:
                trace.create_dataset(name,data=np.zeros((100,10,1,3)))
    monkeypatch.setattr(early_shape_acceptance,'sample',sample)
    if wrong_shape:
        with pytest.raises(ValueError,match='applied dose'):early_shape_acceptance.run_teacher(None,c,{},None,tmp_path/'probe')
    else:
        result=early_shape_acceptance.run_teacher(None,c,{},None,tmp_path/'probe')
        assert result['first_box_actual_scales']==[0,*range(5,51,5)]


def test_window_evidence_rejects_changed_saved_steps_and_keeps_unknown_context(tmp_path,monkeypatch):
    from agenticgen.uniguide.window_evidence import context
    c,s,d,b,_,_=prepared(tmp_path,monkeypatch)
    evidence=context(c,[b],d,b)
    assert evidence['coverage'][0]['observed_levels_at_current_background']==[50]
    assert 1 in evidence['coverage'][0]['unmeasured_levels_at_current_background']
    assert evidence['incumbent_update_diagnostics']['status']=='unavailable'
    write(tmp_path/'baseline/steps.json',[{'modified':True}])
    with pytest.raises(ValueError,match='source changed'):context(c,[b],d,b)


@pytest.mark.parametrize('harmful_boxes',[(0,1),(1,2)])
def test_v2_negative_confirmations_and_optional_joint_fit_the_same_500_ledger(tmp_path,monkeypatch,harmful_boxes):
    from agenticgen.uniguide import box_exclusion
    c=config(tmp_path)
    executor,calls,_=install_executor(tmp_path,monkeypatch,
        score=lambda a,m,n:.4+sum(.1*m[i] if i in harmful_boxes else -.02*m[i] for i in range(6)))
    subject=dict(teacher=dict(target='6pyy_H7S_A_402'),condition=dict(condition_sha256='fixture_condition'))
    calibration.ensure_limits(c,1)
    def provider(messages,schema,out,**kw):
        data=json.loads(messages[1]['content'])['evidence']
        result=dict(excluded_box_ids=data['eligible_box_ids'],evidence_ids=[data['evidence_id']],reason='Use only the actually confirmed negative evidence.')
        if 'confirmed_profile_id' in schema['properties']:
            result['confirmed_profile_id']=next(n for n,mask in data['confirmed_options'].items() if mask==result['excluded_box_ids'])
        kw['response_validator'](result);write(out,result);return result
    selection,_=box_exclusion.Search(c,subject,None,executor,provider).run()
    assert selection['excluded_box_ids']==([1] if harmful_boxes==(0,1) else [1,2])
    assert selection['consumed_attempts']<=500 and sum(r['delta'] for r in calls)==selection['consumed_attempts']
