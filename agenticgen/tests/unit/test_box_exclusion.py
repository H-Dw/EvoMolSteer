from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import httpx

from agenticgen.uniguide import box_gates as gates, box_exclusion as exclusion
from agenticgen.uniguide import profile_diversity as diversity, linear_scale as scale
from agenticgen.uniguide.absolute_coordinate import RMSD
from agenticgen.uniguide.io import read, write, sha, value_sha, Budget
from agenticgen.uniguide.coordinate_statistics import summarize, SUMMARY, RANKING
from test_zero_start_guidance import config as old_config
from test_dynamic_matching_jump import dynamic_condition
from test_absolute_coordinate_protocol import condition


def config(root):
    c=old_config(root)
    c['agent_tools']['skill_release']='2.11.0'
    c['guidance_control']['maximum_window_tick_change']=12
    c['box_exclusion']=dict(enabled=True,policy=gates.POLICY)
    from agenticgen.uniguide.absolute_runtime import validate_config
    return validate_config(c)


def records(value=1,changes=None,count=50):
    return [dict(logical_seed=s,logical_slot=k,**{RMSD:value+(changes[s] if changes else 0)})
            for s in range(5) for k in range(count//5)]


def test_teacher_profiles_and_zero_controls_are_distinct():
    for target in exclusion.BACKGROUNDS:
        p=exclusion.profiles(target)
        assert len(p)==29
        assert len({value_sha([x['actual'],x['guidance_box_mask']]) for x in p.values()})==29
        assert p['NativeControl']['actual']==p['ChemistryOnly']['actual']==[0]*6
        assert p['NativeControl']['guidance_box_mask']!=p['ChemistryOnly']['guidance_box_mask']
        for label,actual in exclusion.BACKGROUNDS[target].items():
            for i in range(6):
                off=p[f'{label}_NativeOff_{i}'];low=p[f'{label}_LowerRescue_{i}']
                assert off['actual'][i]==0 and not off['guidance_box_mask'][i]
                assert low['actual'][i]==max(10,actual[i]-20)
                assert all(v==actual[j] for j,v in enumerate(off['actual']) if i!=j)


def test_negative_evidence_requires_material_effect_and_seed_consistency():
    good=exclusion.paired_effect(records(1.05),records(1))
    assert exclusion.full_support(good)
    assert not exclusion.full_support(exclusion.paired_effect(records(1.001),records(1)))
    assert not exclusion.full_support(exclusion.paired_effect(records(1,{0:1,1:-.01,2:-.01,3:-.01,4:-.01}),records(1)))
    assert not exclusion.full_support(exclusion.paired_effect(records(.9),records(1)))
    assert not exclusion.full_support(exclusion.paired_effect(records(1.05,count=10),records(1,count=10)))
    bad=records(1);bad[0][RMSD]=float('nan')
    assert exclusion.paired_effect(bad,records(1))['status']=='unavailable'


def test_gate_stops_vjp_and_chemistry_but_keeps_allowed_zero_semantics(tmp_path):
    c=config(tmp_path/'run');c['guidance_box_mask']=[False,True,True,True,True,True]
    for step in range(100):
        expected=11<=step<=98
        assert gates.step_enabled(c,step)==expected
        assert gates.apply([50]*10,step,c)==([50]*10 if expected else [0.]*10)
    from agenticgen.uniguide.geometry_chemistry import ChemicalStateInpainting
    unit=object.__new__(ChemicalStateInpainting);unit.absolute_config=c
    unit.sampling_policy={'chemical_inpainting':{'enabled':True,'step_bounds':[1,98]}}
    unit.rows=[{}];state={'mask':torch.ones(10,3)}
    assert unit.apply_chemical_inpainting(state,5) is state
    assert unit.rows[-1]['chemical_inpainting_applied']==[False]*10
    with pytest.raises(ValueError):gates.validate_profile(c,[10]*6)
    gates.validate_profile(c,[0,10,10,10,10,10])


def test_actual_forward_native_gate_skips_autograd(dynamic_condition,tmp_path):
    from agenticgen.uniguide.geometry_sampling import GeometryControl,scale_binding
    from agenticgen.uniguide import absolute_coordinate as coordinates
    from agenticgen.uniguide.geometry_dynamics import policy
    c=config(tmp_path/'run');c['guidance_box_mask']=[False,True,True,True,True,True]
    calls=[]
    class Model:
        coord_scale=1.
        integrator=SimpleNamespace(noise_scheduler=lambda t:t.new_tensor(0.),coord_noise_level=.2)
        def __call__(self,current,*a,**kw):
            calls.append(torch.is_grad_enabled());return dict(coords=current['coords']*1.01+.01)
        def _get_predictions(self,out):return out,None
    u=object.__new__(GeometryControl);u.model=Model();u.absolute_config=c;u.geometry_condition=dynamic_condition
    u.scale_binding=scale_binding([0,50,50,50,50,50],window_bounds=c['windows'])
    u.schedule=u.expand_schedule([0,50,50,50,50,50]);u.sampling_policy=policy(c)
    u.prefix_policy={'enabled':False};u.com=torch.zeros(10,3);u.rows=[];u.resume=None;u.folder=tmp_path
    u.last_assignment=None;u.random_streams=SimpleNamespace(states=[])
    u.absolute_state=coordinates.AssignmentState(u);u.store=lambda *a:None
    curr=dict(coords=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[9.,9.,9.]]]).repeat(10,1,1),mask=torch.ones(10,3,dtype=torch.bool))
    for step in range(100):
        u.forward(curr,{},[torch.full((10,),step*.00999)]*3,None,None,None,step,.00999)
        assert calls[-1]==(11<=step<=98)
        if step<=10 or step==99:assert not torch.count_nonzero(u.delta)


def test_absolute_profile_diversity_is_measured_from_real_best():
    d={'guidance_box_mask':[False,True,True,True,True,True]}
    best=[0,40,40,40,40,40]
    history=[{scale.EFFECTIVE:best}]
    bank=diversity.bank(history,d,best)
    assert any(r['exploration_tier']=='bold' for r in bank)
    assert all(r['target_scale_schedule'][0]==0 for r in bank)
    with pytest.raises(ValueError):diversity.require_portfolio([{'target_scale_schedule':[0,50,40,40,40,40]}],d,best,bank,750)
    chosen=[{'target_scale_schedule':next(r['target_scale_schedule'] for r in bank if r['exploration_tier']=='bold')}]
    diversity.require_portfolio(chosen,d,best,bank,750)
    with pytest.raises(ValueError):diversity.require_portfolio(chosen+chosen,d,best,bank,750)


@pytest.mark.parametrize('scenario,expected,charged',[('normal',[0,1],500),('low_rescue',[1],450),('combined_bad',[0],500),('no_harm',[],370)])
def test_search_real_budget_promotion_and_strict_empty_or_negative_selection(tmp_path,monkeypatch,scenario,expected,charged):
    c=config(tmp_path/'run');root=Path(c['root'])
    from agenticgen.uniguide.linear_calibration import ensure_limits
    ensure_limits(c,1)
    target='6pyy_H7S_A_402';subject={'teacher':{'target':target},'condition':{'condition_sha256':'testcondition'}}
    seen={};calls=[]
    def compatible_binding(local,teacher,condition):
        return dict(teacher_id=target,condition_sha256='testcondition',guidance_box_mask=gates.mask(local),protocol='unit_fixture')
    def executor(model,local,teacher,condition,schedule,folder,stage,count=50,**kw):
        key=value_sha([schedule,gates.mask(local)])
        delta=count-seen.get(key,0);assert delta>0
        Budget(root).reserve(str(folder.relative_to(root)),stage,delta,key);seen[key]=count
        effect=[.1,.1,-.05,-.05,-.05,-.05] if scenario!='no_harm' else [-.05]*6
        if scenario=='low_rescue' and schedule[0]==30:effect[0]=-.1
        score=.4+sum(v*on for v,on in zip(effect,gates.mask(local)))+sum(schedule)*.0001
        if scenario=='combined_bad' and gates.mask(local)==[False,False,True,True,True,True]:score+=.4
        rows=records(score,count=count)
        stats=summarize([r[RMSD] for r in rows])
        row=dict(id=folder.relative_to(root).as_posix(),complete=True,attempts=count,
            teacher_id=target,condition_sha256='testcondition',compatibility_sha256=value_sha(compatible_binding(local,teacher,condition)),
            fingerprint=value_sha([key,count]),
            **{scale.EFFECTIVE:schedule,RMSD:score,RANKING:stats['upper_tail_filtered_mean'],SUMMARY:stats},
            window_diagnostics=[dict(increment_vector_rms_A=1,max_atom_increment_A=1) for _ in range(6)])
        write(folder/'records.json',rows);write(folder/'aggregate.json',row)
        write(folder/'manifest.json',dict(identity=dict(compatibility=compatible_binding(local,teacher,condition))))
        write(folder/'steps.json',[])
        write(folder/'COMPLETE.json',{'files':{n:sha(folder/n) for n in ['records.json','aggregate.json','manifest.json','steps.json']}})
        write(root/'measurement_registry'/(row['fingerprint']+'.json'),dict(source_group=row['id'],
            identity=dict(compatibility=compatible_binding(local,teacher,condition),**{scale.EFFECTIVE:schedule},population=count)))
        calls.append((gates.mask(local),list(schedule),count,delta));return row
    def provider(messages,schema,output,**kw):
        context=json.loads(messages[1]['content']);data=context['evidence']
        assert kw['final_response_mode']=='strict_tool_v1'
        result={'excluded_box_ids':data['eligible_box_ids'],'evidence_ids':[data['evidence_id']],'reason':'Paired negative evidence; no mechanism claim.'}
        if context['phase']=='final':
            options=data['confirmed_options']
            if result['excluded_box_ids'] not in options.values():result['excluded_box_ids']=expected
            result['confirmed_profile_id']=next(n for n,e in options.items() if e==result['excluded_box_ids'])
        kw['response_validator'](result);write(output,result);return result
    search=exclusion.Search(c,subject,None,executor,provider)
    selected,compatible=search.run()
    assert selected['excluded_box_ids']==expected
    assert selected['consumed_attempts']==charged
    assert sum(r[3] for r in calls)==charged
    assert compatible and all(r['attempts']==50 for r in compatible)
    if scenario=='normal':
        from agenticgen.uniguide import geometry_sampling,linear_runtime
        from agenticgen.uniguide.checkpoint_learning import context
        from agenticgen.uniguide.linear_admission import audit_actions
        monkeypatch.setattr(geometry_sampling,'compatibility',compatible_binding)
        monkeypatch.setattr(linear_runtime,'compatibility',compatible_binding)
        monkeypatch.setattr(exclusion.Search,'run',lambda self:(selected,compatible))
        domains=exclusion.run(c,[subject],lambda *args:None,executor)
        domain=domains[target];local=dict(c,guidance_box_mask=domain['guidance_box_mask'])
        history,failures=linear_runtime.history(local,subject['teacher'],subject['condition'],domain)
        assert domain['initial_source'] in {r['id'] for r in history}
        assert all(r[scale.EFFECTIVE][:2]==[0,0] for r in history)
        assert domain['initial_guidance_scale_tick_schedule']!=[0]*6
        learning=context(local,history,[],domain=domain)
        suggestions=diversity.bank(history,domain,learning['best_checkpoint_actual_schedule'],failures)
        chosen=next(r for r in suggestions if r['exploration_tier']=='bold')
        action=dict(id='unit_joint_profile',anchor_id=learning['best_checkpoint_id'],
            target_scale_schedule=chosen['target_scale_schedule'],exploration_tier='bold',
            evidence_ids=[domain['initial_source']],improvement_reason='Unknown joint opportunity',uncertainty='Unmeasured',
            optimization_target=dict(calibration_source_id=domain['initial_source'],
                incumbent_source_id=learning['best_checkpoint_id'],reviewed_path_sha256=learning['best_path_sha256'],
                purpose='seek_potential_global_improvement',path_relation='new_joint_region'))
        report=audit_actions(local,history,domain,[action],failures,learning,portfolio=True)
        assert report['batch_admission']['executable_now']
        assert report['actions'][0]['resolved'][scale.EFFECTIVE]==action['target_scale_schedule']
        from agenticgen.uniguide import linear_council
        from agenticgen.uniguide.tool_runtime import assets
        from agenticgen.providers.deepseek_tools import call_with_tools
        from agenticgen.providers.strict_tools import FINAL_TOOL,ENDPOINT
        skills=tmp_path/'skills';skills.mkdir()
        for name,path in assets(local).items():(skills/name).write_bytes(path.read_bytes())
        monkeypatch.setenv('DEEPSEEK_API_KEY','unit-not-a-key')
        seen=[]
        def handle(request):
            assert str(request.url)==ENDPOINT
            body=json.loads(request.content);seen.append(body)
            assert all(t['function']['strict'] for t in body['tools'])
            ctx=json.loads(body['messages'][1]['content'])
            assert ctx['box_exclusion_contract']['guidance_box_mask']==domain['guidance_box_mask']
            assert ctx['coverage_profile_suggestions'] and all(r['performance'] is None for r in ctx['coverage_profile_suggestions'])
            payload=(dict(decision='propose',proposals=[action],reason='Explore a distinct unmeasured complete profile.')
                     if ctx['final_submission_contract']['payload_field']=='proposals' else
                     dict(decision='propose',selected=[action['id']],reason='Choose an executable joint bold profile.'))
            return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=20),choices=[dict(
                finish_reason='tool_calls',message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(
                    id='submit_'+str(len(seen)),type='function',function=dict(name=FINAL_TOOL,arguments=json.dumps(payload)))]))]))
        before=(root/'attempts.json').read_bytes()
        result=linear_council.advise(local,history,domain,tmp_path/'advice',skills,failures,
            provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
        assert result['selected']==[action['id']] and len(seen)==2
        assert (root/'attempts.json').read_bytes()==before


@pytest.mark.parametrize('phase',['proposal','final'])
def test_selection_native_tool_and_one_complete_field_repair(tmp_path,monkeypatch,phase):
    from agenticgen.providers.deepseek_tools import RoundBudget,call_with_tools
    from agenticgen.providers.strict_tools import FINAL_TOOL,ENDPOINT
    monkeypatch.setenv('DEEPSEEK_API_KEY','unit-not-a-key')
    c=config(tmp_path/'run');source=tmp_path/'source.json';write(source,dict(frozen=True))
    data=dict(eligible_box_ids=[0],confirmed_options={'UniformRef':[],'UniformRef_NativeOff_0':[0]})
    seen=[]
    def handle(request):
        assert str(request.url)==ENDPOINT
        body=json.loads(request.content);seen.append(body)
        assert all(t['function']['strict'] for t in body['tools'])
        ctx=json.loads(body['messages'][1]['content'])
        payload=dict(excluded_box_ids=[0],evidence_ids=[ctx['evidence']['evidence_id']],reason='Complete paired negative evidence.')
        if phase=='final':payload['confirmed_profile_id']='UniformRef_NativeOff_0'
        if len(seen)==1:
            name,args='read_exclusion_evidence',{}
        else:
            name,args=FINAL_TOOL,payload
            if len(seen)==2:args={k:v for k,v in payload.items() if k!='excluded_box_ids'}
            if len(seen)==3:
                feedback=body['messages'][-1]['content']
                assert 'excluded_box_ids explicitly' in feedback and 'supply selected explicitly' not in feedback
        return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=20),choices=[dict(
            finish_reason='tool_calls',message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(
                id='submission_'+str(len(seen)),type='function',function=dict(name=name,arguments=json.dumps(args)))]))]))
    out=tmp_path/phase
    result=exclusion.select(c,out,data,{str(source):sha(source)},RoundBudget(tmp_path/'budget'),phase,
        provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle)))
    assert result['excluded_box_ids']==[0] and len(seen)==3
    receipt=read(out/'decision.receipt.json')
    assert receipt['interface_repairs']==1 and receipt['final_response_mode']=='strict_tool_v1'
    assert read(out/'decision.session/step_0001/TOOL_00.json')['result']['evidence_id']==result['evidence_ids'][0]
