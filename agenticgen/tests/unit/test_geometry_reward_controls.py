"""Template loss, VJP, binary tool selection and candidate cache invariants."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
import torch
from test_uniguide_geometry import features,domain
from guidance_tool_fixture import build
from agenticgen.uniguide import geometry_condition as condition
from agenticgen.uniguide.geometry_reward import *
from agenticgen.uniguide.geometry_sampling import GeometryControl,scale_binding,candidate_identity
from agenticgen.uniguide.geometry_batch import CandidateControl,compile_units
from agenticgen.uniguide.geometry_protocol import NORMALIZED,EFFECTIVE,resolve_action,seen_configuration,choose
from agenticgen.uniguide.analysis_tools import EvidenceTools
from agenticgen.uniguide.geometry_council import advise
from agenticgen.uniguide.tool_runtime import configure,assets
from agenticgen.providers.deepseek_tools import call_with_tools
from agenticgen.uniguide.io import read,write,sha,value_sha


def spec(interact=False,mode="template_mean"):
    f=features()
    if interact:
        f.update(coordinate_frame="generation_context_angstrom",target_atoms=[dict(id="receptor/A/9/O",coordinates_angstrom=[0.,3.,0.])],
            interact_atoms=[dict(atom_id="a",target_atom_ids=["receptor/A/9/O"])])
    return condition.prepare(f,reward=dict(mode=mode))


def tensors(s):
    x=torch.tensor(s["reference_coordinates_angstrom"],dtype=torch.float64)[None]
    return x,torch.ones(x.shape[:2],dtype=torch.bool)


def test_template_relative_default_components_and_all_outer_pairs():
    s=spec();x,valid=tensors(s)
    assert s[WEIGHTS]==dict(zip(KEYS,(1,0,0)))
    assert len(s["pairs"])==5 and [p["atom_ids"] for p in s["pairs"] if p["category"]=="controlled_neighborhood"]==[["a","c"],["a","d"],["b","c"],["b","d"]]
    result,details=evaluate(x,s,valid)
    assert all(torch.equal(details[k],torch.zeros(1,dtype=x.dtype)) for k in LOSSES)
    changed=x.clone();changed[:,2:]+=7;changed.requires_grad_()
    result,details=evaluate(changed,s,valid)
    assert result[0].item()==0 and details["condition_outer_loss"].item()>0
    zero=torch.autograd.grad(result[0].sum(),changed)[0]
    active=dict(zip(KEYS,(1,1,0)))
    value=evaluate(changed,s,valid,active)[0][0]
    grad=torch.autograd.grad(value.sum(),changed)[0]
    assert not zero.any() and grad[:,2:].abs().sum()>0
    assert torch.autograd.gradcheck(lambda p:evaluate(p,s,valid,active)[0][0],(changed,))


def test_rigid_motion_and_template_not_zero_distance():
    s=spec();x,valid=tensors(s)
    rotation=x.new_tensor([[0,1,0],[-1,0,0],[0,0,1]])
    assert evaluate(x@rotation+17,s,valid)[0][0].item()<1e-25
    compressed=x*.5
    assert evaluate(compressed,s,valid)[0][0].item()>0
    assert evaluate(x,s,valid)[0][0].item()==0


def test_interaction_target_pair_default_gradient_and_zero_empty():
    s=spec(True);x,valid=tensors(s)
    assert s[WEIGHTS]==dict(zip(KEYS,(1,0,1)))
    assert evaluate(x,s,valid)[1]["interact_loss"].item()==0
    translated=(x+x.new_tensor([0.,1.,0.])).requires_grad_()
    value,details=evaluate(translated,s,valid)
    assert details["condition_intra_loss"].item()==0 and details["interact_loss"].item()>0
    g=torch.autograd.grad(value[0].sum(),translated)[0]
    assert g[0,0,1]>0 and torch.equal(g[0,1:],torch.zeros_like(g[0,1:]))
    assert torch.autograd.gradcheck(lambda p:evaluate(p,s,valid)[0][0],(translated,))
    assert condition.metrics(translated,s,valid)[INTERACTION_DISTANCE].item()==1.
    assert evaluate(translated,spec(),valid)[1]["interact_loss"].item()==0
    f=features();f['interact_atoms']=[dict(atom_id='a',target_atom_ids=['unknown'])]
    with pytest.raises(ValueError,match="frame"):condition.prepare(f,reward={})
    with pytest.raises(ValueError,match="interact_atoms"):weights(spec(),dict(zip(KEYS,(1,0,1))))


@pytest.mark.parametrize("value",[-1,2,.5,True,"1"])
def test_nonbinary_controls_rejected(value):
    with pytest.raises(ValueError):weights(spec(),dict(zip(KEYS,(1,value,0))))


def test_step_denominator_detached_bounded_and_masked():
    s=spec(mode="bounded_step_mean_detached");x,valid=tensors(s)
    point=(x*1.7).requires_grad_();current=(x*2).requires_grad_()
    value,diag=evaluate(point,s,valid,current=current)
    denominator_value=diag['distance_normalization_denominator_angstrom']
    assert not denominator_value.requires_grad
    assert torch.autograd.grad(value[0].sum(),current,allow_unused=True)[0] is None
    expected=torch.pdist(x[0]).mean()*2
    assert denominator_value.item()==pytest.approx(expected.item())
    base=denominator(x,valid,s,s['distance_normalization'])
    expanded=denominator(x*1e6,valid,s,s['distance_normalization'])
    collapsed=denominator(x*0,valid,s,s['distance_normalization'])
    assert torch.allclose(expanded,base*4) and torch.allclose(collapsed,base,rtol=1e-14,atol=1e-14)
    padded=torch.cat((x,x.new_full((1,1,3),1e8)),1)
    assert torch.equal(denominator(padded,torch.tensor([[1,1,1,1,0]]),s,s['distance_normalization']),base)
    fixed=evaluate(point,s,valid,current=x)[0][0]
    assert value[0].item()==pytest.approx(fixed.item()/4)
    with pytest.raises(ValueError,match="Current-step"):evaluate(point,s,valid)


def test_mixed_candidate_and_single_vjp_identical_no_cross_candidate_normalization():
    import threading
    s=spec(True,mode='bounded_step_mean_detached');base,mask=tensors(s)
    x=(base.repeat(20,1,1)*1.4).requires_grad_();valid=mask.repeat(20,1)
    controls=[weights(s),dict(zip(KEYS,(1,1,1)))]
    control=object.__new__(CandidateControl)
    control.geometry_condition=s;control.units=[dict(scale={WEIGHTS:w}) for w in controls]
    control.before=base.repeat(20,1,1);control.before[10:]*=100
    control.com=torch.zeros(20,3,dtype=x.dtype);control.model=SimpleNamespace(coord_scale=1.)
    control.initial=dict(population=20);control.failed_units={};control.failure_lock=threading.RLock()
    combined=control.condition_loss_for_rows(x,valid,list(range(20)))
    gradients=torch.autograd.grad(combined[0].sum(),x)[0]
    for i,w in enumerate(controls):
        indices=list(range(i*10,(i+1)*10));part=x[indices].detach().requires_grad_()
        single=object.__new__(GeometryControl);single.geometry_condition=s;single.scale_binding={WEIGHTS:w}
        single.before=control.before[indices];single.com=control.com[indices];single.model=control.model
        expected=single.condition_loss_for_rows(part,valid[indices],list(range(10)))
        assert torch.equal(combined[0][indices],expected[0])
        assert torch.equal(gradients[indices],torch.autograd.grad(expected[0].sum(),part)[0])
        assert expected[1]['interact_loss'].shape==(10,)


def control_fixture(root):
    h,d=build(root);s=spec();new_hash=s['condition_sha256']
    for row in h:
        p=Path(root)/row['id'];manifest=read(p/'manifest.json')
        manifest['identity']['compatibility']['condition_sha256']=new_hash
        row.update(condition_sha256=new_hash,compatibility_sha256=value_sha(manifest['identity']['compatibility']))
        row[WEIGHTS]=weights(s);row[INTERACTION_DISTANCE]=0.
        manifest['identity'][WEIGHTS]=row[WEIGHTS]
        rows=read(p/'records.json')
        for r in rows:r[INTERACTION_DISTANCE]=0.
        write(p/'manifest.json',manifest);write(p/'aggregate.json',row);write(p/'records.json',rows)
        write(p/'COMPLETE.json',dict(files={n:sha(p/n) for n in ('records.json','aggregate.json','manifest.json','steps.json')}))
    d.update(condition_sha256=new_hash,sampling_protocol_sha256=h[0]['compatibility_sha256'])
    d['calibration_key']=value_sha([d['teacher_id'],new_hash,d['sampling_protocol_sha256']])
    d['domain_sha256']=value_sha({k:v for k,v in d.items() if k!='domain_sha256'})
    write(Path(root)/'conditions'/f"{d['teacher_id']}.json",s)
    return h,d,s


def test_control_receipt_to_resolved_action_and_gradient(tmp_path):
    h,d,s=control_fixture(tmp_path);folder=tmp_path/'calculations'
    registry=EvidenceTools(tmp_path,h,d,folder,condition=s)
    first=registry.execute('summarize_population',dict(run_id=h[0]['id']))
    args=dict(anchor_id=h[0]['id'],**dict(zip(KEYS,(1,1,0))),evidence_ids=[first['evidence_id']],reason='Test outer constraint after uncertain primary gain')
    with pytest.raises(ValueError,match='paired'):registry.execute('set_guidance_loss_weights',args)
    pair=registry.execute('compare_paired_runs',dict(reference_id=h[0]['id'],candidate_id=h[1]['id']))
    args['evidence_ids']=[pair['evidence_id']]
    receipt=registry.execute('set_guidance_loss_weights',args)
    action=dict(anchor_id=h[0]['id'],branch='redistribute',updates=[],loss_control_id=receipt['evidence_id'])
    bound=registry.resolve_loss_control(action)
    resolved=resolve_action(bound,{r['id']:r for r in h},d,[seen_configuration(r[NORMALIZED],r[WEIGHTS]) for r in h])
    assert resolved[NORMALIZED]==h[0][NORMALIZED] and resolved[WEIGHTS]['condition_outer_weight']==1
    fresh=EvidenceTools(tmp_path,h,d,folder,condition=s)
    assert fresh.resolve_loss_control(action,allow_saved=True)==bound
    with pytest.raises(ValueError):registry.resolve_loss_control(dict(action,anchor_id=h[1]['id']))
    old=scale_binding(resolved[NORMALIZED],2.,condition=s)
    new=scale_binding(resolved[NORMALIZED],2.,condition=s,loss_weights=resolved[WEIGHTS])
    assert value_sha(candidate_identity(old))!=value_sha(candidate_identity(new))
    x,valid=tensors(s);x[:,2:]+=4;x.requires_grad_()
    before=condition.loss(x,s,valid,old[WEIGHTS])[0];after=condition.loss(x,s,valid,new[WEIGHTS])[0]
    assert before.item()==0 and after.item()>0
    assert torch.autograd.grad(after.sum(),x)[0][:,2:].abs().sum()>0
    assert not (tmp_path/'attempts.json').exists()


def test_scheduler_identity_and_primary_ranking(tmp_path,monkeypatch):
    import agenticgen.uniguide.geometry_batch as batch
    s=spec();binding=dict(teacher_id='teacher',condition_sha256=s['condition_sha256'])
    monkeypatch.setattr(batch,'compatibility',lambda *a:binding)
    jobs=[dict(schedule=[1.]*4,folder=tmp_path/str(i),stage='calibration',count=10,loss_weights=dict(zip(KEYS,(1,i,0)))) for i in (0,1)]
    units=compile_units(dict(root=str(tmp_path)),dict(target='teacher'),s,jobs)
    assert len(units)==2 and units[0]['fingerprint']!=units[1]['fingerprint']
    rows=[dict(id='a',complete=True,**{WEIGHTS:weights(s),condition.DISTANCE:4.,condition.CONTROL_DISTANCE:.1}),
          dict(id='b',complete=True,**{WEIGHTS:weights(s),condition.DISTANCE:.2,condition.CONTROL_DISTANCE:.15})]
    assert choose(rows)['id']=='a'


def test_registered_tool_multi_message_role_selection_and_cache(tmp_path,monkeypatch):
    h,d,s=control_fixture(tmp_path);config=configure(dict(root=str(tmp_path),guidance_reward=dict(mode='template_mean'),llm=dict(thinking=True,attempts=1)))
    skills=tmp_path/'skills';skills.mkdir()
    for n,p in assets(config).items():(skills/n).write_bytes(p.read_bytes())
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only');calls=[]
    def handle(request):
        body=json.loads(request.content);calls.append(body)
        context=json.loads(body['messages'][1]['content']);tool_results=[json.loads(m['content']) for m in body['messages'] if m['role']=='tool']
        role=context.get('branch')
        if role=='increase' and len(tool_results)<2:
            if not tool_results:name,args='compare_paired_runs',dict(reference_id=h[0]['id'],candidate_id=h[1]['id'])
            else:name,args='set_guidance_loss_weights',dict(anchor_id=h[0]['id'],**dict(zip(KEYS,(1,1,0))),evidence_ids=[tool_results[-1]['evidence_id']],reason='Paired evidence suggests uncertain internal progress; test outer geometry')
            message=dict(role='assistant',content=None,reasoning_content='opaque',tool_calls=[dict(id='call'+str(len(calls)),type='function',function=dict(name=name,arguments=json.dumps(args)))])
            finish='tool_calls'
        else:
            ev=context['existing_statistics_receipts'][0]['evidence_id']
            if role:
                candidates=copy.deepcopy(context['feasible_examples'])
                if role=='increase':candidates[0]=dict(anchor_id=h[0]['id'],branch=role,updates=[],loss_control_id=tool_results[-1]['evidence_id'])
                result=dict(analysis=ev,mechanism_uncertainty='Synthetic contract validation',proposals=[dict(p,id=str(i),anchor_reason='measured anchor',evidence=ev,prediction='outer distance improves',falsifier='condition geometry deteriorates') for i,p in enumerate(candidates)])
            else:result=dict(analysis=ev,choice_evidence=ev,selected={b:b+':0' for b in ('increase','decrease','redistribute')})
            message=dict(role='assistant',content=json.dumps(result));finish='stop'
        return httpx.Response(200,json=dict(model='deepseek-flash',usage=dict(completion_tokens=12),choices=[dict(message=message,finish_reason=finish)]))
    provider=lambda *a,**kw:call_with_tools(*a,**kw,transport=httpx.MockTransport(handle))
    proposals=advise(config,h,d,tmp_path/'round',skills,provider=provider)
    assert len(calls)==6 and proposals[0]['updates']==[] and proposals[0]['loss_control_id'].startswith('ev_')
    before=len(calls);assert advise(config,h,d,tmp_path/'round',skills,provider=provider)==proposals
    assert len(calls)==before
    assert 'set_guidance_loss_weights' in {t['function']['name'] for t in calls[0]['tools']}


def test_release_bound_complete_and_parent_unchanged():
    root=Path(__file__).resolve().parents[2]/'src/agenticgen/resources/skill_releases/guidance'
    release=read(root/'2.2.0/RELEASE.json')
    for p,h in release['old_files_sha256'].items():assert sha(root/'2.1.0'/p)==h
    for p,h in release['files_sha256'].items():assert sha(root/'2.2.0'/p)==h
    bundle=read(root/'2.2.0/runtime/internal_distance/BUNDLE.json')
    for p,h in bundle['hashes'].items():assert sha(root/'2.2.0/runtime/internal_distance'/p)==h


def test_verified_trace_component_diagnostics_and_tamper_rejection(tmp_path):
    import h5py
    import numpy as np
    h,d,s=control_fixture(tmp_path);group=tmp_path/h[0]['id']
    with h5py.File(group/'trace.h5','w') as f:
        for name in (*LOSSES,'distance_normalization_denominator_angstrom',*KEYS):
            f[name]=np.repeat(np.arange(100,dtype=float)[:,None],50,axis=1)
    complete=read(group/'COMPLETE.json');complete['files']['trace.h5']=sha(group/'trace.h5');write(group/'COMPLETE.json',complete)
    registry=EvidenceTools(tmp_path,h,d,tmp_path/'memory',condition=s)
    receipt=registry.execute('inspect_interval_evidence',dict(run_id=h[0]['id']))
    diagnostics=receipt['reward_diagnostics']
    assert diagnostics['status']=='complete'
    assert diagnostics['windows'][0]['values']['condition_intra_loss']==dict(mean=17.5,minimum=10.,maximum=25.)
    with h5py.File(group/'trace.h5','a') as f:f['condition_intra_loss'][10,0]=999.
    with pytest.raises(ValueError,match='trace changed'):EvidenceTools(tmp_path,h,d,tmp_path/'other',condition=s)


def test_runtime_executes_selected_loss_control_and_reports_it(tmp_path,monkeypatch):
    from agenticgen.uniguide import geometry_runtime as runtime,geometry_calibration as calibration
    from agenticgen.uniguide.geometry_protocol import POLICY,legal_options,BRANCHES
    h,d,s=control_fixture(tmp_path);target=d['teacher_id'];received=[]
    config=dict(protocol=POLICY,root=str(tmp_path),generation_context_root=str(tmp_path.parent/'source'),rounds=1,
        output_policy='raw999',maximum_single_atom_guidance_increment_angstrom=3.)
    write(tmp_path/'calibration/domains.json',{target:d})
    monkeypatch.setattr(runtime,'require_acceptance',lambda c:None)
    monkeypatch.setattr(runtime,'source_binding',lambda:{'fixture':'control-integration'})
    monkeypatch.setattr(runtime,'calibrated',lambda c:[dict(teacher=dict(target=target),condition=s)])
    monkeypatch.setattr(runtime,'history',lambda *a:(h,[]))
    monkeypatch.setattr(runtime,'compatibility',lambda *a:{})
    monkeypatch.setattr(calibration,'verify_frozen',lambda *a:None)
    def advisor(config,history,domain,folder,skills,failures,final=False):
        if final:return []
        registry=EvidenceTools(tmp_path,h,d,folder/'task_memory/calculations',condition=s)
        pair=registry.execute('compare_paired_runs',dict(reference_id=h[0]['id'],candidate_id=h[1]['id']))
        receipt=registry.execute('set_guidance_loss_weights',dict(anchor_id=h[0]['id'],**dict(zip(KEYS,(1,1,0))),
            evidence_ids=[pair['evidence_id']],reason='Synthetic plateau hypothesis'))
        result=[legal_options(history,domain,b)[0] for b in BRANCHES]
        result[0]=dict(branch='increase',anchor_id=h[0]['id'],updates=[],loss_control_id=receipt['evidence_id'])
        return result
    def executor(model,config,teacher,condition,schedule,folder,stage,domain,loss_weights):
        received.append(loss_weights)
        return dict(h[0],id=folder.relative_to(tmp_path).as_posix(),**{WEIGHTS:loss_weights})
    result=runtime.run_version(config,'GD01',model_loader=lambda *a:object(),executor=executor,advisor=advisor,retainer=lambda *a:None)
    assert [w['condition_outer_weight'] for w in received]==[1,0,0]
    assert result['current']['groups'][0][WEIGHTS]['condition_outer_weight']==1
    assert INTERACTION_DISTANCE in result['current']['teachers'][target]


def test_exhausted_temporal_examples_do_not_block_registered_component_tools(tmp_path,monkeypatch):
    import agenticgen.uniguide.geometry_council as council
    h,d,s=control_fixture(tmp_path);config=configure(dict(root=str(tmp_path),guidance_reward={},llm={}))
    skills=tmp_path/'skills';skills.mkdir()
    for name,path in assets(config).items():(skills/name).write_bytes(path.read_bytes())
    monkeypatch.setattr(council,'legal_options',lambda *a:[])
    class ToolPhaseReached(Exception):pass
    def inspect(messages,schema,out,**kwargs):
        context=json.loads(messages[1]['content'])
        assert context['temporal_examples_exhausted'] and context['feasible_examples']==[]
        assert 'set_guidance_loss_weights' in context['registered_computation_tools']
        assert schema['properties']['proposals']['items']['properties']['updates']['minItems']==0
        raise ToolPhaseReached()
    with pytest.raises(ToolPhaseReached):council.advise(config,h,d,tmp_path/'call',skills,provider=inspect)


def test_outer_control_rejects_incomplete_primary_paired_measurements(tmp_path):
    h,d,s=control_fixture(tmp_path);group=tmp_path/h[1]['id']
    records=read(group/'records.json');records[0][condition.CONTROL_DISTANCE]=None
    write(group/'records.json',records)
    complete=read(group/'COMPLETE.json');complete['files']['records.json']=sha(group/'records.json');write(group/'COMPLETE.json',complete)
    registry=EvidenceTools(tmp_path,h,d,tmp_path/'memory',condition=s)
    pair=registry.execute('compare_paired_runs',dict(reference_id=h[0]['id'],candidate_id=h[1]['id']))
    with pytest.raises(ValueError,match='paired'):
        registry.execute('set_guidance_loss_weights',dict(anchor_id=h[0]['id'],**dict(zip(KEYS,(1,1,0))),
            evidence_ids=[pair['evidence_id']],reason='Missing measurement cannot establish primary plateau'))
