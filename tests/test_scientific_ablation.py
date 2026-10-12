"""Scientific tests of fitted support, interventions and inference bindings."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch

from evomolsteer.ablation.evidence import fit_continuous, evaluate_curve, turnover_accounting
from evomolsteer.ablation.modules import MODULES, assemble
from evomolsteer.ablation.diagnostics import contrast_fixture, geometry_fixture
from evomolsteer.ablation.statistics import compare, compare_strain, pruning_decision, classify_module
from evomolsteer.generation.temporal_teacher_reward import TemporalTeacherReward
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
from evomolsteer.generation.dependency_controller import DependencyExtension
from evomolsteer.generation.endpoint_controller import EndpointCoordinateExtension
from evomolsteer.generation.flowcompat_controller import FlowCompatibilityExtension
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.io import read_json
from evomolsteer.continuous.m2_skill_api import BASE, REF
from evomolsteer.generation.batches import loading_count,indexed_pairs
from evomolsteer.ablation.compiler import compile_program
from evomolsteer.io import write_json, digest
from evomolsteer.ablation import agents
from evomolsteer.ablation import protocol
import gzip
import json
import jsonschema


@pytest.fixture
def compiler_files(tmp_path):
    repo=tmp_path/'repo';study=repo/'study'
    p=read_json(REPO/BASE)
    (repo/BASE).parent.mkdir(parents=True);write_json(repo/BASE,p)
    (repo/REF).parent.mkdir(parents=True,exist_ok=True);(repo/REF).write_bytes((REPO/REF).read_bytes())
    study.mkdir()
    write_json(study/'activation.json',{'protocol':{'lanes':{
        'G':{'allowed':{'teacher_neighbors':[2,4,8],'pointcloud_delta_A':[.5,1.,2.],
            'region_index':'evidence-bound','temporal_region_index':'evidence-bound','temporal_teacher_blend':[.25,.5,1.]}},
        'U':{'allowed':{'native_rms_ratio':[.2,.33,.5],'time_ramp_power':[0.,1.,2.]}}}}})
    write_json(study/'evidence/index.json',{'ids':['regional_effects:0'],'regions':{'regional_effects:0':0}})
    write_json(study/'evidence/packet.json',{'tables':{'survival_effects':{'columns':['evidence_id','feature'],
        'rows':[['survival:0','landmark_00_soft_mass']]}}})
    _,reference=temporal_inputs()
    (study/'evidence/temporal_reference.json.gz').write_bytes(gzip.compress(json.dumps(reference).encode(),mtime=0))
    write_json(study/'evidence/regional_salience.json',{'times':reference['times'],'window':reference['window'],
        'regions':{'0':np.linspace(.1,2.,len(reference['times'])).tolist()}})
    return repo,study,p


def edit(parameter,value,ids=None):
    return {'decision':'edit','edit':{'parameter':parameter,'value':value},'evidence_ids':ids or []}


def test_compiler_preserves_source_hash_and_separates_shape_from_dose(compiler_files):
    repo,study,p=compiler_files
    path,ref,sha,r=compile_program(repo,study,edit('temporal_teacher_blend',.5),p)
    compiled=read_json(path)
    assert compiled['reference_sha256']==digest(ref)==r['reference_sha256']
    assert compiled['native_rms_ratio']==p['native_rms_ratio'] and compiled['mixture_temperature']==.5
    assert compiled['reward_view']=='endpoint_temporal_pointcloud'
    compiled['_reference_relative_path']=ref.relative_to(repo).as_posix()
    changed,ref2,_,_=compile_program(repo,study,edit('native_rms_ratio',.5),compiled,lane='U')
    assert ref2==ref and read_json(changed)['temporal_teacher_blend']==.5
    with pytest.raises(ValueError,match='Illegal'):compile_program(repo,study,edit('mixture_temperature',.25),p)
    with pytest.raises(ValueError,match='Illegal'):compile_program(repo,study,edit('teacher_neighbors',2),p,lane='U')


def test_compiler_refuses_unbound_region_and_synthetic_probe(compiler_files):
    repo,study,p=compiler_files
    with pytest.raises(ValueError,match='bound'):compile_program(repo,study,edit('region_index',0),p)
    path,ref,_,_=compile_program(repo,study,edit('temporal_region_index',0,['regional_effects:0']),p)
    reward=make_coordinate_reward(read_json(path),load_reference(ref))
    assert reward.node_salience[0,0]==.1 and reward.node_salience[-1,0]==2.
    path,_,_,_=compile_program(repo,study,edit('region_index',0,['survival:0']),p)
    assert read_json(path)['coordinate_region_weights'][0]==1.
    with pytest.raises(ValueError,match='quarantined'):
        compile_program(repo,study,edit('teacher_neighbors',2),p,folder=study/'diagnostics')


def test_compiler_missing_reference_binding_does_not_generate(compiler_files):
    repo,study,p=compiler_files;p['reference_sha256']='0'*64
    with pytest.raises(ValueError,match='Reference'):compile_program(repo,study,{'decision':'retain','edit':None},p)


def test_explicit_default_reuses_immutable_m2_program(compiler_files):
    repo,study,p=compiler_files
    first,_,sha,_=compile_program(repo,study,{'decision':'retain','edit':None},p)
    original=first.read_bytes()
    second,_,same,receipt=compile_program(repo,study,edit('pointcloud_delta_A',1.),p)
    assert first==second and sha==same
    assert second.read_bytes()==original
    assert receipt['edit']=={'parameter':'pointcloud_delta_A','value':1.}

REPO=Path(__file__).resolve().parents[1]


def test_continuous_fit_recovers_linear_effect_without_bins_or_extrapolation():
    t=np.linspace(.13,.43,31);y=2*t- .45
    f=fit_continuous(t,np.stack([y-.01,y,y+.01]))
    assert f['method']=='legendre' and f['degree']==1
    np.testing.assert_allclose(evaluate_curve(f,t),y,atol=1e-12)
    assert f['first_rate_z']==pytest.approx(2)
    with pytest.raises(ValueError,match='extrapolation'):evaluate_curve(f,[.1])


def test_bad_polynomial_uses_observed_interpolation_not_invented_sign():
    t=np.linspace(0,.49,50);y=np.where(np.arange(50)%2,1.,-1.)
    f=fit_continuous(t,np.stack([y,y+.02,y-.02]))
    assert f['method']=='observed_node_linear'
    np.testing.assert_allclose(evaluate_curve(f,t),y)
    assert f['sign_changes']==49


def test_turnover_identity_excludes_initial_placeholder_and_is_not_causal():
    t=np.array([.1,.2,.4]);c=np.array([[0.,2.,5.],[1.,0.,-2.]])
    e=np.array([[np.nan,1.,1.],[np.nan,-.5,-.5]])
    r=turnover_accounting(t,c,e)
    np.testing.assert_allclose(r['population_rate'],r['edge_rate']+r['turnover_residual_rate'])
    np.testing.assert_allclose(r['edge_rate'],[2/.3,-1/.3])


def test_literal_interventions_have_no_outcome_or_environment_overrides():
    ids=[m['id'] for m in MODULES if m['role']=='Analyst']
    assert len(ids)==9 and len(MODULES)==17
    full=assemble(MODULES,'Analyst',ids)
    drop=assemble(MODULES,'Analyst',[x for x in ids if x!='A4.1'])
    alt=assemble(MODULES,'Analyst',ids,'A4.1')
    m=next(m for m in MODULES if m['id']=='A4.1')
    assert m['instruction'] in full and m['instruction'] not in drop
    assert m['alternative_instruction'] in alt
    for m in MODULES:
        for forbidden in ['CK2','0.5','vpn','github','affinity_head_gradient']:
            assert forbidden not in m['instruction']
    with pytest.raises(ValueError):assemble(MODULES,'Designer',['A4.1'])


def temporal_inputs(blend=.5):
    p=read_json(REPO/BASE);ref=load_reference(REPO/REF)
    p.update(reward_view='endpoint_temporal_pointcloud',temporal_teacher_blend=blend)
    for i,f in enumerate(ref['frames']):
        f['trajectory_teacher_endpoint_A']=(np.asarray(f['teacher_endpoint_A'])+ .5*(1-i/(len(ref['frames'])-1))).tolist()
        f['trajectory_teacher_batches']=f['teacher_batches']
    return p,ref


def test_zero_temporal_blend_is_exact_M2_and_window_is_metadata():
    p,ref=temporal_inputs(0.);r=make_coordinate_reward(p,ref)
    base=copy.deepcopy(p);base['reward_view']='endpoint_pointcloud'
    expected=EndpointGeometryReward(base,ref)
    x=torch.tensor(ref['frames'][5]['teacher_endpoint_A'][:1],dtype=torch.float64)+.1
    m=torch.ones(x.shape[:2],dtype=torch.bool);a=torch.zeros_like(m,dtype=torch.long)
    got,_=r(x,a,m,.05,x);old,_=expected(x,a,m,.05,x)
    assert torch.equal(got,old)
    r.window=[.2,.4]
    assert r.active(.2,.21) and not r.active(.19,.2) and not r.active(.4,.41)


def test_temporal_reward_conditional_gradient_and_late_static_coincidence():
    p,ref=temporal_inputs();r=TemporalTeacherReward(p,ref)
    x=torch.tensor(ref['frames'][5]['teacher_endpoint_A'][:1],dtype=torch.float64)+.2
    x.requires_grad_(True);anchor=x.detach().clone()
    m=torch.ones(x.shape[:2],dtype=torch.bool);a=torch.zeros_like(m,dtype=torch.long)
    value,_=r(x,a,m,.05,anchor);g,=torch.autograd.grad(value.sum(),x)
    direction=g/g.norm();eps=1e-5
    plus,_=r(x.detach()+eps*direction,a,m,.05,anchor)
    minus,_=r(x.detach()-eps*direction,a,m,.05,anchor)
    torch.testing.assert_close((plus-minus).sum()/(2*eps),(g*direction).sum(),rtol=1e-5,atol=1e-7)
    late,_=r(x,a,m,.49,anchor);static,_=r.static(x,a,m,.49,anchor)
    torch.testing.assert_close(late,static,rtol=1e-12,atol=1e-12)
    assert not torch.equal(g,torch.autograd.grad(r.static(x,a,m,.05,anchor)[0].sum(),x)[0])


@pytest.mark.parametrize('view',['endpoint_temporal_pointcloud','endpoint_regional_pointcloud'])
def test_dependency_adapter_does_not_replace_registered_reward(monkeypatch,tmp_path,view):
    sentinel=object();model=SimpleNamespace()
    def configure(self,model,opt,out):self.reward=sentinel
    monkeypatch.setattr(EndpointCoordinateExtension,'configure',configure)
    def forbidden(*args):raise AssertionError('Factory overwritten by frozen parent')
    monkeypatch.setattr(FlowCompatibilityExtension,'configure',forbidden)
    monkeypatch.setattr('evomolsteer.generation.dependency_controller.DecoderSensitivity',lambda x:None)
    ext=DependencyExtension();ext.program={'reward_view':view}
    ext.configure(model,None,tmp_path)
    assert ext.reward is sentinel


def scores(delta=0.):
    return pd.DataFrame([{'batch':b,'slot':i,'pic50_on_rescore':7.+ .1*b+ i*.01+delta} for b in range(4) for i in range(5)])


def test_shared_rewards_do_not_create_false_generation_precision():
    f=scores();r=compare([f]*6,[f]*6,replicates=1000,family=108,paired_roles=True)
    assert r['independent_generation_batches']==4 and r['simultaneous_CI']==[0.,0.]
    assert pruning_decision(r,r,{'passed':True})['adopt']
    assert classify_module(r,r,functional_valid=True)=='PERFORMANCE_EQUIVALENT_IN_THIS_CONTEXT'


def test_missing_scores_and_attempt_ids_never_become_success():
    f=scores();bad=f.copy();bad.loc[0,'pic50_on_rescore']=np.nan
    with pytest.raises(ValueError,match='coverage'):compare([bad],[f],replicates=1000)
    with pytest.raises(ValueError,match='attempt'):compare([f,f.iloc[1:]],[f],replicates=1000)


def test_inconclusive_interval_is_not_a_null_module_and_secondary_gate_matters():
    r={'simultaneous_CI':[-.1,.1]}
    assert not pruning_decision(r,r,{'passed':True})['adopt']
    assert classify_module(r,r,functional_valid=True)=='INCONCLUSIVE'
    good={'simultaneous_CI':[0.,0.]}
    assert not pruning_decision(good,good,{'passed':False})['adopt']


def test_synthetic_sign_missing_and_unit_probes_are_quarantined():
    p={'tables':{'regional_effects':{'columns':['effect_z','CI_low','CI_high','positive_batch_fraction'],
        'rows':[[.5,.2,.8,.75]]}},'teacher_modes':{'count':2,'source':'observed','assignment_rms_A_quantiles':[1.,2.,3.]},'clocks':{}}
    q=contrast_fixture(p,'sign_changed')
    assert q['diagnostic_only'] and q['tables']['regional_effects']['rows'][0]==[-.5,-.8,-.2,.25]
    assert contrast_fixture(p,'missing')['tables']=={}
    np.testing.assert_allclose(geometry_fixture(p,'nm')['teacher_modes']['assignment_rms_quantiles'],[.1,.2,.3])
    assert p['tables']['regional_effects']['rows'][0][0]==.5


def test_role_ablation_changes_only_instructions_not_evidence_or_science_schema(tmp_path,monkeypatch):
    study=tmp_path/'study';study.mkdir()
    packet={'tables':{},'clocks':{'observed_score_times':[.1,.2],'learning_window':[.1,.3]},'teacher_modes':{'landmarks_A':[[0.,0.,0.]]}}
    write_json(study/'evidence/packet.json',packet)
    write_json(study/'evidence/continuous_functions.json',{'times':[.1,.2],'window':[.1,.3],
        'functions':{'feature':{'method':'observed_node_linear','observed_support':[.1,.2],'values_z':[1.,-1.]}}})
    write_json(study/'evidence/index.json',{'ids':[]})
    write_json(study/'evidence/observable_definitions.json',{'occupancy':'radial shell, not within-radius count'})
    write_json(study/'module_registry.json',{'modules':MODULES})
    monkeypatch.setattr(agents,'validate_activation',lambda *a:{'protocol':{
        'lanes':{'G':{'allowed':{'teacher_neighbors':[2,4,8]}}},
        'temperature_policy':{'parameters':{'mixture_temperature':.5,'teacher_endpoint_temperature_A2':4.}}},
        'packet_sha256':digest(study/'evidence/packet.json')})
    observed=[]
    def caller(messages,schema,path,**kwargs):
        observed.append((messages,schema))
        if json.loads(messages[-1]['content'])['role']=='Designer':
            return {'role':'Designer','decision':'retain','edit':None,'evidence_ids':[],'rationale':'Retain'}
        return {'role':'Analyst','findings':[]}
    agents.call_role(REPO,study,'full',0,'Analyst',['A1.1','A4.1'],caller=caller)
    agents.call_role(REPO,study,'drop',0,'Analyst',['A1.1'],caller=caller)
    assert observed[0][0][1]==observed[1][0][1]
    assert observed[0][1]==observed[1][1]
    assert '## A4.1' in observed[0][0][0]['content'] and '## A4.1' not in observed[1][0][0]['content']
    assert observed[1][1]['required']==['role','findings']
    payload=json.loads(observed[1][0][1]['content'])
    assert payload['continuous_functions']['functions']['feature']['values_z']==[1.,-1.]
    assert payload['observable_definitions']['occupancy']=='radial shell, not within-radius count'
    fresh={'findings':[],'role':'Analyst'};cached={'role':'Analyst','findings':[]}
    agents.call_role(REPO,study,'handoff',0,'Designer',[],analyst=fresh,caller=caller)
    agents.call_role(REPO,study,'handoff',0,'Designer',[],analyst=cached,caller=caller)
    assert observed[2][0]==observed[3][0], 'Fresh/cached handoff must have identical request bytes'


def test_design_schema_repairs_illegal_grid_without_forcing_an_edit():
    schema=agents.design_schema({'teacher_neighbors':[2,4,8],'region_index':'bound region'},20)
    response={'role':'Designer','decision':'retain','edit':None,'evidence_ids':[],'rationale':'Unsupported edit'}
    jsonschema.validate(response,schema)
    response.update(decision='edit',edit={'parameter':'teacher_neighbors','value':3})
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(response,schema)
    response['edit']={'parameter':'region_index','value':20}
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(response,schema)
    response['edit']['value']=19
    jsonschema.validate(response,schema)


def test_indexed_input_policy_avoids_unused_prefix_and_preserves_master_seed():
    assert loading_count(50,[2100,2101],'legacy_stream')==105100
    assert loading_count(50,[2100,2101],'indexed_batch')==50
    calls=[]
    def loader(args,data,interpolant):
        calls.append((args.seed,data));return [(args.seed,data)]
    args=SimpleNamespace(seed=42)
    pairs=list(indexed_pairs(args,'target','off',None,loader,[2100,2101],42))
    assert args.seed==42 and len(calls)==4
    assert pairs[0][1][0][0]==pairs[0][1][1][0]==42+100003*2100
    assert pairs[1][1][0][0]==42+100003*2101


def test_prior_amendment_cannot_adopt_legacy_results_or_change_on_resume(tmp_path):
    study=tmp_path/'study';study.mkdir()
    plan=tmp_path/protocol.PLAN;plan.mkdir(parents=True)
    write_json(plan/'protocol.json',{'frozen':True})
    amendment={'prior_seed_mode':'indexed_batch','master_seed':42,'batch_seed_multiplier':100003,
        'protocol_sha256':digest(plan/'protocol.json'),'fresh_matched_controls_required':True}
    write_json(study/'prior_policy_amendment.json',amendment)
    config={'prior_seed_mode':'indexed_batch'}
    bound=protocol.validate_prior_policy(tmp_path,study,config,{})
    assert protocol.validate_prior_policy(tmp_path,study,config,bound)==bound
    with pytest.raises(ValueError,match='earlier results'):
        protocol.validate_prior_policy(tmp_path,study,config,{'program_results':{'legacy':{}}})
    with pytest.raises(ValueError,match='mix prior'):
        protocol.validate_prior_policy(tmp_path,study,config,{'prior_seed_mode':'legacy_stream'})
    amendment['reason']='changed';write_json(study/'prior_policy_amendment.json',amendment)
    with pytest.raises(ValueError,match='changed after activation'):
        protocol.validate_prior_policy(tmp_path,study,config,bound)


def test_role_incumbent_exposes_executing_curvature_and_no_inert_motif_force():
    original=read_json(REPO/BASE);view=agents.operative_incumbent(original)
    assert view['pointcloud_delta_A']==1.
    assert not set(view)&{'robust_delta','geometry_block_weights','motif_components'}
    assert original['robust_delta']==2 and 'pointcloud_delta_A' not in original


def test_secondary_necessity_requires_affinity_equivalence_and_converged_energy():
    full=scores().assign(energy_status='converged',mmff_relief_per_heavy=1.)
    drop=full.assign(mmff_relief_per_heavy=1.2)
    secondary=compare_strain([full]*6,[drop]*6,replicates=1000)
    assert secondary['endpoints']['median']['simultaneous_CI']==pytest.approx([.2,.2])
    eq={'simultaneous_CI':[0.,0.]}
    assert classify_module(eq,eq,functional_valid=True,secondary=[secondary,secondary])=='STRAIN_NECESSARY_IN_THIS_CONTEXT'
    assert classify_module({'simultaneous_CI':[-.1,.1]},eq,functional_valid=True,secondary=[secondary,secondary])=='INCONCLUSIVE'
    failed=drop.assign(energy_status='failed')
    assert compare_strain([full],[failed],replicates=1000)['status']=='UNAVAILABLE'


def test_landmark_occupancy_is_a_shell_and_not_a_within_radius_contact_count():
    from evomolsteer.continuous.affinity_geometry import names,numpy_geometry
    points=np.zeros((1,3));fields=names(points);i=fields.index('landmark_00_occupancy3')
    x=np.array([[[0.,0.,0.],[0.,0.,0.]],[[3.,0.,0.],[3.,0.,0.]]])
    geometry=numpy_geometry(x,points,np.zeros(3))
    assert geometry[0,i]==pytest.approx(np.exp(-4.5))
    assert geometry[1,i]==pytest.approx(1.)
    assert geometry[1,fields.index('landmark_00_softmin')]==pytest.approx(3.)


def test_evidence_schema_prevents_invented_table_ids_without_forcing_findings():
    schema=copy.deepcopy(agents.ANALYST_SCHEMA)
    agents.bind_evidence_schema(schema,{'continuous:199','observable_definitions'})
    jsonschema.validate({'role':'Analyst','findings':[]},schema)
    response={'role':'Analyst','findings':[{'claim':'Supplied curve',
        'evidence_ids':['continuous:597'],'priority':1}]}
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(response,schema)
    response['findings'][0]['evidence_ids']=['continuous:199']
    jsonschema.validate(response,schema)
