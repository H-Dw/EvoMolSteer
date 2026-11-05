import copy
from pathlib import Path
import pytest
import torch
from agenticgen.uniguide import iterative_affinity as it, full_graph, condition_evaluation as fitted
from agenticgen.uniguide.io import read


def test_ten_round_configs_and_caps(tmp_path):
    from agenticgen.uniguide.absolute_runtime import validate_config
    from agenticgen.uniguide.linear_calibration import ensure_limits
    folder=Path(__file__).resolve().parents[2]/'configs'
    for name,on in [('uniguide_stage_10_chemical.json',True),('uniguide_stage_10_coordinate_only.json',False)]:
        c=read(folder/'uniguide_absolute_coordinate.json');c.update(read(folder/name));c['root']=str(tmp_path)
        c=validate_config(c);assert c['rounds']==10 and c['chemical_inpainting']['enabled'] is on
        ensure_limits(c,2)
        assert read(tmp_path/'ATTEMPT_LIMITS.json')['per_teacher_stage_caps']==dict(formal=2500,calibration=500)
    c['rounds']=5
    with pytest.raises(ValueError):validate_config(c)


def test_seed_disjointness_and_fixed_iterative_config():
    all_seeds=[s for r in range(6) for s in it.seeds(r)]
    assert len(set(all_seeds))==30 and not set(all_seeds)&{42,100045,200048,300051,400054}
    folder=Path(__file__).resolve().parents[2]/'configs'
    c=read(folder/'uniguide_absolute_coordinate.json');c.update(read(folder/'uniguide_iterative_affinity.json'))
    c['conditions']=c['conditions'][:1];it.validate(c)
    c['rounds']=10
    with pytest.raises(ValueError):it.validate(c)


def test_fixed_scale_iteration_binds_native_early_shape_without_an_agent():
    from types import SimpleNamespace
    from agenticgen.uniguide.box_search import early_shape,enabled
    from agenticgen.uniguide.geometry_sampling import scale_binding,GeometryControl
    from agenticgen.uniguide.geometry_dynamics import policy
    folder=Path(__file__).resolve().parents[2]/'configs'
    cfg=dict(read(folder/'uniguide_absolute_coordinate.json'),**read(folder/'uniguide_iterative_affinity.json'))
    cfg['conditions']=cfg['conditions'][:1];cfg=it.validate(cfg)
    assert 'agent_tools' not in cfg and not enabled(cfg)
    schedule=[50,100,40,60,80,200]
    binding=scale_binding(schedule,window_bounds=cfg['windows'],early_shape=early_shape(cfg))
    control=SimpleNamespace(scale_binding=binding)
    expanded=GeometryControl.expand_schedule(control,schedule)
    assert early_shape(cfg)==policy(cfg)['early_guidance_shape']
    assert expanded[0]==expanded[99]==0
    for scale,(lo,hi) in zip(schedule,cfg['windows']):
        assert expanded[max(1,lo):min(99,hi+1)]==[scale]*len(range(max(1,lo),min(99,hi+1)))


def test_iterative_chemical_policy_equals_experiment_one():
    from agenticgen.uniguide.geometry_chemistry import policy
    folder=Path(__file__).resolve().parents[2]/'configs'
    base=read(folder/'uniguide_absolute_coordinate.json')
    first=dict(base,**read(folder/'uniguide_stage_10_chemical.json'))
    third=dict(base,**read(folder/'uniguide_iterative_affinity.json'))
    assert first['chemical_inpainting']==third['chemical_inpainting']
    assert policy(first)==policy(third)
    third['conditions']=third['conditions'][:1]
    old=copy.deepcopy(third);old['chemical_inpainting']['scope']='full_molecule_graph'
    with pytest.raises(ValueError):it.validate(old)


def test_iterative_inpaints_only_current_interaction_region(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agenticgen.uniguide import geometry_chemistry as chemistry
    from agenticgen.uniguide.io import value_sha
    import torch.nn.functional as F
    folder=Path(__file__).resolve().parents[2]/'configs'
    cfg=dict(read(folder/'uniguide_absolute_coordinate.json'),**read(folder/'uniguide_iterative_affinity.json'))
    cond=dict(atoms=[dict(id=str(i)) for i in range(4)],controlled_atom_ids=['0','2'],condition_sha256='test')
    features=dict(atomics=F.one_hot(torch.tensor([0,2]),3).float(),
        charges=F.one_hot(torch.tensor([0,1]),2).float(),hybridization=F.one_hot(torch.tensor([1,0]),2).float(),
        bonds=F.one_hot(torch.zeros((2,2),dtype=torch.long),5).float())
    observed=[]
    def encode(model,condition):
        observed.append(condition['controlled_atom_ids'])
        binding=dict(controlled_atom_ids=condition['controlled_atom_ids'])
        return features,[],dict(binding,chemical_condition_sha256=value_sha(binding))
    monkeypatch.setattr(chemistry,'condition_features',encode)
    monkeypatch.setattr(full_graph,'assignment',lambda *a:pytest.fail('Full graph assignment must not run'))
    control=SimpleNamespace(sampling_policy=dict(chemical_inpainting=chemistry.policy(cfg)),absolute_config=cfg,
        geometry_condition=cond,model=object(),folder=tmp_path,rows=[{}],
        absolute_state=SimpleNamespace(rows=[dict(slots=[3,1])]),store=lambda *a:None)
    chemistry.ChemicalStateInpainting.configure_chemistry(control)
    assert observed==[['0','2']]
    state=dict(coords=torch.randn((1,4,3)),mask=torch.ones((1,4),dtype=torch.bool),
        atomics=F.one_hot(torch.ones((1,4),dtype=torch.long),3).float(),
        charges=F.one_hot(torch.zeros((1,4),dtype=torch.long),2).float(),
        hybridization=F.one_hot(torch.ones((1,4),dtype=torch.long),2).float(),
        bonds=F.one_hot(torch.ones((1,4,4),dtype=torch.long),5).float())
    before=torch.get_rng_state().clone()
    for step in (0,99):assert chemistry.ChemicalStateInpainting.apply_chemical_inpainting(control,state,step) is state
    for step in (1,98):
        out=chemistry.ChemicalStateInpainting.apply_chemical_inpainting(control,state,step)
        assert out['coords'] is state['coords'] and out['mask'] is state['mask']
        for key in ('atomics','charges','hybridization'):
            assert torch.equal(out[key][:,[0,2]],state[key][:,[0,2]])
        for i in range(4):
            for j in range(4):
                if (i,j) not in ((3,1),(1,3)):assert torch.equal(out['bonds'][:,i,j],state['bonds'][:,i,j])
        assert out['atomics'][0,[3,1]].argmax(-1).tolist()==[0,2]
    assert torch.equal(before,torch.get_rng_state())


def test_full_graph_anchors_core_and_covers_all_slots():
    cond=dict(atoms=[dict(id=str(i)) for i in range(4)],controlled_atom_ids=['0','2'],
        reference_coordinates_angstrom=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
        protein_alignment=dict(rotation=torch.eye(3).tolist(),translation_angstrom=[0,0,0]))
    xyz=torch.tensor([[[0.,0,1],[0,1,0],[1,0,0],[0,0,0]]])
    out=full_graph.assignment(xyz,torch.ones((1,4),dtype=torch.bool),cond,torch.tensor([[3,1]]))
    assert out.tolist()==[[3,2,1,0]]
    with pytest.raises(ValueError):full_graph.assignment(xyz,torch.tensor([[1,1,1,0]],dtype=torch.bool),cond,torch.tensor([[3,1]]))


def test_full_graph_projection_covers_nonbonds_without_moving_coords():
    from types import SimpleNamespace
    from agenticgen.uniguide.geometry_chemistry import ChemicalStateInpainting
    import torch.nn.functional as F
    cond=dict(atoms=[dict(id=str(i)) for i in range(4)],controlled_atom_ids=['0','2'],
        reference_coordinates_angstrom=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],
        protein_alignment=dict(rotation=torch.eye(3).tolist(),translation_angstrom=[0,0,0]))
    features=dict(atomics=F.one_hot(torch.tensor([0,1,2,0]),3).float(),bonds=F.one_hot(torch.zeros((4,4),dtype=torch.long),5).float())
    state=dict(coords=torch.tensor([[[0.,0,1],[0,1,0],[1,0,0],[0,0,0]]]),mask=torch.ones((1,4),dtype=torch.bool),
        atomics=F.one_hot(torch.ones((1,4),dtype=torch.long),3).float(),bonds=F.one_hot(torch.ones((1,4,4),dtype=torch.long),5).float())
    controller=SimpleNamespace(sampling_policy=dict(chemical_inpainting=dict(enabled=True,step_bounds=[1,98],scope='full_molecule_graph')),
        rows=[{}],absolute_config={},absolute_state=SimpleNamespace(rows=[dict(slots=[3,1])]),
        model=SimpleNamespace(coord_scale=1),com=torch.zeros((1,3)),geometry_condition=cond,chemical_features=features,store=lambda *a:None)
    before=torch.get_rng_state().clone()
    out=ChemicalStateInpainting.apply_chemical_inpainting(controller,state,1)
    assert out['coords'] is state['coords'] and torch.equal(before,torch.get_rng_state())
    assert out['bonds'].argmax(-1).sum()==0
    assert out['atomics'].argmax(-1).tolist()==[[0,2,1,0]]


def test_promotion_filters_top10_pb_errors_and_stable_ties():
    rows=[dict(node_id=f'{i:02d}',build_success=True,connected=True,pic50_on_rescore=10-i*.1,**{fitted.METRIC:.2}) for i in range(15)]
    rows[0][fitted.METRIC]=.5;rows[1]['pic50_on_rescore']=float('nan')
    checked=[]
    def pb(row):
        checked.append(row['node_id'])
        return dict(posebusters_pass=row['node_id']!='02',posebusters_evaluable=True)
    def condition(row):
        if row['node_id']=='03':raise ValueError('No PLIP site')
        return {'next':row['node_id']}
    choice,audit=it.select(rows,dict(pic50=8),pb,condition)
    assert choice[0]['node_id']=='04' and len(checked)==10
    assert len(audit)==10
    assert it.select(rows,dict(pic50=100),pb,condition)[0] is None and len(checked)==10


def test_nonfinite_or_unbuilt_never_promoted():
    rows=[dict(node_id='x',build_success=False,connected=True,pic50_on_rescore=20,**{fitted.METRIC:.1})]
    assert it.shortlist(rows,1)==[]
    rows[0]['build_success']=True;rows[0][fitted.METRIC]=float('inf')
    assert it.shortlist(rows,1)==[]


def test_ten_rounds_execute_all_fifty_schedules_and_resume(tmp_path,monkeypatch):
    import test_early_path_search as old
    from agenticgen.uniguide import linear_runtime,absolute_runtime,checkpoint_learning
    from agenticgen.uniguide.geometry_protocol import choose
    from agenticgen.uniguide.io import write
    original=old.config
    def config(root):
        c=original(root);c['rounds']=10;c['agent_tools']['skill_release']='2.16.0'
        return old.validate_config(c)
    monkeypatch.setattr(old,'config',config)
    c,subject,domain,h,_,executor,calls=old.setup(tmp_path,monkeypatch)
    monkeypatch.setattr(absolute_runtime,'calibrated',lambda cfg:[subject])
    monkeypatch.setattr(linear_runtime,'source_binding',lambda:{'fixture':'216'})
    part=dict(early=[0,1],middle=[2,3],late=[4,5],reason='Temporal stages')
    def advisor(cfg,history,d,folder,skills,failures=(),final=False):
        state=cfg['guidance_checkpoint_context']['search_path_context']
        actions=[] if final else old.five(cfg,history,d,int(folder.name[-3:]),cfg['guidance_checkpoint_context'])
        focus=dict(stage='early',operation='start' if state['first_round'] else 'refine',source_id=choose(history)['id'])
        write(folder/'proposal.json',dict(proposals=actions))
        return dict(decision='hold' if final else 'propose',actions=actions,reason='Evidence',stage_partition=part,search_focus=focus)
    result=linear_runtime.run_version(c,'GD01',lambda *a:None,executor,advisor,lambda *a:None)
    assert len(calls)==60 and result['current']['completed_rounds']==10
    assert sum(x['count'] for x in calls if x['stage']=='formal')==2500
    assert linear_runtime.run_version(c,'GD01',None,None,None,None)==result and len(calls)==60


def test_rescore_uses_decoded_returned_endpoint_and_preserves_rng(monkeypatch):
    # CPU equivalent of the device-independent scoring arithmetic.
    original=torch.full
    monkeypatch.setattr(torch,'full',lambda *a,**kw:original(*a,**dict(kw,device='cpu')))
    class Model:
        coord_scale=2
        def _predict_affinity(self,a,b,p,t):
            assert a is b and len(a['coords'])==10
            assert torch.allclose(t[0],torch.full((10,),.9999))
            assert (a['atomics'].sum(-1)==1).all()
            return dict(affinity=dict(pic50=a['coords'].sum((1,2))[:,None]))
    output=dict(coords=torch.tensor([[[4.,2,0]]]),atomics=torch.tensor([[[.2,.8]]]),mask=torch.ones((1,1),dtype=torch.long))
    result=it.rescore(Model(),output,dict(mask=torch.ones((1,1))),torch.tensor([[2.,0,0]]))
    assert result==[2.] and output['coords'][0,0,0]==4


@pytest.mark.parametrize('atoms',[1,4,29])
def test_initial_teacher_rescore_supports_native_size_embedding(atoms):
    import torch.nn.functional as F
    channels=dict(atomics=torch.tensor([[.2,.8]]).repeat(atoms,1),
        bonds=F.one_hot(torch.zeros((atoms,atoms),dtype=torch.long),5).float())
    coordinates=[[4.,2.,0.]]*atoms
    endpoint=it.teacher_endpoint(channels,coordinates,'cpu')
    before=copy.deepcopy(endpoint);rng=torch.get_rng_state().clone()
    class Model:
        coord_scale=2
        def _predict_affinity(self,a,b,p,t):
            assert a is b and len(a['coords'])==10
            assert (a['mask'].sum(-1)==atoms).all()
            assert (a['atomics'].argmax(-1)==1).all()
            # This is the native FLOWR size-embedding operation that failed online.
            counts=a['mask'].sum(dim=-1,keepdim=True)
            embedded=F.embedding(counts,torch.arange(atoms+1,dtype=torch.float32)[:,None])
            return dict(affinity=dict(pic50=embedded[:,0]+a['coords'].sum((1,2))[:,None]))
    result=it.rescore(Model(),endpoint,dict(mask=torch.ones((1,1),dtype=torch.long)),torch.tensor([[2.,0.,0.]]))
    assert result==[float(3*atoms)]
    assert all(torch.equal(endpoint[k],before[k]) for k in endpoint)
    assert torch.equal(rng,torch.get_rng_state())


def test_fresh_prior_converts_native_float_mask_without_consuming_caller_rng(monkeypatch):
    import sys
    from types import ModuleType,SimpleNamespace
    from agenticgen.uniguide import sampler
    from agenticgen.runtime.rng import rng_state
    module=ModuleType('flowr.util.molrepr')
    seen=[]
    class Batch:
        @staticmethod
        def from_list(mols):
            assert len(mols)==50
            return SimpleNamespace(coords=torch.zeros((50,4,3)),atomics=torch.zeros((50,4,3)),
                bonds=torch.zeros((50,4,4,5)),charges=torch.zeros((50,4,2)),hybridization=None,
                # Matches upstream GeometricMolBatch.mask's float torch.ones.
                mask=torch.ones((50,4)))
    class Prior:
        def sample_molecule(self,n,symmetrize):
            assert n==4 and symmetrize is False
            seen.append(float(torch.rand(())));return object()
    module.GeometricMolBatch=Batch
    monkeypatch.setitem(sys.modules,'flowr.util.molrepr',module)
    monkeypatch.setattr(sampler,'population',lambda *a:dict(prior={},population=50))
    before=it.tensor_digest(rng_state())
    initial=it.fresh_initial({},'teacher',4,1,Prior(),False)
    # Exercise the exact consumer contract, rather than checking a dtype alone.
    counts=initial['prior']['mask'].sum(-1)
    embedded=torch.nn.functional.embedding(counts,torch.arange(5,dtype=torch.float32)[:,None])
    assert embedded[:,0].tolist()==[4.]*50
    assert len(seen)==100 and len(initial['stream_states'])==5
    assert it.tensor_digest(rng_state())==before


def test_iterative_five_round_budget_failure_and_replay(tmp_path,monkeypatch):
    from rdkit import Chem
    from agenticgen.uniguide import prepare,sampler,geometry_sampling,absolute_coordinate,geometry_chemistry
    from agenticgen.uniguide.io import write,value_sha
    import test_early_path_search as old
    folder=Path(__file__).resolve().parents[2]/'configs'
    cfg=read(folder/'uniguide_absolute_coordinate.json');cfg.update(read(folder/'uniguide_iterative_affinity.json'))
    cfg.update(root=str(tmp_path/'run'),generation_context_root=str(tmp_path/'context'),fixed_scale_source=str(tmp_path/'source'))
    cfg['conditions']=[dict(teacher_id='t',features_path=str(tmp_path/'features.json'))]
    mol=Chem.MolFromSmiles('CCCC');mol.AddConformer(Chem.Conformer(4));mol.GetConformer().Set3D(True)
    with Chem.SDWriter(str(tmp_path/'teacher.sdf')) as w:w.write(mol)
    features=dict(atoms=[dict(id=str(i),element='C') for i in range(4)],bonds=[],controlled_atom_ids=['0','1'],coordinates_angstrom=[[0,0,0]]*4)
    write(tmp_path/'features.json',features)
    write(tmp_path/'context/teachers.json',[dict(target='t',protein='p',sdf=str(tmp_path/'teacher.sdf'),node_id='initial',pic50=7)])
    write(tmp_path/'source/versions/GD01/FINALIZED.json',dict(complete=True,configured_rounds=10,teachers=dict(t=dict(representative='best'))))
    row=old.scored('best',[.1]*50,2);row['effective_guidance_scale_schedule']=[10]*6;write(tmp_path/'source/best/aggregate.json',row)
    monkeypatch.setattr(prepare,'load_model',lambda *a,**kw:(object(),None,False))
    monkeypatch.setattr(absolute_coordinate,'prepare',lambda f,*a:dict(f,condition_sha256=value_sha(f)))
    monkeypatch.setattr(geometry_chemistry,'condition_features',lambda *a:(dict(atomics=torch.zeros((4,3)),bonds=torch.zeros((4,4,5))),[],{}))
    monkeypatch.setattr(it,'device',lambda x:x)
    original_to=torch.Tensor.to
    monkeypatch.setattr(torch.Tensor,'to',lambda self,*a,**kw:original_to(self,*(['cpu',*a[1:]] if a and a[0]=='cuda' else a),**kw))
    for key in ('tensor','ones'):
        original=getattr(torch,key)
        monkeypatch.setattr(torch,key,lambda *a,_fn=original,**kw:_fn(*a,**dict(kw,device='cpu')))
    initial=dict(pocket_target=dict(mask=torch.ones((50,1))),target_com=torch.zeros((50,3)))
    monkeypatch.setattr(sampler,'population',lambda *a:copy.deepcopy(initial))
    monkeypatch.setattr(it,'fresh_initial',lambda *a:copy.deepcopy(initial))
    def rescore(model,out,*a):
        if len(out['coords'])==1:
            torch.nn.functional.embedding(out['mask'].sum(-1),torch.zeros((5,1)))
        return [7 if len(out['coords'])==1 else 8]*len(out['coords'])
    monkeypatch.setattr(it,'rescore',rescore)
    calls=[]
    def sample(model,c,t,s,dest,*a,**kw):
        from agenticgen.uniguide.geometry_sampling import GeometryControl
        from types import SimpleNamespace
        expanded=GeometryControl.expand_schedule(SimpleNamespace(scale_binding=kw['control_factory'].keywords['scale_binding']),s)
        assert expanded[0]==expanded[99]==0 and expanded[1:99]==[10]*98
        calls.append(dest.name)
        if dest.name=='003':raise FloatingPointError('injected')
        write(dest/'steps.json',[dict(step=i,chemical_inpainting_applied=[1<=i<=98]*50) for i in range(100)])
        return dict(coords=torch.zeros((50,4,3))),{},None,None
    monkeypatch.setattr(sampler,'sample',sample)
    def records(model,out,condition,dest,*a,**kw):
        (dest/'molecules_all_built.sdf').write_text('');write(dest/'COORDINATE_ASSIGNMENT.json',{})
        return [dict(node_id=str(i),tensor_row=i,build_success=False,connected=False,**{fitted.METRIC:.1}) for i in range(50)]
    monkeypatch.setattr(geometry_sampling,'terminal_records',records)
    result=it.run(cfg)
    assert calls==['acceptance','001','002','003','004','005'] and result['attempts']==250
    assert not result['complete'] and result['rounds'][2]['failed']
    again=it.run(cfg)
    assert again==result and len(calls)==6
