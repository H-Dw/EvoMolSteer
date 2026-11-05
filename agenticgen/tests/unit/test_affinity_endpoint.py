"""Scoring, teacher/frame identity and engineering replay regression contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from agenticgen.uniguide import iterative_affinity as it,affinity_inputs as inputs,affinity_replay as replay
from agenticgen.uniguide.io import read,write,sha,value_sha
from agenticgen.uniguide.native_tools import save_pt,tensor_digest


def endpoint(b=1,n=4):
    return dict(coords=torch.arange(b*n*3,dtype=torch.float32).reshape(b,n,3)+4,
        atomics=torch.tensor([.2,.8]).expand(b,n,2).clone(),
        charges=torch.tensor([.9,.1]).expand(b,n,2).clone(),
        bonds=torch.tensor([.7,.1,.1,.05,.05]).expand(b,n,n,5).clone(),
        mask=torch.ones((b,n),dtype=torch.long))


class Scorer(torch.nn.Module):
    coord_scale=2.
    def __init__(self,where='cpu'):
        super().__init__();self.register_parameter('weight',torch.nn.Parameter(torch.zeros((),device=where)))
    def _predict_affinity(self,a,b,p,t):
        assert a is b
        for group in (a,p):
            assert all(v.device==self.weight.device for v in group.values() if torch.is_tensor(v))
        assert all(x.device==self.weight.device and (x==.9999).all() for x in t)
        assert a['mask'].dtype==torch.long
        # Real consumer of integer atom-count indices; also check row/pocket pairing.
        counts=torch.nn.functional.embedding(a['mask'].sum(-1),torch.arange(20.,device=self.weight.device)[:,None])
        value=a['coords'].sum((1,2))+p['coords'].sum((1,2))+counts[:,0]
        return dict(affinity=dict(pic50=value[:,None]))


def test_cpu_raw_endpoint_centering_decoding_and_input_isolation():
    out=endpoint(13);pocket=dict(coords=torch.arange(39.).reshape(13,1,3),mask=torch.ones((13,1),dtype=torch.int32))
    com=torch.arange(39.).reshape(13,3)
    before=tensor_digest(dict(out=out,pocket=pocket,com=com));rng=torch.get_rng_state().clone()
    scores=it.rescore(Scorer(),out,pocket,com)
    expected=((out['coords']-com[:,None,:])/2).sum((1,2))+pocket['coords'].sum((1,2))+4
    assert scores==expected.tolist() and len(scores)==13
    assert before==tensor_digest(dict(out=out,pocket=pocket,com=com))
    assert torch.equal(rng,torch.get_rng_state())
    final,p,t=inputs.scoring_inputs(Scorer(),out,pocket,com)
    for name in ('atomics','charges','bonds'):assert (final[name].sum(-1)==1).all()
    assert (final['atomics'].argmax(-1)==1).all()
    final['coords'].zero_();p['coords'].zero_()
    assert before==tensor_digest(dict(out=out,pocket=pocket,com=com))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires the real GPU scoring boundary')
def test_cpu_terminal_gpu_model_and_mixed_com_pocket_devices():
    model=Scorer('cuda');out=endpoint(50)
    pocket=dict(coords=torch.ones((50,3,3)),mask=torch.ones((50,3),dtype=torch.int32))
    com=torch.ones((50,3),device='cuda')
    before=tensor_digest(dict(out=out,pocket=pocket,com=com));rng=torch.cuda.get_rng_state().clone()
    expected=it.rescore(Scorer(),out,pocket,com.cpu())
    assert it.rescore(model,out,pocket,com)==expected
    assert it.rescore(model,out,{k:v.cuda() for k,v in pocket.items()},com.cpu())==expected
    assert before==tensor_digest(dict(out=out,pocket=pocket,com=com))
    assert torch.equal(rng,torch.cuda.get_rng_state())


def test_same_endpoint_input_as_experiment_one_runtime_scoring(tmp_path):
    from agenticgen.runtime.scoring import final_metrics
    class Captured(Exception):pass
    class Capture(Scorer):
        def _predict_affinity(self,a,b,p,t):
            self.inputs=copy.deepcopy((a,p,t));raise Captured
    model=Capture();out=endpoint(10);com=torch.arange(30.).reshape(10,3)
    pocket=dict(coords=torch.ones((10,3,3)),mask=torch.ones((10,3),dtype=torch.long),
        complex=[SimpleNamespace(com=x.numpy()) for x in com])
    with pytest.raises(Captured):final_metrics(model,out,SimpleNamespace(pt=pocket,po=pocket),tmp_path)
    model.inputs[1].pop('complex')
    first=tensor_digest(model.inputs)
    with pytest.raises(Captured):it.rescore(model,out,pocket,com)
    model.inputs[1].pop('complex')
    assert first==tensor_digest(model.inputs)


def test_score_float_dtype_is_the_model_dtype_without_changing_integer_metadata():
    out={k:v.double() if v.is_floating_point() else v for k,v in endpoint().items()}
    pocket=dict(coords=torch.ones((1,2,3),dtype=torch.float64),mask=torch.ones((1,2),dtype=torch.int32))
    final,p,t=inputs.scoring_inputs(Scorer(),out,pocket,torch.zeros((1,3),dtype=torch.float64))
    assert final['coords'].dtype==final['atomics'].dtype==p['coords'].dtype==t[0].dtype==torch.float32
    assert final['mask'].dtype==torch.int64 and p['mask'].dtype==torch.int32
    assert out['coords'].dtype==pocket['coords'].dtype==torch.float64


@pytest.mark.parametrize('fault',['coords','mask','bonds','center','scale'])
def test_invalid_endpoint_inputs_fail_before_network(fault):
    model=Scorer();out=endpoint();com=torch.zeros((1,3))
    if fault=='coords':out['coords'][0,0,0]=float('nan')
    elif fault=='mask':out['mask'][0,0]=2
    elif fault=='bonds':out['bonds']=out['bonds'][:,:,:-1]
    elif fault=='center':com=com[:,:2]
    else:model.coord_scale=0
    with pytest.raises(ValueError):it.rescore(model,out,dict(coords=torch.ones((1,1,3))),com)


def initial_state(n=4):
    return dict(population=50,prior=endpoint(50,n),target_com=torch.ones((50,3)),
        stream_states=[{}]*5,pocket_target=dict(coords=torch.ones((50,3,3))),pocket_off=dict(coords=torch.ones((50,3,3))))


def test_teacher_size_changes_preserve_fixed_protein_frame_and_dense_schema():
    reference=initial_state();initial=initial_state(7);condition=dict(atoms=[{}]*7)
    assert inputs.audit_initial(initial,reference,condition)['atoms']==7
    initial['target_com'][0,0]=2
    with pytest.raises(ValueError,match='frame'):inputs.audit_initial(initial,reference,condition)
    initial=initial_state(7);initial['prior']['mask'][0,0]=0
    with pytest.raises(ValueError,match='counts'):inputs.audit_initial(initial,reference,condition)
    initial=initial_state(7);initial['pocket_target']['coords'][0,0,0]=2
    with pytest.raises(ValueError,match='frame'):inputs.audit_initial(initial,reference,condition)


def test_teacher_features_and_condition_cannot_drift(tmp_path,monkeypatch):
    from agenticgen.uniguide import absolute_coordinate,geometry_chemistry
    protein=tmp_path/'protein';protein.write_text('fixed frame')
    features=dict(atoms=[dict(id='a')],bonds=[],coordinates_angstrom=[[0,0,0]],controlled_atom_ids=['a'])
    condition=dict(features_sha256=value_sha(features),atoms=features['atoms'],bonds=[],
        reference_coordinates_angstrom=features['coordinates_angstrom'],controlled_atom_ids=['a'],condition_sha256='c',
        protein_alignment=dict(moving_protein_sha256=sha(protein)))
    monkeypatch.setattr(absolute_coordinate,'validate',lambda c:c)
    monkeypatch.setattr(geometry_chemistry,'condition_features',lambda *a:({},[],dict(chemical_condition_sha256='chem')))
    teacher=dict(id='teacher',source_id='candidate-id',features=features,condition=condition)
    assert inputs.audit_teacher(None,teacher,protein)['atoms']==1
    teacher['features']=dict(features,coordinates_angstrom=[[1,0,0]])
    with pytest.raises(ValueError,match='bound'):inputs.audit_teacher(None,teacher,protein)


def test_initial_scoring_molecule_matches_geometry_graph_and_plip_region(tmp_path):
    from rdkit import Chem
    from agenticgen.uniguide.geometry_condition import from_molecule
    mol=Chem.MolFromSmiles('CCCO');mol.AddConformer(Chem.Conformer(4));mol.GetConformer().Set3D(True)
    features=from_molecule(mol,['atom_0000','atom_0003'],coordinate_frame='generation_context_angstrom')
    protein=tmp_path/'protein';protein.write_text('fixed protein')
    sdf=tmp_path/'teacher.sdf'
    with Chem.SDWriter(str(sdf)) as writer:writer.write(mol)
    template=dict(sdf=str(sdf),protein=str(protein),protein_sha256=sha(protein),heavy_atoms=4,core_sdf_atoms=[0,3])
    assert inputs.audit_original_molecule(features,mol,template)['graph_matches']
    bad=copy.deepcopy(features);bad['coordinates_angstrom'][0][0]+=.01
    with pytest.raises(ValueError,match='coordinates'):inputs.audit_original_molecule(bad,mol,template)
    bad=copy.deepcopy(features);bad['atoms'][0]['element']='N'
    with pytest.raises(ValueError,match='graph'):inputs.audit_original_molecule(bad,mol,template)
    with pytest.raises(ValueError,match='PLIP'):inputs.audit_original_molecule(features,mol,dict(template,core_sdf_atoms=[0,1]))


def cache_fixture(tmp_path):
    source=tmp_path/'old';acceptance=source/'acceptance';destination=tmp_path/'new/acceptance'
    cfg=dict(root=str(source),checkpoint_sha256='frozen',windows=[[0,10],[11,99]])
    teacher=dict(id='initial',condition={'condition_sha256':'c'},features={'atoms':[]},source_id='initial-id',pic50=5.)
    scale=dict(scale=[50,200],aggregate_sha256='fixed')
    for name,value in [('GEOMETRY_CONFIG.json',cfg),('BASELINE.json',teacher),('SCALE_SOURCE.json',scale),
            ('SOURCE_BINDING.json',dict(implementation={'box_search.py':sha(Path(it.__file__).with_name('box_search.py'))}))]:write(source/name,value)
    write(acceptance/'steps.json',[dict(step=i,chemical_inpainting_applied=[1<=i<=98]*50,
        **{f'chemical_{k}_after_match_fraction':[1.]*50 for k in ('atomics','charges','bonds')}) for i in range(100)])
    write(acceptance/'progress.json',dict(step_completed=100))
    write(acceptance/'OUTPUT_POLICY.json',dict(output_policy='raw999',extra_terminal_forward=False,chemical_projection=False,
        last_loop=dict(step=99,output_is_uncovered_prediction=True)))
    for name in ('INPUT.json','CHEMICAL_CONDITION.json','COORDINATE_ASSIGNMENT.json'):write(acceptance/name,{})
    write(acceptance/'rng_ledger.json',[{}]*100);(acceptance/'trace.h5').write_bytes(b'opaque trace, hash bound')
    save_pt(acceptance/'initial.pt.gz',initial_state());save_pt(acceptance/'terminal.pt.gz',endpoint(50))
    save_pt(acceptance/'checkpoint_100.pt.gz',dict(step=100))
    spec=dict(source_attempt=str(source),source_commit='old-frozen-commit',files={name:sha(source/name) for name in replay.REPLAY_FILES})
    new=dict(cfg,root=str(destination.parent),engineering_replay=spec)
    return source,destination,new,teacher,scale


def test_completed_engineering_replay_does_not_sample_or_reserve_science(tmp_path):
    source,dest,cfg,teacher,scale=cache_fixture(tmp_path)
    initial,out=replay.replay(cfg,dest,teacher,scale)
    assert initial['population']==len(out['coords'])==50
    receipt=read(dest/'ENGINEERING_REPLAY.json')
    assert receipt['new_trajectory_evaluations']==receipt['scientific_candidates']==0
    assert not (dest.parent/'attempts.json').exists()
    assert all(sha(source/name)==digest for name,digest in cfg['engineering_replay']['files'].items())
    with pytest.raises(ValueError,match='fresh'):replay.replay(cfg,dest,teacher,scale)


@pytest.mark.parametrize('fault',['hash','config','teacher','scale','science','partial','chemistry','projection','path'])
def test_replay_rejects_incompatible_or_consumed_sources(tmp_path,fault):
    source,dest,cfg,teacher,scale=cache_fixture(tmp_path)
    if fault=='hash':(source/'acceptance/trace.h5').write_bytes(b'changed')
    elif fault=='config':cfg['checkpoint_sha256']='different'
    elif fault=='teacher':teacher=dict(teacher,pic50=5.1)
    elif fault=='scale':scale=dict(scale,scale=[30,100])
    elif fault=='science':write(source/'attempts.json',{'formal':dict(count=50)})
    elif fault=='path':cfg['engineering_replay']['files']['../unsafe']='digest'
    else:
        path=source/'acceptance'/('steps.json' if fault in ('partial','chemistry') else 'OUTPUT_POLICY.json')
        obj=read(path)
        if fault=='partial':obj=obj[:-1]
        elif fault=='chemistry':obj[30]['chemical_atomics_after_match_fraction'][0]=.9
        else:obj['chemical_projection']=True
        write(path,obj);cfg['engineering_replay']['files']['acceptance/'+path.name]=sha(path)
    with pytest.raises(ValueError):replay.replay(cfg,dest,teacher,scale)
    assert not dest.exists()
