"""Run on the specified remote host; no local test execution."""
import copy
import math
import numpy as np
import pytest
import torch
from agenticgen.uniguide.condition import condition_map,coordinate_loss,guidance_delta
from agenticgen.uniguide.protocol import validate_action,choose,coarse_scales,expand,regressed
from agenticgen.uniguide.io import Budget,read,frozen
from agenticgen.uniguide.review import teacher_triggers,recover_base,vector_change


def test_fixed_assignment_vjp_and_scale_conversion():
    x=torch.tensor([[[.2,1.,0.],[2.,.4,0.],[3.,1.,.5]]],dtype=torch.double,requires_grad=True)
    # Cross-slot Jacobian deliberately gives non-core coordinates a gradient.
    endpoint=2*(x+.2*x.mean(1,keepdim=True))+torch.tensor([4.,5.,6.])
    target,selected,assignment=condition_map(endpoint,[[4.,5.,6.]],torch.ones(1,3,dtype=torch.bool))
    assert not target.requires_grad and not selected.requires_grad and len(set(assignment[0].tolist()))==1
    gradient=torch.autograd.grad(coordinate_loss(endpoint,target,selected).sum(),x)[0]
    assert gradient[0,2].abs().sum()>0
    direction=torch.tensor([[[.1,.2,.3],[.4,.1,-.2],[-.1,.3,.1]]],dtype=torch.double)
    def f(z): return coordinate_loss(2*(z+.2*z.mean(1,keepdim=True))+torch.tensor([4.,5.,6.]),target,selected).sum()
    eps=1e-6
    numerical=(f(x.detach()+eps*direction)-f(x.detach()-eps*direction))/(2*eps)
    assert torch.allclose(numerical,(gradient*direction).sum(),rtol=1e-7,atol=1e-8)


def test_flow_score_algebra_linear_scale_and_no_clipping():
    x=torch.randn(2,5,3,dtype=torch.double);v=torch.randn_like(x);g=torch.randn_like(x)
    t=torch.tensor([.1,.7],dtype=torch.double);gamma=.4;S=torch.tensor([50.,300.],dtype=torch.double);dt=.01
    q=(t[:,None,None]*v-x)/(1-t+1e-6)[:,None,None]
    q2=q-S[:,None,None]*g
    v2=(q2*(1-t+1e-6)[:,None,None]+x)/t[:,None,None]
    delta,_=guidance_delta(g,t,dt,S,gamma,torch.ones(2,5))
    assert torch.allclose(delta,dt*((v2+gamma*q2)-(v+gamma*q)),rtol=1e-12,atol=1e-12)
    doubled,_=guidance_delta(g,t,dt,2*S,gamma,torch.ones(2,5))
    assert torch.equal(doubled,2*delta) and delta.abs().max()>.025
    zero,_=guidance_delta(g,torch.zeros(2),dt,[0,0],gamma,torch.ones(2,5))
    assert torch.equal(zero,torch.zeros_like(g))
    with pytest.raises(FloatingPointError): guidance_delta(g*float('nan'),t,dt,S,gamma,torch.ones(2,5))


def test_condition_map_injective_and_unmatched_targets_equal_prediction():
    x=torch.tensor([[[0.,0,0],[1.,0,0],[9.,0,0],[.1,0,0]]])
    valid=torch.tensor([[1,1,1,0]],dtype=torch.bool)
    target,mask,ids=condition_map(x,[[.1,0,0],[.2,0,0]],valid)
    assert set(ids[0].tolist())=={0,1}
    assert torch.equal(target[~mask],x[~mask])
    with pytest.raises(ValueError): condition_map(x,[],valid)
    with pytest.raises(FloatingPointError): condition_map(x+1e30,[[0.,0,0]],valid)


def test_log_actions_no_rounding_duplicate_or_hidden_endpoint():
    anchors={'a':{'schedule':[1.]*4}};domain={'min_index':-4,'max_index':4}
    p=dict(anchor_id='a',mode='coarse',branch='redistribute',schedule=[2.,.5,1.,1.])
    assert validate_action(p,anchors,domain,[])==p['schedule']
    for changes in [{'schedule':[3.,.5,1,1]},{'schedule':[2.,1,1,1]},{'mode':'fine'},{'branch':'increase'}]:
        with pytest.raises(ValueError): validate_action(dict(p,**changes),anchors,domain,[])
    with pytest.raises(ValueError): validate_action(p,anchors,domain,[p['schedule']])
    assert len(coarse_scales())==26
    e=expand([1,2,3,4]);assert e[9]==0 and e[10]==1 and e[25]==1 and e[26]==2 and e[90]==4 and e[91]==0


def test_budget_failed_attempts_remain_charged(tmp_path):
    b=Budget(tmp_path);b.reserve('a','calibration',4990,'x');b.reserve('a','calibration',4990,'x')
    with pytest.raises(ValueError): b.reserve('b','calibration',20,'y')
    with pytest.raises(ValueError): b.reserve('a','calibration',10,'y')
    b.reserve('b','calibration',10,'y');assert sum(x['count'] for x in read(tmp_path/'attempts.json').values())==5000


def test_frozen_json_roundtrip_does_not_reject_plip_tuples(tmp_path):
    p=tmp_path/'template.json'
    frozen(p,{'fingerprints':[('hbond',1,'A')]})
    frozen(p,{'fingerprints':[('hbond',1,'A')]})
    with pytest.raises(ValueError): frozen(p,{'fingerprints':[('hbond',2,'A')]})


def test_empty_built_structure_file_is_a_reportable_population(tmp_path):
    from agenticgen.uniguide.quality import built_molecules
    p=tmp_path/'none_built.sdf';p.write_text('')
    assert built_molecules(p)=={}


def test_pdb_fixed_width_overflow_is_not_coordinate_correction():
    from agenticgen.uniguide.profiling import check_pdb_coordinates,PDBRepresentationError
    check_pdb_coordinates([[-999.999,0,9999.999]])
    for value in [-1109.4329,29230066.,-999.9999,9999.9999]:
        original=np.array([[0.,value,0.]])
        with pytest.raises(PDBRepresentationError) as error:check_pdb_coordinates(original)
        assert error.value.reason=='pdb_coordinate_field_overflow'
        assert original[0,1]==value


def test_posebusters_exception_is_unknown_not_a_pass(tmp_path,monkeypatch):
    from pathlib import Path
    from rdkit import Chem
    import posebusters
    from agenticgen.uniguide.profiling import quality
    source=Chem.MolFromMolFile(str(Path(__file__).parent/'fixtures/uniguide_explicit_h/source.mol'))
    before=Chem.MolToMolBlock(source)
    protein=tmp_path/'protein.pdb'
    protein.write_text('ATOM      1  C   ALA A   1      99.000  99.000  99.000  1.00  0.00           C  \n')
    class OverflowEvaluator:
        def __init__(self,**kwargs):pass
        def bust(self,*args,**kwargs):raise IndexError('-1003147201')
    monkeypatch.setattr(posebusters,'PoseBusters',OverflowEvaluator)
    result=quality(source,protein)
    assert result['posebusters_pass'] is None and result['posebusters_evaluable'] is False
    assert result['posebusters_error']['type']=='IndexError'
    assert result['valid_connected'] and result['severe_clash'] is False
    assert Chem.MolToMolBlock(source)==before


def test_unevaluable_plip_keeps_attempt_and_other_quality(tmp_path,monkeypatch):
    from pathlib import Path
    from rdkit import Chem
    from agenticgen.uniguide.io import write
    import agenticgen.uniguide.profiling as profiling
    from agenticgen.uniguide.quality import assess
    mol=Chem.MolFromMolFile(str(Path(__file__).parent/'fixtures/uniguide_explicit_h/source.mol'))
    mol.GetConformer().SetAtomPosition(0,(29230066.,0.,0.));mol.SetProp('node_id','overflow')
    protein=tmp_path/'protein.pdb';protein.write_text('fixture protein')
    group=tmp_path/'group';group.mkdir()
    with Chem.SDWriter(str(group/'molecules_all_built.sdf')) as writer:writer.write(mol)
    write(group/'records.json',[{'node_id':'overflow'},{'node_id':'unbuilt'}])
    before=(group/'molecules_all_built.sdf').read_bytes()
    def profile(mol,protein,out):
        profiling.check_pdb_coordinates(mol.GetConformer().GetPositions())
        raise AssertionError('Must reject encoding before PLIP')
    monkeypatch.setattr(profiling,'plip_profile',profile)
    monkeypatch.setattr(profiling,'quality',lambda mol,protein:{'posebusters_pass':False,'valid_connected':False,'severe_clash':False})
    result=assess(group,{'protein':str(protein),'plip_profile':{'fingerprints':[['hbond','A','1','ALA','ligand_acceptor']]}})
    assert result['attempts']==2 and result['built']==1 and result['plip_evaluable']==0
    assert result['plip_unevaluable_built']==1 and result['plip_unevaluable_reasons']=={'pdb_coordinate_field_overflow':1}
    rows=read(group/'quality.json')
    assert rows[0]['plip_recovery'] is None and rows[0]['quality']['plip_evaluable'] is False
    assert rows[0]['quality']['posebusters_pass'] is False
    assert (group/'molecules_all_built.sdf').read_bytes()==before
    assert assess(group,{'protein':str(protein)})==result


def test_quality_enqueue_is_nonblocking_full_population_and_durable(tmp_path,monkeypatch):
    from pathlib import Path
    from rdkit import Chem
    from agenticgen.uniguide.io import write
    import agenticgen.uniguide.quality as quality
    import agenticgen.uniguide.quality_queue as queue
    protein=tmp_path/'protein.pdb'
    protein.write_text('ATOM      1  C   ALA A   1      99.000  99.000  99.000  1.00  0.00           C  \n')
    teacher={'protein':str(protein),'plip_profile':{'fingerprints':[]}}
    folder=tmp_path/'group';folder.mkdir()
    source=Chem.MolFromMolFile(str(Path(__file__).parent/'fixtures/uniguide_explicit_h/source.mol'))
    with Chem.SDWriter(str(folder/'molecules_all_built.sdf')) as stream:
        for i in range(49):
            mol=Chem.Mol(source);mol.SetProp('node_id',str(i));stream.write(mol)
    write(folder/'records.json',[{'node_id':str(i)} for i in range(50)])
    write(folder/'COMPLETE.json',{'fixture':'already generated'})
    def must_not_block(*args):raise AssertionError('Slow quality entered inference path')
    monkeypatch.setattr(quality,'assess',must_not_block)
    launches=[];monkeypatch.setattr(queue,'ensure_worker',lambda config:launches.append(config['root']))
    config={'root':str(tmp_path)}
    light=queue.enqueue(config,folder,teacher)
    assert light['attempts']==50 and light['built']==49 and light['clash_evaluable']==49
    assert 'posebusters_pass' not in light
    requests=list((tmp_path/'quality_queue/requests').glob('*.json'))
    assert len(requests)==1 and read(requests[0])['population']==50
    assert read(requests[0])['scope']=='all_attempts' and launches==[str(tmp_path)]
    # Re-enqueue is idempotent and does not sample, wait, or change metrics.
    assert queue.enqueue(config,folder,teacher)==light
    assert len(list((tmp_path/'quality_queue/requests').glob('*.json')))==1
    assert queue.snapshot(tmp_path,['group'])['pending']==1
    # A slow-worker error stays distinct from a bad molecule and never passes.
    assert queue.run_request(requests[0])=='failed'
    assert queue.snapshot(tmp_path,['group'])['failed']==1
    assert read(folder/'QUALITY_FAILED.json')['does_not_imply_molecular_failure']
    # Worker checks all attempts and preserves source identity, including failures.
    (folder/'QUALITY_FAILED.json').unlink()
    calls=[]
    def measured(folder,teacher):
        calls.append(str(folder));write(folder/'QUALITY_COMPLETE.json',{'attempts':50});return {'attempts':50}
    monkeypatch.setattr(quality,'assess',measured)
    assert queue.run_request(requests[0])=='complete' and len(calls)==1
    assert queue.snapshot(tmp_path,['group'])['complete']==1
    (folder/'QUALITY_COMPLETE.json').unlink()
    write(folder/'records.json',[{'node_id':'changed'}])
    assert queue.run_request(requests[0])=='failed' and len(calls)==1
    assert 'source changed' in read(folder/'QUALITY_FAILED.json')['error']


def test_plip_roundtrip_imine_h_representation_preserves_chemistry_and_pose():
    from pathlib import Path
    from rdkit import Chem
    from agenticgen.uniguide.profiling import validate_roundtrip
    fixture=Path(__file__).parent/'fixtures/uniguide_explicit_h'
    source=Chem.MolFromMolFile(str(fixture/'source.mol'),removeHs=True)
    actual=Chem.MolFromMolFile(str(fixture/'openbabel_roundtrip.sdf'),removeHs=True)
    # Real failed generated pose: default removal retains an imine stereo H.
    assert source.GetNumAtoms()==17 and actual.GetNumAtoms()==18
    before=Chem.MolToMolBlock(actual)
    assert validate_roundtrip(source,actual)<1e-3
    assert Chem.MolToMolBlock(actual)==before
    # Different atom ordering must not change this verdict.
    assert validate_roundtrip(source,Chem.RenumberAtoms(actual,list(reversed(range(18)))))<1e-3
    changed=Chem.Mol(actual);changed.GetConformer().SetAtomPosition(0,(100,100,100))
    with pytest.raises(ValueError,match='coordinate'):validate_roundtrip(source,changed)
    changed=Chem.Mol(actual)
    oxygen=next(a for a in changed.GetAtoms() if a.GetFormalCharge()==-1)
    oxygen.SetFormalCharge(0);oxygen.SetNumExplicitHs(1)
    Chem.SanitizeMol(changed)
    with pytest.raises(ValueError,match='graph'):validate_roundtrip(source,changed)
    changed=Chem.Mol(actual);changed.GetAtomWithIdx(0).SetIsotope(13)
    with pytest.raises(ValueError,match='chemistry'):validate_roundtrip(source,changed)
    changed=Chem.Mol(actual)
    bond=next(b for b in changed.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE and
              {b.GetBeginAtom().GetSymbol(),b.GetEndAtom().GetSymbol()}=={'C','N'})
    bond.SetBondType(Chem.BondType.SINGLE);bond.SetStereo(Chem.BondStereo.STEREONONE)
    Chem.SanitizeMol(changed)
    with pytest.raises(ValueError,match='graph'):validate_roundtrip(source,changed)


def row(i,c,g): return {'id':i,'C':c,'gap':g,'best':8.,'complete':True}


def test_anchored_selection_and_two_adjacent_regressions():
    assert choose([row('a',1,.5),row('b',1.019,.1),row('c',1.03,0)])['id']=='b'
    versions=[{'id':f'UG{i+1:02d}','overall':row(f'UG{i+1:02d}',c,g)} for i,(c,g) in enumerate([(1,.5),(1.04,.4),(1.08,.3)])]
    assert recover_base(versions)==('UG01',True)
    versions.append({'id':'UG04','overall':row('UG04',1.12,.2),'recovery_derived':True})
    assert recover_base(versions)==('UG04',False)
    assert regressed(row('a',1,.1),row('b',1.01,.2))
    assert not regressed(row('a',1,.2),row('b',1.01,.1))


def test_stagnation_either_teacher_and_missing_vectors(tmp_path):
    old=row('a',1,.2);new=row('b',1.01,.21)
    evidence=teacher_triggers(old,new,[[1]*4]*15,[[1]*4]*12+[[2]*4]*3,{'known':False})
    assert len(evidence['reasons'])==2
    assert not vector_change(tmp_path/'missing',tmp_path/'missing')['known']
    import h5py
    for name,sign in [('a',1),('b',-1)]:
        with h5py.File(tmp_path/name,'w') as f:f['guidance_delta_A']=np.ones((100,2,3,3))*sign
    changed=vector_change(tmp_path/'a',tmp_path/'b')
    assert changed['known'] and changed['relative_change']==[2.]*4
    assert not changed['all_windows_at_most_5pct']


def test_native_twenty_row_crn_is_shared_and_external_rng_unchanged():
    from agenticgen.uniguide.random_streams import NativeStreams,capture,equal
    states=[]
    for seed in [42,100045,200048,300051,400054]:
        torch.manual_seed(seed);states.append(capture())
    outside=capture()
    streams=NativeStreams(states,3,10)
    samples=streams.normal('randn',(150,17,3),dtype=torch.float32)
    assert torch.equal(samples[:50],samples[50:100]) and torch.equal(samples[:50],samples[100:])
    assert equal(outside,capture())
    probabilities=torch.full((150,17,4),.25)
    cats=streams.categorical(probabilities)
    assert torch.equal(cats[:50],cats[50:100]) and torch.equal(cats[:50],cats[100:])
    assert equal(outside,capture())


def test_chunking_preserves_nested_native_tensordict_and_input_jacobian():
    from tensordict import TensorDict
    from agenticgen.uniguide.chunking import chunk_call
    x=torch.arange(21.,requires_grad=True).reshape(7,3)
    called=[]
    def network(batch):
        called.append(len(batch['coords']))
        return {'coords':batch['coords'].square(),'affinity':TensorDict({'pic50':batch['coords'].sum(-1,keepdim=True)},batch_size=[len(batch['coords'])])}
    output=chunk_call(network,({'coords':x},),{},7,rows=3,pad_to=10)
    assert called==[10,10,10]
    assert torch.equal(output['coords'],x.square())
    assert torch.equal(output['affinity']['pic50'],x.sum(-1,keepdim=True))
    assert torch.equal(torch.autograd.grad(output['coords'].sum(),x)[0],2*x)


def test_partial_batch_keeps_row_metadata_and_structural_time_tensors():
    from agenticgen.uniguide.chunking import chunk_call,concatenate
    coords=torch.arange(75.,requires_grad=True).reshape(25,3)
    metadata=[f'row_{i}' for i in range(25)]
    def network(batch):
        x=batch['coords']
        return {'coords':x.square(),'fragment_mode':batch['fragment_mode'],
                'times':[x[:,0:1],x[:,1:2]]}
    output=chunk_call(network,({'coords':coords,'fragment_mode':metadata},),{},25,pad_to=10)
    assert output['fragment_mode']==metadata
    assert len(output['times'])==2 and torch.equal(output['times'][0],coords[:,0:1])
    assert torch.equal(output['coords'],coords.square())
    assert torch.equal(torch.autograd.grad(output['coords'].sum(),coords)[0],2*coords)
    # The metadata rule also applies to equal chunks, where the old code
    # silently retained only the first chunk's metadata instead of 20 rows.
    assert concatenate([metadata[:10],metadata[10:20]],[10,10])==metadata[:20]


def test_version_snapshot_is_immutable_and_ug21_is_rejected(tmp_path):
    from agenticgen.uniguide.pipeline import freeze_version
    seed=tmp_path/'UG_INIT/skills';seed.mkdir(parents=True)
    (seed/'analysis.md').write_text('Selected V17 ideas, no old scores')
    config={'root':str(tmp_path),'llm':{'model':'deepseek-flash'}}
    domain={'initial':[1.]*4,'min_index':-4,'max_index':4}
    first=freeze_version(config,'UG01',domain)
    assert first['base_checkpoint']=='UG_INIT' and first['dynamic_memory_reset']
    assert freeze_version(config,'UG01',domain)==first
    (tmp_path/'versions/UG01/skills/analysis.md').write_text('illegal mid-version edit')
    with pytest.raises(ValueError): freeze_version(config,'UG01',domain)
    with pytest.raises(ValueError): freeze_version(config,'UG21',domain)


def test_final_version_review_never_schedules_ug21(tmp_path):
    from agenticgen.uniguide.review import build_review
    from agenticgen.uniguide.io import write
    folder=tmp_path/'versions/UG20'
    write(folder/'FINALIZED.json',{'id':'UG20','overall':row('UG20',.3,.2),'teachers':{},'strategy_contexts':['a','b','c','d']})
    result=build_review({'root':str(tmp_path)},'UG20')
    assert result['next_version'] is None and result['no_best_rerun']


def test_synthesis_selects_existing_actions_and_final_round_has_no_proposals(tmp_path,monkeypatch):
    import jsonschema
    import agenticgen.uniguide.council as council
    from agenticgen.uniguide.io import write
    for phase in ['coarse','fine','windows','full']:
        write(tmp_path/'calibration'/(phase+'_results.json'),[])
    skills=tmp_path/'skills';skills.mkdir();(skills/'policy.md').write_text('Fixture policy')
    history=[dict(row('R0',1.,.2),schedule=[1.]*4)]
    teacher={'target':'test','node_id':'t','pic50':8.,'heavy_atoms':17}
    domain={'min_index':-4,'max_index':4}
    options={'increase':[[2,1,1,1],[1,2,1,1]],'decrease':[[.5,1,1,1],[1,.5,1,1]],
             'redistribute':[[2,.5,1,1],[1,1,2,.5]]}
    calls=[]
    def provider(messages,schema,output,response_validator,**kwargs):
        name=output.stem;calls.append(name)
        if name=='synthesis':
            result={'analysis':'measured comparison','choice_evidence':'existing candidate IDs'}
            if 'selected' in schema['properties']:result['selected']={b:b+':0' for b in options}
        else:
            proposals=[]
            if schema['properties']['proposals']['maxItems']:
                proposals=[dict(id=str(i),branch=name,anchor_id='R0',anchor_reason='measured',mode='coarse',schedule=s,
                    evidence='R0',prediction='testable',falsifier='wrong direction') for i,s in enumerate(options[name])]
            result={'analysis':'measured','mechanism_uncertainty':'explicit','proposals':proposals}
        jsonschema.validate(result,schema);response_validator(result)
        return result
    monkeypatch.setattr(council,'call_json',provider)
    config={'root':str(tmp_path),'llm':{}}
    selected=council.advise(config,teacher,'UG01',1,history,domain,skills,tmp_path/'R1')
    assert [p['schedule'] for p in selected]==[options[b][0] for b in options]
    assert calls==['increase','decrease','redistribute','synthesis']
    assert council.advise(config,teacher,'UG01',5,history,domain,skills,tmp_path/'R5',final=True)==[]
