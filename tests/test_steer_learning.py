"""Meaningful capture contracts: failed siblings, bounds, native precision, launch."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from evomolsteer.io import digest,read_json
from evomolsteer.storage.arrays import byte_equal,assert_no_filters
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.storage.selection_dataset import SelectionLearningWriter,selection_steps,read_scoring_event,verify_learning_batch,terminal_descendants
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.generation.steer_trace import LearningTraceMixin,terminal_numeric_arrays
from evomolsteer.generation import steer_launcher
from evomolsteer.generation.steer_launcher import SteerLearningConfig,launch,resolve_inputs,INPUT_FORMAT


def trajectory_fixture(times=None,window=(0.,.5),arm='single'):
    rng=np.random.default_rng(42)
    times=np.linspace(0,1,7,dtype=np.float32)[:-1] if times is None else np.asarray(times,dtype=np.float32)
    t,b,n=len(times),3,4
    active=set(selection_steps(times,*window).tolist())
    indices=np.tile(np.arange(b,dtype=np.int64),(t,1))
    if arm!='unguided':
        for i in active:indices[i]=[i%2,i%2,2]
    proposal=rng.normal(size=(t,b,n,3)).astype(np.float32)
    current=np.concatenate([rng.normal(size=(1,b,n,3)).astype(np.float32),
                            proposal[np.arange(t-1)[:,None],indices[:-1]]])
    roots=np.empty((t,b),dtype=np.int64);roots[0]=np.arange(b)
    for i in range(1,t):roots[i]=roots[i-1,indices[i-1]]
    atoms=rng.integers(0,4,(t,b,n),dtype=np.uint8)
    bonds=np.zeros((t,b,n,n),dtype=np.uint8)
    bonds[...,0,1]=bonds[...,1,0]=1
    steps=np.diff(np.r_[times,np.float32(1)]).astype(np.float32)
    data={'current_coords':current,'proposal_coords':proposal,
          'predicted_coords':rng.normal(size=proposal.shape).astype(np.float32),
          'mask':np.ones((t,b,n),dtype=np.uint8),'score_time':np.repeat(times[:,None],3,1),
          'state_time':times+steps,'step_size':steps,'selected_indices':indices,
          'offspring_count':np.stack([np.bincount(v,minlength=b) for v in indices]),
          'root_slot':roots,'parent_slot':np.concatenate([np.arange(b)[None],indices[:-1]]),
          'pic50_on':rng.uniform(6,9,(t,b)).astype(np.float32),
          'pic50_off':rng.uniform(6,7,(t,b)).astype(np.float32),
          'weight_on':np.full((t,b),1/b,dtype=np.float32),
          'weight_off':np.full((t,b),1/b,dtype=np.float32),
          'selection_probability':np.full((t,b),1/b,dtype=np.float32),
          'resampled':np.asarray([i in active and arm!='unguided' for i in range(t)],dtype=np.uint8),
          'predicted_atomics_probs':rng.uniform(0,1,(t,b,n,4)).astype(np.float32)}
    for rep in ('current','proposal','predicted'):
        data[rep+'_atomics']=atoms.copy()
        data[rep+'_bonds']=bonds.copy()
        data[rep+'_charges']=np.zeros_like(atoms)
    for key in ('atomics','bonds','charges'):
        data['current_'+key][1:]=data['proposal_'+key][np.arange(t-1)[:,None],indices[:-1]]
    terminal={'coords':proposal[-1]*2+10,'mask':data['mask'][-1],
              'atomics':data['predicted_atomics_probs'][-1],
              'affinity__pic50':data['pic50_on'][-1,:,None]}
    return times,data,terminal


def capture(path,*,codec='gzip_shuffle',times=None,window=(0.,.5),arm='single'):
    times,data,terminal=trajectory_fixture(times,window,arm)
    writer=SelectionLearningWriter(path,score_times=times,window=window,codec=codec,
        metadata={'arm':arm,'batch':0,'seed':42,'coord_scale':2.})
    for i in range(len(times)):writer.append(i,{k:v[i] for k,v in data.items()})
    manifest=writer.finish(terminal)
    return writer,data,terminal,manifest


@pytest.mark.parametrize('codec',['none','gzip_shuffle'])
def test_window_keeps_rejected_siblings_and_full_tail_without_geometry(tmp_path,codec):
    writer,data,terminal,m=capture(tmp_path,codec=codec)
    keep=np.asarray(m['selection_steps'])
    assert keep.tolist()==[0,1,2,3]
    assert m['last_selected_proposal_time']>m['window'][1]
    with TrajectoryPackage(tmp_path/'trajectory.h5') as p:
        for key in data:assert byte_equal(p.read(key),data[key][keep])
        # slot1 was rejected at step0; its coordinates and score still exist.
        assert p.read('offspring_count',0,1)==0
        assert byte_equal(p.read('proposal_coords',0,1),data['proposal_coords'][0,1])
        if codec=='none':assert_no_filters(p.file)
    with TrajectoryPackage(tmp_path/'lineage/trajectory.h5') as p:
        assert byte_equal(p.read('selected_indices'),data['selected_indices'])
        assert not any('coords' in k for k in p.keys)
    with TrajectoryPackage(tmp_path/'terminal.h5') as p:
        for key in terminal:assert byte_equal(p.read(key)[0],terminal[key])
    assert not list(tmp_path.rglob('*.npz')) and not list(tmp_path.rglob('*.pt.gz'))
    assert not list(tmp_path.rglob('trajectory.stages'))
    assert not list(tmp_path.rglob('*.h5.gz')) if codec=='none' else True
    table=pq.read_table(tmp_path/'selection_events.parquet').to_pandas()
    assert len(table)==12 and table.pic50_on.dtype==np.float32
    assert not any('terminal' in k for k in table.columns)
    assert verify_learning_batch(tmp_path)['verified']
    assert writer.finish(terminal)==m


def test_event_api_exposes_failed_candidate_and_separate_terminal_label(tmp_path):
    _,data,_,m=capture(tmp_path)
    record=read_scoring_event(tmp_path,0,1,terminal_outcome=True)
    assert record['observation']['offspring_count']==0
    assert record['terminal_outcome']['1']==[]
    assert record['array_dtypes']['proposal_coords']=='<f4'
    np.testing.assert_array_equal(record['observation']['proposal_coords'],data['proposal_coords'][0,1])
    with pytest.raises(ValueError):read_scoring_event(tmp_path,5)
    with pytest.raises(ValueError):read_scoring_event(tmp_path,0,4)
    scores=read_scoring_event(tmp_path,3,geometry=False)
    assert 'proposal_coords' not in scores['observation']
    assert scores['observation']['score_time'][0]==.5


def test_original_step_mapping_for_dynamic_nonzero_start(tmp_path):
    times=np.array([0,.15,.4,.6,.8,.92],dtype=np.float32)
    _,data,_,m=capture(tmp_path,times=times,window=(.1,.6))
    assert m['selection_steps']==[1,2,3]
    row=read_scoring_event(tmp_path,2,0)
    assert row['local_step']==1 and row['source_step']==2
    assert row['observation']['root_slot']==int(data['root_slot'][2,0])
    verify_learning_batch(tmp_path)


def test_partial_inference_never_becomes_complete_and_bad_selection_rejected(tmp_path):
    times,data,terminal=trajectory_fixture()
    w=SelectionLearningWriter(tmp_path,score_times=times,window=(0,.5),metadata={'arm':'single'})
    row={k:v[0] for k,v in data.items()}
    with pytest.raises(ValueError):w.append(0,{**row,'offspring_count':np.ones(3,dtype=int)})
    w.append(0,row)
    assert w.append(0,row) is None  # exact replay is idempotent
    with pytest.raises(ValueError):w.finish(terminal)
    assert read_json(w.manifest_path)['state']=='recording'
    with pytest.raises(ValueError):read_scoring_event(tmp_path,0)
    with pytest.raises(ValueError):SelectionLearningWriter(tmp_path,score_times=times,window=(0,.4),metadata={'arm':'single'})


def test_descendant_counts_match_explicit_backward_paths():
    _,data,_=trajectory_fixture()
    indices=data['selected_indices'];steps=[0,2,3]
    observed=terminal_descendants(indices,steps)
    for row,t in enumerate(steps):
        ancestor=np.arange(3)
        for i in range(len(indices)-1,t-1,-1):ancestor=indices[i,ancestor]
        np.testing.assert_array_equal(observed[row],np.bincount(ancestor,minlength=3))


def test_probability_capture_preserves_native_float32_and_skips_restart_dump(tmp_path):
    class Base:
        def __init__(self,path,arm,seed,batch,scale,steps,save=True):
            self.path=Path(path);self.arm=arm;self.seed=seed;self.batch=batch;self.scale=scale
            self.save=save;self.arr=defaultdict(list)
        def score(self,pred,onoff):
            for k in ('atomics','charges'):
                self.arr['predicted_'+k+'_probs_f16'].append(pred[k].astype(np.float16))
    class Trace(LearningTraceMixin,Base):pass
    trace=Trace(tmp_path,'single',42,0,2.,steps=6,score_grid=np.linspace(0,1,7,dtype=np.float32)[:-1],window=(0,.5))
    pred={'atomics':np.full((3,4,5),.3333334,dtype=np.float32),
          'charges':np.full((3,4,2),.10000003,dtype=np.float32)}
    trace.score(pred,{})
    assert trace.save is False and trace.anchor_steps==set()
    assert not any('f16' in k for k in trace.arr)
    assert byte_equal(trace.arr['predicted_atomics_probs'][0],pred['atomics'])
    assert not byte_equal(pred['atomics'],pred['atomics'].astype(np.float16).astype(np.float32))
    final=terminal_numeric_arrays({'coords':np.zeros((3,4,3),np.float32),'affinity':{'pic50':np.ones((3,1),np.float32)}})
    assert 'affinity__pic50' in final and final['coords'].dtype==np.float32


def config_fixture(tmp_path,off=False):
    flowr=tmp_path/'FLOWR with spaces'
    for name in ('flowr/models/fm_pocket.py','flowr/gen/generate_from_pdb_selective.py'):
        p=flowr/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture')
    inputs=tmp_path/'input';inputs.mkdir()
    roles={'target_protein':'protein_custom.pdb','target_ligand':'ligand_custom.sdf'}
    if off:roles.update(off_target_protein='other.pdb',off_target_ligand='other.sdf')
    for name in roles.values():(inputs/name).write_text('input fixture')
    atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'files':roles})
    checkpoint=tmp_path/'model.ckpt';checkpoint.write_bytes(b'checkpoint fixture')
    return SteerLearningConfig(str(flowr),str(checkpoint),str(inputs),str(tmp_path/'out'),
                              python_executable=sys.executable,samples=3,batch_size=3,steps=6)


def test_generic_inputs_dry_run_and_off_target_requirements(tmp_path):
    cfg=config_fixture(tmp_path)
    plan=launch(cfg,dry_run=True)
    assert plan['mode']=='selective_smc' and plan['gradient_guidance'] is False
    assert plan['off_target_is_target_alias'] is True and plan['archive_container']=='directory'
    assert cfg.storage_codec=='none'
    assert plan['command'][1:4]==['-u','-m','evomolsteer.generation.steer_worker']
    assert not Path(cfg.output_dataset).exists()
    with pytest.raises(ValueError):launch(replace(cfg,arms=('joint',)),dry_run=True)
    with pytest.raises(ValueError):launch(replace(cfg,window_end=1.2),dry_run=True)
    with pytest.raises(ValueError):launch(replace(cfg,output_dataset=cfg.input_dataset),dry_run=True)
    with pytest.raises(ValueError):launch(replace(cfg,samples=4),dry_run=True)
    manifest=Path(cfg.input_dataset)/'pocket_inputs.json'
    spec=read_json(manifest);spec['files'].update(off_target_protein='missing.pdb',off_target_ligand='missing.sdf')
    atomic_json(manifest,spec)
    with pytest.raises(FileNotFoundError):launch(cfg,dry_run=True)


def test_launcher_full_capture_contract_and_input_integrity(tmp_path,monkeypatch):
    cfg=config_fixture(tmp_path)
    input_hashes={p.name:digest(p) for p in Path(cfg.input_dataset).iterdir()}
    class Process:
        def __init__(self,command,**kwargs):
            assert kwargs['cwd']==str(Path(cfg.flowr_root).resolve())
            assert 'shell' not in kwargs
            root=Path(cfg.output_dataset);camp=root/'results'/cfg.campaign
            batch=camp/'single/batch_000'
            _,data,terminal,m=capture(batch)
            atomic_json(batch/'final_records.json',[{'slot':i,'build_success':i!=1,'failure_reason':'decode_failed' if i==1 else ''} for i in range(3)])
            atomic_json(batch/'COMPLETE.json',{'n':3})
            # Add the fake model's terminal records to its verified file list.
            m['files'].append({'path':'final_records.json','sha256':digest(batch/'final_records.json'),
                               'bytes':(batch/'final_records.json').stat().st_size})
            atomic_json(batch/'learning_manifest.json',m)
            atomic_json(camp/'COMPLETE.json',{'status':'complete','records':3})
            self.pid=123;self.returncode=0;self.stdout=iter(['CPU recording harness; not model inference\n'])
        def wait(self,timeout=None):return 0
        def poll(self):return 0
    monkeypatch.setattr(steer_launcher.subprocess,'Popen',Process)
    result=launch(cfg)
    assert result['state']=='complete' and result['verification']['records']==3
    assert result['off_target_is_target_alias'] and not list(Path(cfg.output_dataset).rglob('*.tar.gz'))
    from evomolsteer.trajectory_source import validate_input_bundle
    validate_input_bundle(cfg.output_dataset,{'campaign':cfg.campaign})
    for p in Path(cfg.input_dataset).iterdir():assert digest(p)==input_hashes[p.name]
    with pytest.raises(FileExistsError):launch(cfg)
    path=Path(cfg.output_dataset)/'inputs/target_protein.pdb';path.write_text('corruption')
    with pytest.raises(ValueError):validate_input_bundle(cfg.output_dataset,{'campaign':cfg.campaign})


def test_corrupt_completed_package_is_rejected(tmp_path):
    capture(tmp_path)
    with h5py.File(tmp_path/'trajectory.h5','r+') as f:
        f.attrs['untrusted_change']='must invalidate file checksum'
    with pytest.raises(ValueError):verify_learning_batch(tmp_path)
    with pytest.raises(ValueError):read_scoring_event(tmp_path,0)


def test_generic_catalog_uses_specified_inputs_and_event_frame(tmp_path):
    from evomolsteer.geometry import build_catalog
    root=tmp_path/'dataset';inputs=root/'inputs';inputs.mkdir(parents=True)
    (inputs/'target.pdb').write_text('ATOM      1  CA  GLY Z   7       1.000   2.000   3.000  1.00 20.00           C  \n')
    (inputs/'reference.sdf').write_text('Reference\nEvoMolSteer\n\n  1  0  0  0  0  0            999 V2000\n'+
                                       f'{1.:10.4f}{2.:10.4f}{3.:10.4f} C   0  0  0\nM  END\n$$$$\n')
    roles={'target_protein':'target.pdb','target_ligand':'reference.sdf',
           'off_target_protein':'target.pdb','off_target_ligand':'reference.sdf'}
    atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'files':roles,'off_target_is_target_alias':True})
    cfg={'pocket_radius_A':8.,'softmin_temperature_A':.25,'contact_midpoint_A':4.5,'contact_width_A':.5}
    catalog=build_catalog(root,cfg,{'C':0})
    assert list(catalog['regions'])==['target:Z:GLY7']
    batch=root/'results/c/single/batch_000';capture(batch)
    atomic_json(batch.parents[1]/'frame_batch_000.json',{'target_com':[[1.,2.,3.]]*3})
    atomic_json(batch.parents[1]/'config.json',{'atom_vocabulary':{'C':0},'input_files':roles})
    event=read_scoring_event(batch,0,1)
    assert event['frame']['target_com']==[1.,2.,3.] and event['vocabularies']['atom_vocabulary']=={'C':0}


def test_portable_input_hashes_and_historical_fallback(tmp_path):
    from evomolsteer.trajectory_source import pocket_input_path,pocket_input_hashes
    root=tmp_path/'portable';inputs=root/'inputs';inputs.mkdir(parents=True)
    for name in ('custom.pdb','custom.sdf'):(inputs/name).write_text(name)
    roles=dict(target_protein='custom.pdb',target_ligand='custom.sdf',
               off_target_protein='custom.pdb',off_target_ligand='custom.sdf')
    atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'files':roles})
    assert pocket_input_path(root,'target_protein')==inputs/'custom.pdb'
    assert pocket_input_hashes(root)=={name:digest(inputs/name) for name in ('custom.pdb','custom.sdf')}
    outside=root/'outside.pdb';outside.write_text('outside')
    roles['target_protein']='../outside.pdb'
    atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'files':roles})
    with pytest.raises(ValueError):pocket_input_path(root,'target_protein')
    historical=tmp_path/'historical';(historical/'inputs').mkdir(parents=True)
    from evomolsteer.generation.launcher import INPUT_FILES
    for name in INPUT_FILES:(historical/'inputs'/name).write_text(name)
    assert set(pocket_input_hashes(historical))==set(INPUT_FILES)


def test_coordinate_context_tampering_rejects_event_and_batch(tmp_path):
    root=tmp_path/'results/c';batch=root/'single/batch_000'
    _,_,_,manifest=capture(batch)
    context=root/'frame_batch_000.json'
    atomic_json(context,{'target_com':[[1.,2.,3.]]*3})
    manifest['campaign_context']=[{'path':context.name,'sha256':digest(context),'bytes':context.stat().st_size}]
    atomic_json(batch/'learning_manifest.json',manifest)
    assert read_scoring_event(batch,0,1)['frame']['target_com']==[1.,2.,3.]
    atomic_json(context,{'target_com':[[99.,2.,3.]]*3})
    with pytest.raises(ValueError):read_scoring_event(batch,0,1)
    with pytest.raises(ValueError):verify_learning_batch(batch)


def test_existing_whole_window_ingest_accepts_portable_window_dataset(tmp_path):
    from evomolsteer.ingest import ingest
    root=tmp_path/'dataset';inputs=root/'inputs';inputs.mkdir(parents=True)
    (inputs/'target.pdb').write_text('ATOM      1  CA  GLY Z   7       1.000   2.000   3.000  1.00 20.00           C  \n')
    (inputs/'reference.sdf').write_text('Reference\nEvoMolSteer\n\n  1  0  0  0  0  0            999 V2000\n'+
                                       f'{1.:10.4f}{2.:10.4f}{3.:10.4f} C   0  0  0\nM  END\n$$$$\n')
    atomic_json(inputs/'pocket_inputs.json',{'format':INPUT_FORMAT,'off_target_is_target_alias':True,
        'files':{'target_protein':'target.pdb','target_ligand':'reference.sdf',
                 'off_target_protein':'target.pdb','off_target_ligand':'reference.sdf'}})
    campaign=root/'results/example'
    for arm in ('single','unguided'):
        batch=campaign/arm/'batch_000';capture(batch,codec='none',arm=arm)
        atomic_json(batch/'COMPLETE.json',{'n':3})
    atomic_json(campaign/'frame_batch_000.json',{'target_com':[[1.,2.,3.]]*3})
    atomic_json(campaign/'config.json',{'coord_scale':2.,'atom_vocabulary':{'C':0,'N':1,'O':2,'S':3}})
    cfg=read_json(Path(__file__).resolve().parents[1]/'configs/default.json')
    cfg.update(campaign='example',discovery_batches=[0],validation_batches=[],heldout_batches=[],
               feature_schema='geometry',feature_pockets=['target'],ingest_workers=1)
    output=tmp_path/'analysis'
    audits=ingest(root,output,cfg)
    frame=pq.read_table(output/'features.parquet').to_pandas()
    assert len(audits)==2 and len(frame)==4*3*2*2
    assert set(frame.stage)=={0} and frame.score_time.max()==.5
    assert not any('terminal' in key for key in frame.columns)
    assert read_json(output/'selection_scope.json')['steps']==[0,1,2,3]
