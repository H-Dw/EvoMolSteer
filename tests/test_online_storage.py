from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
from evomolsteer.io import read_json,digest
from evomolsteer.storage import lifecycle,streaming,trajectory
from evomolsteer.storage.arrays import byte_equal
from evomolsteer.storage.streaming import StepTrajectoryWriter
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.storage.transactions import exclusive_lock
from evomolsteer.generation.launcher import FlowrRunConfig,launch,INPUT_FILES
from test_storage_lossless import make_trajectory


def source_npz(root):
    root.mkdir(parents=True,exist_ok=True)
    data=make_trajectory();path=root/'trajectory.npz';np.savez_compressed(path,**data)
    return path,data


def test_each_step_is_durable_before_finish_and_final_cleanup(tmp_path):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    for i in range(4):
        row={k:v[i] for k,v in data.items()}
        record=w.append(i,row)
        with TrajectoryPackage(tmp_path/record['file']) as p:
            assert all(byte_equal(p.read(k),v[i:i+1]) for k,v in data.items())
        assert len(read_json(w.manifest_path)['steps'])==i+1
        assert not w.final_path.exists()
        assert w.append(i,row)==record
    with pytest.raises(ValueError):w.append(0,{**{k:v[0] for k,v in data.items()},'mask':data['mask'][0]*0})
    report=w.finalize()
    with TrajectoryPackage(w.final_path) as p:assert p.verify(data)==len(data)
    assert not w.stages.exists() and not list(tmp_path.rglob('*.npz'))
    assert w.finalize()==report


def test_publication_failure_never_deletes_source_and_retry_is_safe(tmp_path,monkeypatch):
    source,data=source_npz(tmp_path/'input');out=tmp_path/'output/trajectory.h5'
    original=TrajectoryPackage.verify
    def fail(*args,**kwargs):raise RuntimeError('injected failed verification')
    monkeypatch.setattr(TrajectoryPackage,'verify',fail)
    with pytest.raises(RuntimeError):lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)
    assert source.exists() and not out.exists()
    monkeypatch.setattr(TrajectoryPackage,'verify',original)
    record=lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)
    assert record['source_deleted'] and not source.exists()
    with TrajectoryPackage(out) as p:p.verify(data)


def test_conversion_recovers_after_unlink_before_receipt_update(tmp_path,monkeypatch):
    source,_=source_npz(tmp_path/'input');out=tmp_path/'output/trajectory.h5'
    original=lifecycle.atomic_json
    def fail_complete(path,value):
        if value.get('state')=='complete':raise RuntimeError('injected journal interruption')
        return original(path,value)
    monkeypatch.setattr(lifecycle,'atomic_json',fail_complete)
    with pytest.raises(RuntimeError):lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)
    assert not source.exists() and out.exists()
    monkeypatch.setattr(lifecycle,'atomic_json',original)
    assert lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)['state']=='complete'


def test_source_changed_after_conversion_is_not_deleted(tmp_path,monkeypatch):
    source,data=source_npz(tmp_path/'input');out=tmp_path/'output/trajectory.h5'
    original=lifecycle.pack_trajectory
    def change_source(*args,**kwargs):
        result=original(*args,**kwargs)
        data['pic50_on']=data['pic50_on']+1
        np.savez_compressed(source,**data)
        return result
    monkeypatch.setattr(lifecycle,'pack_trajectory',change_source)
    with pytest.raises(ValueError):lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)
    assert source.exists()


def test_outside_source_and_unowned_partial_are_never_removed(tmp_path):
    source,_=source_npz(tmp_path/'outside');root=tmp_path/'declared';root.mkdir()
    out=root/'trajectory.h5';partial=out.with_suffix('.h5.partial');partial.write_bytes(b'not ours')
    with pytest.raises(ValueError):lifecycle.convert_trajectory(source,out,input_dataset=root,delete_source=True)
    with pytest.raises(FileExistsError):lifecycle.convert_trajectory(source,out,input_dataset=source.parent,delete_source=True)
    assert source.exists() and partial.read_bytes()==b'not ours'


def test_partial_generation_and_consolidation_failure_keep_stages(tmp_path,monkeypatch):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    with pytest.raises(ValueError):w.append(1,{k:v[1] for k,v in data.items()})
    w.append(0,{k:v[0] for k,v in data.items()})
    with pytest.raises(ValueError):w.finalize()
    for i in range(1,4):w.append(i,{k:v[i] for k,v in data.items()})
    def fail(*args,**kwargs):raise RuntimeError('injected consolidation failure')
    monkeypatch.setattr(streaming,'pack_arrays',fail)
    with pytest.raises(RuntimeError):w.finalize()
    assert len(list(w.stages.glob('*.h5.gz')))==4 and not w.final_path.exists()


def test_cleanup_resumes_after_interruption(tmp_path,monkeypatch):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    for i in range(4):w.append(i,{k:v[i] for k,v in data.items()})
    original=streaming.unlink_verified;calls=[]
    def interrupt(path,*args,**kwargs):
        if calls:raise RuntimeError('interrupted stage cleanup')
        calls.append(path);return original(path,*args,**kwargs)
    monkeypatch.setattr(streaming,'unlink_verified',interrupt)
    with pytest.raises(RuntimeError):w.finalize()
    assert read_json(w.manifest_path)['state']=='consolidated' and len(list(w.stages.glob('*.h5.gz')))==3
    monkeypatch.setattr(streaming,'unlink_verified',original)
    w.finalize()
    assert not w.stages.exists()
    with TrajectoryPackage(w.final_path) as p:p.verify(data)


def test_lock_excludes_another_writer_and_releases(tmp_path):
    lock=tmp_path/'writer.lock'
    with exclusive_lock(lock):
        with pytest.raises(RuntimeError):
            with exclusive_lock(lock):pass
    with exclusive_lock(lock):pass


def test_dataset_conversion_copies_context_and_retains_nontrajectory(tmp_path):
    root=tmp_path/'in';source,data=source_npz(root/'results/c/joint/batch_000')
    (source.parent/'COMPLETE.json').write_text('{}')
    (root/'results/c/COMPLETE.json').write_text('{}')
    (source.parent/'restart.pt.gz').write_bytes(b'checkpoint must survive')
    (root/'inputs').mkdir();(root/'inputs/context.json').write_text('{}')
    unrelated=root/'inputs/trajectory.npz';unrelated.write_bytes(b'not a recorded trajectory')
    out=tmp_path/'out'
    report=lifecycle.convert_dataset(root,out,delete_source=True)
    assert report['complete'] and not source.exists()
    assert (source.parent/'restart.pt.gz').exists() and (out/'inputs/context.json').exists()
    assert unrelated.exists() and (out/'inputs/trajectory.npz').read_bytes()==unrelated.read_bytes()
    with TrajectoryPackage((out/source.relative_to(root)).with_suffix('.h5')) as p:p.verify(data)
    assert lifecycle.convert_dataset(root,out,delete_source=True)['complete']
    from evomolsteer.io import validate_checksums
    from evomolsteer.trajectory_source import validate_input_bundle
    assert validate_checksums(out)>0
    with pytest.raises(ValueError,match='retired'):validate_input_bundle(root,{'campaign':'c'})
    validate_input_bundle(out,{'campaign':'c'})


def test_stage_envelope_failure_preserves_verified_container(tmp_path,monkeypatch):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    original=streaming.gzip.compress
    monkeypatch.setattr(streaming.gzip,'compress',lambda *args,**kwargs:b'broken envelope')
    with pytest.raises(OSError):w.append(0,{k:v[0] for k,v in data.items()})
    assert not read_json(w.manifest_path)['steps']
    assert (w.stages/'step_000000.h5').exists()
    monkeypatch.setattr(streaming.gzip,'compress',original)
    record=w.append(0,{k:v[0] for k,v in data.items()})
    assert not (w.stages/'step_000000.h5').exists()
    with TrajectoryPackage(w.root/record['file']) as p:p.verify({k:v[:1] for k,v in data.items()})


def test_stage_publication_recovers_before_journal_commit(tmp_path,monkeypatch):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    original=streaming.atomic_json
    def interrupted(path,value):
        if value.get('steps'):raise RuntimeError('interrupted journal commit')
        return original(path,value)
    monkeypatch.setattr(streaming,'atomic_json',interrupted)
    with pytest.raises(RuntimeError):w.append(0,{k:v[0] for k,v in data.items()})
    published=w.stages/'step_000000.h5.gz';before=digest(published)
    assert not read_json(w.manifest_path)['steps']
    monkeypatch.setattr(streaming,'atomic_json',original)
    assert w.append(0,{k:v[0] for k,v in data.items()})['sha256']==before


def test_corrupted_stage_blocks_consolidation_and_retirement(tmp_path):
    data=make_trajectory();w=StepTrajectoryWriter(tmp_path,expected_steps=4)
    for i in range(4):w.append(i,{k:v[i] for k,v in data.items()})
    stage=w.stages/'step_000001.h5.gz'
    stage.write_bytes(stage.read_bytes()+b'modified after commit')
    with pytest.raises(ValueError,match='checksum'):w.finalize()
    assert len(list(w.stages.glob('*.h5.gz')))==4 and not w.final_path.exists()


def test_dataset_refuses_open_batches_without_deleting(tmp_path):
    root=tmp_path/'in';source,_=source_npz(root/'results/c/joint/batch_000')
    with pytest.raises(ValueError):lifecycle.convert_dataset(root,tmp_path/'out',delete_source=True)
    assert source.exists() and not (tmp_path/'out').exists()


def test_launcher_uses_explicit_checkout_and_runtime_without_shell(tmp_path):
    flowr=tmp_path/'FLOWR path with spaces'
    for name in ['flowr/models/fm_pocket.py','flowr/gen/generate_from_pdb_selective.py']:
        p=flowr/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture')
    inputs=tmp_path/'inputs';inputs.mkdir()
    for name in INPUT_FILES:(inputs/name).write_text('fixture')
    checkpoint=tmp_path/'model.ckpt';checkpoint.write_bytes(b'fixture')
    cfg=FlowrRunConfig(str(flowr),str(checkpoint),str(inputs),str(tmp_path/'output'),python_executable=sys.executable)
    plan=launch(cfg,dry_run=True)
    assert plan['config']['flowr_root']==str(flowr.resolve()) and plan['mode']=='selective_smc'
    assert plan['command'][1:4]==['-u','-m','evomolsteer.generation.worker']
    assert not (tmp_path/'output').exists()
    with pytest.raises(ValueError):launch(replace(cfg,campaign='../escape'),dry_run=True)
    with pytest.raises(ValueError):launch(replace(cfg,window_end=2),dry_run=True)
