from dataclasses import replace
from pathlib import Path
import sys

import pytest

from evomolsteer.io import digest,read_json
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.storage.target_archive import restore_target
from evomolsteer.generation import steer_campaign,steer_launcher
from evomolsteer.generation.steer_campaign import TargetCampaignConfig,run_campaign
from evomolsteer.generation.target_catalog import discover_targets,FORMAT,target_key
from evomolsteer.generation.steer_launcher import SteerLearningConfig,launch
from test_steer_learning import capture


def config(tmp_path,n=12):
    root=tmp_path/'inputs';root.mkdir()
    for i in range(n):
        p=root/f'target_{i:03d}';p.mkdir()
        (p/'reference_pocket10.pdb').write_text(f'protein {i}')
        (p/'reference.sdf').write_text(f'ligand {i}')
    flowr=tmp_path/'flowr'
    for name in ('flowr/models/fm_pocket.py','flowr/gen/generate_from_pdb_selective.py'):
        p=flowr/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# CPU fixture only')
    checkpoint=tmp_path/'model.ckpt';checkpoint.write_bytes(b'model fixture')
    return TargetCampaignConfig(str(flowr),str(checkpoint),str(root),str(tmp_path/'output'),
                               python_executable=sys.executable,expected_targets=n,samples=3,batch_size=3,steps=6)


def fake_generation(monkeypatch,*,fail_once=None):
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):
            request=read_json(command[-1]);root=Path(request['output_dataset']);camp=root/'results'/request['campaign']
            batch=camp/'single/batch_000';_,_,_,m=capture(batch,codec=request['storage_codec'])
            rows=[{'slot':i,'build_success':i!=1,'failure_reason':'decode_failed' if i==1 else '',
                   'smiles':'C' if i!=1 else '', 'pic50_on_upstream':8+i/10,'pic50_on_rescore':8.1+i/10} for i in range(3)]
            atomic_json(batch/'final_records.json',rows);atomic_json(camp/'final_records.json',rows)
            atomic_json(batch/'COMPLETE.json',{'n':3})
            m['files'].append({'path':'final_records.json','sha256':digest(batch/'final_records.json'),'bytes':(batch/'final_records.json').stat().st_size})
            atomic_json(batch/'learning_manifest.json',m)
            atomic_json(camp/'COMPLETE.json',{'status':'complete','records':3})
            self.pid=42;self.returncode=0;self.stdout=iter(['CPU capture fixture; not FLOWR inference\n'])
        def wait(self,timeout=None):return 0
        def poll(self):return 0
    monkeypatch.setattr(steer_launcher.subprocess,'Popen',Process)
    failed=set()
    def execute(command,log,env):
        assert Path(command[2]).name=='generate_steer_learning.py' and Path(command[2]).parent.name=='scripts'
        cfg=SteerLearningConfig(**read_json(command[-1]));calls.append(cfg)
        assert tuple(cfg.arms)==('single',) and cfg.seed==42
        if fail_once and fail_once in str(cfg.output_dataset) and fail_once not in failed:
            failed.add(fail_once);raise RuntimeError('Simulated unavailable GPU')
        launch(cfg);Path(log).write_text('CPU capture fixture; not FLOWR inference')
    monkeypatch.setattr(steer_campaign,'run_target_job',execute)
    return calls


def test_crossdocked_discovery_is_exact_and_preserves_explicit_identity(tmp_path):
    cfg=config(tmp_path,2);rows=discover_targets(cfg.input_dataset,expected_targets=2)
    assert [r['target_id'] for r in rows]==['target_000/reference','target_001/reference']
    assert rows[0]['files']['target_ligand']['path']=='target_000/reference.sdf'
    with pytest.raises(ValueError):discover_targets(cfg.input_dataset,expected_targets=100)
    p=Path(cfg.input_dataset)/'target_000/other_pocket10.pdb';p.write_text('protein');p.with_name('other.sdf').write_text('ligand')
    rows=discover_targets(cfg.input_dataset,expected_targets=3)
    assert [r['target_id'] for r in rows]==['target_000/other','target_000/reference','target_001/reference']
    manifest=tmp_path/'targets.json';atomic_json(manifest,{'format':FORMAT,'targets':[
        {'target_id':'chosen / A','target_protein':'target_000/reference_pocket10.pdb','target_ligand':'target_000/reference.sdf'}]})
    rows=discover_targets(cfg.input_dataset,manifest=manifest)
    assert rows[0]['key']==target_key('chosen / A') and '/' not in rows[0]['key']
    duplicated=read_json(manifest);duplicated['targets']*=2;atomic_json(manifest,duplicated)
    with pytest.raises(ValueError,match='duplicate'):discover_targets(cfg.input_dataset,manifest=manifest)
    atomic_json(manifest,{'format':FORMAT,'targets':[duplicated['targets'][0]]})
    data=read_json(manifest);data['targets'][0]['target_ligand']='../model.ckpt';atomic_json(manifest,data)
    with pytest.raises(ValueError):discover_targets(cfg.input_dataset,manifest=manifest)


def test_twelve_targets_create_ten_plus_two_archives_and_are_restorable(tmp_path,monkeypatch):
    cfg=config(tmp_path);calls=fake_generation(monkeypatch)
    input_hashes={str(p):digest(p) for p in Path(cfg.input_dataset).rglob('*') if p.is_file()}
    result=run_campaign(cfg)
    assert len(calls)==12 and result['status']=='complete'
    assert result['totals']['successful_targets']==12
    assert [len(r['target_ids']) for r in result['archives']]==[10,2]
    assert all(r['summary']['failed_slots']==1 for r in result['targets'].values())
    assert not list((Path(cfg.output_dataset)/'targets').iterdir())
    item=result['archives'][0];output=tmp_path/'restored'
    restore_target(Path(cfg.output_dataset)/item['path'],'target_000/reference',output,item['metadata'])
    assert read_json(output/'results/single_w050/final_records.json')[1]['build_success'] is False
    for p,h in input_hashes.items():assert digest(p)==h
    run_campaign(cfg)
    assert len(calls)==12  # archived targets are never generated again


def test_multiple_pockets_in_one_protein_folder_run_independent_targets(tmp_path,monkeypatch):
    cfg=config(tmp_path,1)
    p=Path(cfg.input_dataset)/'target_000/other_pocket10.pdb';p.write_text('another pocket')
    p.with_name('other.sdf').write_text('another ligand')
    cfg=replace(cfg,expected_targets=2,compress=False);calls=fake_generation(monkeypatch)
    result=run_campaign(cfg)
    assert result['status']=='complete' and len(calls)==2
    assert set(result['targets'])=={'target_000/other','target_000/reference'}
    proteins={read_json(c.input_manifest)['files']['target_protein'] for c in calls}
    assert proteins=={str(p),str(p.with_name('reference_pocket10.pdb'))}
    assert all(tuple(c.arms)==('single',) for c in calls)


def test_resume_partial_group_and_disabled_compression(tmp_path,monkeypatch):
    cfg=config(tmp_path,3);cfg=replace(cfg,archive_every=2);calls=fake_generation(monkeypatch)
    first=run_campaign(cfg,max_targets=1)
    assert first['status']=='pending' and not first['archives'] and len(calls)==1
    done=run_campaign(cfg)
    assert done['status']=='complete' and len(calls)==3
    assert [len(r['target_ids']) for r in done['archives']]==[2,1]
    other=replace(cfg,output_dataset=str(tmp_path/'no_compression'),compress=False)
    result=run_campaign(other)
    assert not result['archives'] and result['status']=='complete'
    assert all((Path(other.output_dataset)/r['dataset']/'learning_dataset_manifest.json').exists() for r in result['targets'].values())


def test_failed_task_requires_explicit_marker_reset_and_keeps_failure_log(tmp_path,monkeypatch):
    cfg=config(tmp_path,2);calls=fake_generation(monkeypatch,fail_once='target_000')
    failed=run_campaign(cfg)
    assert failed['status']=='partial_failure' and failed['totals']['failed_targets']==1
    skipped=run_campaign(cfg,retry_failed=True)
    assert skipped['worker']['no_available_targets'] and len(calls)==2
    error=Path(cfg.output_dataset)/'generation_progress'/(target_key('target_000/reference')+'.error')
    assert 'Simulated unavailable GPU' in error.read_text() and 'Traceback' in error.read_text()
    error.rename(error.with_suffix('.error.reviewed'))
    result=run_campaign(cfg)
    assert result['status']=='complete' and len(calls)==3
    attempts=result['targets']['target_000/reference']['attempts']
    assert len(attempts)==2 and attempts[0]['state']=='failed' and attempts[1]['state']=='complete'
    assert '__attempt_002' in result['targets']['target_000/reference']['dataset']


def test_dry_run_and_input_change_do_not_launch_or_overwrite(tmp_path,monkeypatch):
    cfg=config(tmp_path,2);calls=fake_generation(monkeypatch)
    plan=run_campaign(cfg,dry_run=True)
    assert plan['archive_every_targets']==10 and plan['arms']==['single']
    assert not Path(cfg.output_dataset).exists() and not calls
    run_campaign(cfg,max_targets=1)
    (Path(cfg.input_dataset)/'target_001/reference.sdf').write_text('changed after freeze')
    with pytest.raises(ValueError):run_campaign(cfg)
    assert len(calls)==1
    for change in (dict(archive_every=0),dict(expected_targets=100),dict(samples=4),dict(campaign='../bad'),dict(target_ids=('missing',))):
        with pytest.raises(ValueError):run_campaign(replace(cfg,**change),dry_run=True)


def test_real_child_bridge_propagates_failure_and_keeps_log(tmp_path):
    log=tmp_path/'child log with spaces.txt'
    with pytest.raises(RuntimeError,match='exited with 3'):
        steer_campaign.run_target_job([sys.executable,'-u','-c',"print('CPU bridge fixture'); raise SystemExit(3)"],log,{})
    assert log.read_text().strip()=='CPU bridge fixture'
