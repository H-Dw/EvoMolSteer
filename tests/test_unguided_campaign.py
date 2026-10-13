"""Unguided-only collection contracts, frozen reference identity and retirement."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evomolsteer.io import digest,read_json
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.storage.target_archive import restore_target
from evomolsteer.generation import steer_campaign,steer_launcher
from evomolsteer.generation.steer_campaign import run_campaign
from evomolsteer.generation.steer_launcher import SteerLearningConfig,launch,verify_dataset
from evomolsteer.generation.steer_trace import SteerLearningExtension
from evomolsteer.generation.instrumentation import instrument_source
from test_target_campaign import config
from test_steer_learning import capture


def fixture_worker(monkeypatch):
    calls=[]
    class Process:
        def __init__(self,command,**kwargs):
            request=read_json(command[-1]);root=Path(request['output_dataset']);camp=root/'results'/request['campaign']
            arm=request['arms'][0];batch=camp/arm/'batch_000'
            _,_,_,manifest=capture(batch,codec=request['storage_codec'],arm=arm)
            rows=[{'arm':arm,'slot':i,'build_success':i!=1,'failure_reason':'fixture_valence_failure' if i==1 else '',
                   'smiles':'C' if i!=1 else '', 'pic50_on_upstream':8.,'pic50_on_rescore':8.} for i in range(3)]
            atomic_json(batch/'final_records.json',rows);atomic_json(camp/'final_records.json',rows)
            atomic_json(batch/'COMPLETE.json',{'n':3})
            manifest['files'].append({'path':'final_records.json','sha256':digest(batch/'final_records.json'),'bytes':(batch/'final_records.json').stat().st_size})
            atomic_json(batch/'learning_manifest.json',manifest)
            atomic_json(camp/'COMPLETE.json',{'status':'complete','records':3})
            self.pid=42;self.stdout=iter(['CPU fixture, not FLOWR inference\n'])
        def wait(self,timeout=None):return 0
        def poll(self):return 0
    monkeypatch.setattr(steer_launcher.subprocess,'Popen',Process)
    def execute(command,log,environment):
        cfg=SteerLearningConfig(**read_json(command[-1]));calls.append(cfg)
        launch(cfg);Path(log).write_text('CPU fixture, not FLOWR inference')
    monkeypatch.setattr(steer_campaign,'run_target_job',execute)
    return calls


def test_unguided_only_twelve_targets_archive_ten_plus_two_preserving_all_slots(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path),arm='unguided',campaign='unguided_w050')
    calls=fixture_worker(monkeypatch)
    result=run_campaign(cfg)
    assert result['status']=='complete' and len(calls)==12
    assert all(tuple(c.arms)==('unguided',) for c in calls)
    assert [len(a['target_ids']) for a in result['archives']]==[10,2]
    assert not list((Path(cfg.output_dataset)/'targets').iterdir())
    item=result['archives'][0];restored=tmp_path/'restored'
    restore_target(Path(cfg.output_dataset)/item['path'],'target_000/reference',restored,item['metadata'])
    verification=verify_dataset(restored,cfg.campaign)
    assert verification['records']==3 and verification['batches'][0]['resampling_events']==0
    assert read_json(restored/'results/unguided_w050/final_records.json')[1]['build_success'] is False
    manifest=read_json(restored/'learning_dataset_manifest.json')
    assert manifest['particle_resampling'] is False and manifest['window_role']=='observation_only'
    run_campaign(cfg)
    assert len(calls)==12


def test_reference_mismatch_fails_before_inference_or_target_claim(tmp_path,monkeypatch):
    cfg=config(tmp_path,2);calls=fixture_worker(monkeypatch)
    run_campaign(cfg)
    baseline=replace(cfg,arm='unguided',campaign='unguided_w050',output_dataset=str(tmp_path/'baseline'),reference_campaign=cfg.output_dataset)
    plan=run_campaign(baseline,dry_run=True)
    assert plan['arms']==['unguided'] and plan['reference_comparison']['targets']==2
    assert not Path(baseline.output_dataset).exists()
    for changes in ({'samples':6},{'seed':43},{'window_end':.4},{'target_ids':('target_000/reference',)}):
        with pytest.raises(ValueError,match='differs|differ'):run_campaign(replace(baseline,**changes))
    assert len(calls)==2 and not Path(baseline.output_dataset).exists()
    result=run_campaign(baseline)
    assert result['status']=='complete' and len(calls)==4
    assert read_json(Path(baseline.output_dataset)/'reference_comparison.json')['targets']==2
    assert run_campaign(baseline)['status']=='complete' and len(calls)==4
    Path(baseline.checkpoint).write_bytes(b'different model')
    with pytest.raises(ValueError,match='Checkpoint differs'):run_campaign(baseline)


def test_unguided_verifier_rejects_a_resampled_trajectory(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path,1),arm='unguided',campaign='unguided_w050',compress=False)
    fixture_worker(monkeypatch);result=run_campaign(cfg)
    root=Path(cfg.output_dataset)/next(iter(result['targets'].values()))['dataset']
    with pytest.raises(ValueError,match='Missing or unexpected batches'):verify_dataset(root,cfg.campaign,arms=('single',))
    # A valid SMC archive cannot masquerade as an unguided archive, even if its
    # mode label is changed and all individual file checksums remain valid.
    other=tmp_path/'swapped';batch=other/'results/c/unguided/batch_000'
    _,_,_,m=capture(batch,arm='single');m['metadata']['arm']='unguided'
    atomic_json(batch/'learning_manifest.json',m)
    atomic_json(batch/'final_records.json',[{'slot':i} for i in range(3)])
    atomic_json(batch/'COMPLETE.json',{'n':3})
    atomic_json(other/'results/c/COMPLETE.json',{'status':'complete'})
    atomic_json(other/'results/c/generation_request.json',{'arms':['unguided'],'samples':3,'batch_size':3})
    with pytest.raises(ValueError,match='resampled'):verify_dataset(other,'c')


def test_native_unguided_unlock_only_removes_assertion_and_recording_calls():
    source=(Path(__file__).parent/'fixtures/upstream_generate_selective.py.txt').read_text()
    changed=instrument_source(source,allow_unguided=True)
    stripped=''.join(line for line in changed.splitlines(True) if 'self._lineage.' not in line)
    assert stripped==source.replace('    assert apply_guidance\n','',1)
    # Preserve the actual native selection condition; it evaluates False.
    assert 'apply_guidance\n' in changed
    ext=SteerLearningExtension(SimpleNamespace(arms=('unguided',),capture_bond_probabilities=False),{})
    ext.model=SimpleNamespace(_lineage=SimpleNamespace(arm='unguided'))
    assert ext.generation_kwargs()=={'apply_guidance':False}
    ext.model._lineage.arm='single'
    assert ext.generation_kwargs()=={'apply_guidance':True}
    ext.opt=SimpleNamespace(window_start=0.,window=.5)
    config={};ext.amend_config(config)
    assert config['selection_scope']['actual_selection_window'] is None
    assert config['selection_scope']['window_role']=='observation_only'


def test_unguided_entry_cannot_accidentally_launch_a_steer_arm(monkeypatch):
    from evomolsteer.generation.unguided_campaign import main
    seen=[]
    def no_gpu(cfg,**kwargs):
        seen.append(cfg);return {'status':'complete'}
    monkeypatch.setattr(steer_campaign,'run_campaign',no_gpu)
    main(['--dry-run'])
    assert seen[-1].arm=='unguided' and seen[-1].samples==100 and seen[-1].archive_every==10
    with pytest.raises(SystemExit):main(['--dry-run','--arm','single'])
    assert len(seen)==1
