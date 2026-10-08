"""Real processes exercise claims, simultaneous inference and archive merges.

The generation backend is a CPU capture fixture, never a FLOWR model run.
"""
from dataclasses import asdict,replace
import multiprocessing as mp
from pathlib import Path
import traceback

import pytest

from evomolsteer.io import read_json
from evomolsteer.generation import steer_campaign,target_archiving
from evomolsteer.generation.steer_campaign import TargetCampaignConfig,run_campaign
from evomolsteer.generation.target_catalog import discover_targets,target_key
from evomolsteer.generation.target_progress import TargetProgress
from evomolsteer.storage.target_archive import verify_target_archive
from evomolsteer.storage.transactions import atomic_json
from test_target_campaign import config,fake_generation


def _marker_worker(output,catalog,barrier,queue):
    progress=TargetProgress(output,catalog)
    barrier.wait(timeout=30)
    claim=progress.claim(catalog[0]['key'])
    if claim:
        assert progress.path(claim.key,'running').stat().st_size==0
        progress.finish(claim)
    queue.put(bool(claim))


def _campaign_worker(values,barrier,queue):
    patch=pytest.MonkeyPatch()
    try:
        fake_generation(patch);backend=steer_campaign.run_target_job;first=True
        def execute(command,log,env):
            nonlocal first
            request=read_json(command[-1]);inputs=read_json(request['input_manifest'])
            queue.put(('start',inputs['files']['target_protein']))
            # Both workers must enter different target jobs simultaneously.
            # Holding a global inference lock would fail this barrier.
            if first:
                first=False;barrier.wait(timeout=60)
            backend(command,log,env)
        patch.setattr(steer_campaign,'run_target_job',execute)
        run_campaign(TargetCampaignConfig(**values))
        queue.put(('worker_done',None))
    except BaseException:
        queue.put(('failure',traceback.format_exc()))
        raise
    finally:patch.undo()


def _join(processes):
    for p in processes:p.join(timeout=120)
    try:assert all(not p.is_alive() and p.exitcode==0 for p in processes)
    finally:
        for p in processes:
            if p.is_alive():p.terminate();p.join(timeout=10)


def test_four_processes_claim_one_target_exactly_once(tmp_path):
    output=tmp_path/'out';output.mkdir();catalog=[{'target_id':'A','key':'A'}]
    ctx=mp.get_context('spawn');queue=ctx.Queue();barrier=ctx.Barrier(4)
    processes=[ctx.Process(target=_marker_worker,args=(str(output),catalog,barrier,queue)) for _ in range(4)]
    for p in processes:p.start()
    _join(processes)
    assert sum(queue.get(timeout=10) for _ in processes)==1
    assert (output/'generation_progress/A.finished').stat().st_size==0
    assert not (output/'generation_progress/A.running').exists()


def test_two_campaign_processes_generate_once_and_merge_archives(tmp_path):
    cfg=replace(config(tmp_path,4),archive_every=2)
    ctx=mp.get_context('spawn');queue=ctx.Queue();barrier=ctx.Barrier(2)
    processes=[ctx.Process(target=_campaign_worker,args=(asdict(cfg),barrier,queue)) for _ in range(2)]
    for p in processes:p.start()
    _join(processes)
    messages=[queue.get(timeout=10) for _ in range(6)]  # four jobs + two workers
    assert not any(kind=='failure' for kind,_ in messages)
    started=[value for kind,value in messages if kind=='start']
    assert len(started)==len(set(started))==4
    root=Path(cfg.output_dataset);state=read_json(root/'campaign_state.json')
    assert state['status']=='complete' and len(state['archives'])==2
    assert all(len(r['attempts'])==1 and r['state']=='archived' for r in state['targets'].values())
    assert [len(item['target_ids']) for item in state['archives']]==[2,2]
    assert len({i for item in state['archives'] for i in item['target_ids']})==4
    for item in state['archives']:assert verify_target_archive(root/item['path'],item['metadata'])['target_count']==2
    assert not list((root/'targets').iterdir())
    assert len(list((root/'generation_progress').glob('*.finished')))==4
    assert not list((root/'generation_progress').glob('*.running'))


def test_existing_running_finished_and_error_all_skip_without_inference(tmp_path,monkeypatch,capsys):
    cfg=config(tmp_path,3);calls=fake_generation(monkeypatch)
    progress=TargetProgress(cfg.output_dataset,discover_targets(cfg.input_dataset))
    for row,suffix in zip(discover_targets(cfg.input_dataset),('running','finished','error')):
        progress.migrate(row['key'],suffix,'Already recorded error\n')
    result=run_campaign(cfg)
    assert not calls and result['worker']['claimed']==0 and result['worker']['skipped']==3
    assert result['worker']['no_available_targets'] and 'no_available_targets' in capsys.readouterr().out
    assert not (Path(cfg.output_dataset)/'jobs').exists()


def test_running_claim_is_empty_before_child_and_competing_finish_becomes_error(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path,2),compress=False);calls=fake_generation(monkeypatch)
    backend=steer_campaign.run_target_job;root=Path(cfg.output_dataset)
    def execute(command,log,env):
        request=read_json(command[-1]);key=Path(request['output_dataset']).name
        running=root/'generation_progress'/(key+'.running')
        assert running.is_file() and running.stat().st_size==0
        backend(command,log,env)
        if key==target_key('target_000/reference'):
            (root/'generation_progress'/(key+'.finished')).write_bytes(b'')
    monkeypatch.setattr(steer_campaign,'run_target_job',execute)
    result=run_campaign(cfg)
    assert len(calls)==2 and result['worker']['errors']==1 and result['worker']['finished']==1
    key=target_key('target_000/reference');error=root/'generation_progress'/(key+'.error')
    text=error.read_text()
    assert 'Concurrent target execution' in text and 'Traceback' in text
    assert 'CPU capture fixture' in text  # complete child log accompanies traceback
    assert not error.with_suffix('.finished').exists() and not error.with_suffix('.running').exists()


def test_no_remainder_archive_while_another_target_is_running(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path,2),archive_every=2);calls=fake_generation(monkeypatch)
    progress=TargetProgress(cfg.output_dataset,discover_targets(cfg.input_dataset))
    progress.migrate(target_key('target_000/reference'),'running')
    result=run_campaign(cfg)
    assert len(calls)==1 and not result['archives'] and result['progress']['running']==1
    assert result['status']=='running'


def test_published_archive_reservation_recovers_without_regenerating(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path,2),archive_every=2);calls=fake_generation(monkeypatch)
    original=target_archiving.archive_target_directories
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        raise KeyboardInterrupt('Simulated stop after tar publication')
    monkeypatch.setattr(target_archiving,'archive_target_directories',interrupted)
    with pytest.raises(KeyboardInterrupt):run_campaign(cfg)
    root=Path(cfg.output_dataset)
    assert read_json(root/'campaign_state.json')['archive_pending']['target_ids']==['target_000/reference','target_001/reference']
    monkeypatch.setattr(target_archiving,'archive_target_directories',original)
    result=run_campaign(cfg)
    assert len(calls)==2 and result['status']=='complete' and len(result['archives'])==1
    assert result['worker']['no_available_targets'] and not list((root/'targets').iterdir())


def test_legacy_campaign_backfills_finished_and_complete_error_log(tmp_path,monkeypatch):
    cfg=replace(config(tmp_path,2),compress=False);calls=fake_generation(monkeypatch,fail_once='target_000')
    run_campaign(cfg);root=Path(cfg.output_dataset);state=read_json(root/'campaign_state.json')
    old_error_log=root/state['targets']['target_000/reference']['attempts'][0]['log']
    old_error_log.write_text('Legacy child traceback: device unavailable\nFinal diagnostic line\n')
    state.pop('progress_format');atomic_json(root/'campaign_state.json',state)
    for p in (root/'generation_progress').iterdir():
        if p.suffix in ('.running','.finished','.error'):p.unlink()
    result=run_campaign(cfg)
    assert len(calls)==2 and result['worker']['no_available_targets']
    error=root/'generation_progress'/(target_key('target_000/reference')+'.error')
    assert 'Final diagnostic line' in error.read_text() and 'Simulated unavailable GPU' in error.read_text()
    assert (root/'generation_progress'/(target_key('target_001/reference')+'.finished')).stat().st_size==0


def test_progress_api_rejects_keys_that_escape_the_marker_directory(tmp_path):
    with pytest.raises(ValueError,match='safe'):
        TargetProgress(tmp_path,[{'target_id':'A','key':'../elsewhere'}])
