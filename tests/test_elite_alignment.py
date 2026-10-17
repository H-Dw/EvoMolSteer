"""Input identity, controller equivalence and actual subprocess fault isolation."""
import copy
import json
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json, exclusive_lock, WriterBusyError
from evomolsteer.elite_alignment.common import binding, read_result, validate_protocol, validate_condition, result_path
from evomolsteer.elite_alignment.queue import run_condition

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/elite_initial_alignment_20261017/config.json'


def test_actual_cohort_and_initial_identity():
    cfg = read_json(CONFIG); validate_protocol(cfg)
    assert [cfg['cohorts'][k]['n'] for k in ('top05', 'top10', 'top20')] == [50, 100, 200]
    assert [cfg['cohorts'][k]['unique_roots'] for k in ('top05', 'top10', 'top20')] == [11, 14, 18]
    assert cfg['attempts'][0]['id'] == 'F_b011_s030'
    assert cfg['attempts'][0]['root_slot'] == 31
    assert len({a['suffix_seed'] for a in cfg['attempts']}) == 200
    assert len({a['initial_id'] for a in cfg['attempts']}) == 18
    for c in cfg['conditions']: validate_condition(ROOT, c)


def test_complete_receipt_requires_artifact_integrity(tmp_path):
    cfg = {'protocol_sha256': 'frozen'}; c = {'id': 'test'}; a = {'id': 'A'}
    file = result_path(tmp_path, c, a); file.parent.mkdir(parents=True)
    artifact = file.parent / 'states.npz'; artifact.write_bytes(b'trusted')
    atomic_json(file, {'status': 'complete', 'binding': binding(cfg, c, a), 'artifacts': {'states.npz': digest(artifact)}})
    assert read_result(tmp_path, cfg, c, a)['status'] == 'complete'
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'): read_result(tmp_path, cfg, c, a)


@pytest.mark.parametrize('timed_out', [False, True])
def test_abnormal_child_does_not_prevent_next_condition(tmp_path, timed_out):
    cfg = {'protocol_sha256': 'frozen', 'max_worker_restarts': 0, 'poll_seconds': .01,
           'sample_timeout_seconds': .2 if timed_out else 5, 'startup_timeout_seconds': 5,
           'attempts': [{'id': 'A'}, {'id': 'B'}]}
    first, second = {'id': 'bad'}, {'id': 'next'}
    text = "import json,time;from pathlib import Path;Path('worker_state.json').write_text(json.dumps({'condition':'bad','attempt':'A'}));time.sleep("+('10' if timed_out else '.3')+");raise SystemExit(7)"
    run_condition(tmp_path, tmp_path / 'unused', cfg, first, tmp_path, [sys.executable, '-c', text])
    assert read_result(tmp_path, cfg, first, cfg['attempts'][0])['status'] == 'failed'
    assert read_result(tmp_path, cfg, first, cfg['attempts'][1])['status'] == 'skipped_condition_error'
    # A second actual child is still dispatched after the previous crash/timeout.
    run_condition(tmp_path, tmp_path / 'unused', cfg, second, tmp_path,
                  [sys.executable, '-c', "from pathlib import Path;Path('next_was_run').write_text('yes')"])
    assert (tmp_path / 'next_was_run').read_text() == 'yes'


def test_lock_rejects_duplicate_queue(tmp_path):
    with exclusive_lock(tmp_path / 'lock'):
        with pytest.raises(WriterBusyError):
            with exclusive_lock(tmp_path / 'lock'): pass


def test_legacy_adapter_matches_frozen_original_after_native(tmp_path,monkeypatch):
    from evomolsteer.generation.endpoint_controller import EndpointCoordinateExtension
    from evomolsteer.elite_alignment.control import LegacyControl
    # The old method imports only this conversion helper from a FLOWR-loading module.
    monkeypatch.setitem(sys.modules,'evomolsteer.generation.controller',SimpleNamespace(nparr=lambda x:x.detach().cpu().numpy()))
    cfg = read_json(CONFIG); c = next(x for x in cfg['conditions'] if x['id'] == 'M2')
    program, reference = validate_condition(ROOT, c)
    torch.manual_seed(42)
    before = torch.randn(1, 25, 3); endpoint = before + .4*torch.randn_like(before)
    curr = {'coords': before+.03*torch.randn_like(before), 'mask': torch.ones(1,25,dtype=torch.int32)}
    g = torch.randn_like(before); detail = {'dose_gate': torch.ones(1), 'core_mask': curr['mask'].bool()}
    model = SimpleNamespace(coord_scale=1., integrator=SimpleNamespace(use_cosine_scheduler=False))
    pocket = {'coords': torch.full((1,3,3), 30.), 'mask': torch.ones(1,3,dtype=torch.int32)}
    original = object.__new__(EndpointCoordinateExtension)
    model._lineage = SimpleNamespace(i=10,t=.1,dt=.01,arm='gradient',arr=defaultdict(list),path=tmp_path)
    original.model=model;original.program=program;original.before=before;original.endpoint_coords=endpoint
    original.cached=(g,torch.zeros(1),detail);original.pocket=pocket;original.path_rms=torch.zeros(1)
    original.controlled_updates=0;original.production_calls_this_step=1;original.audit_calls_this_step=0
    original.opt=SimpleNamespace(window=.5)
    expected=original.after_native(copy.deepcopy(curr))
    new=LegacyControl(model,program,reference,torch.zeros(1,3),capture_time=None)
    new.before=before;new.pocket=pocket;new.t=.1;new.dt=.01
    new.legacy_cache=(g,torch.zeros(1),detail,endpoint)
    actual=new.after(copy.deepcopy(curr),10)
    assert torch.equal(expected['coords'],actual['coords'])
    assert torch.equal(original.path_rms,new.total)


def test_constant_blend_alias_same_value_and_gradient():
    from evomolsteer.elite_alignment.control import frozen_reward
    cfg=read_json(CONFIG); old=next(x for x in cfg['conditions'] if x['id']=='blend_0p5')
    p,r=validate_condition(ROOT,old)
    job=read_json(ROOT/'configs/experiments/temporal_guidance_run_20261013/jobs/DSF_287e3b494ad5_b6376_g_indexed_batch.json')
    newer=copy.deepcopy(old);newer.update(program=job['program'],program_sha256=digest(ROOT/job['program']))
    q,s=validate_condition(ROOT,newer)
    a,b=frozen_reward(p,r),frozen_reward(q,s)
    torch.manual_seed(42)
    for t in (0.,.1,.25,.49):
        x=(torch.tensor(r['frames'][0]['teacher_endpoint_A'][0],dtype=torch.float64)[None]+torch.randn(1,25,3,dtype=torch.float64)*.3).requires_grad_(True)
        atoms=torch.zeros(1,25,dtype=torch.long);mask=torch.ones(1,25,dtype=torch.bool)
        va,da=a(x,atoms,mask,t,x.detach());vb,db=b(x,atoms,mask,t,x.detach())
        ga,=torch.autograd.grad(va.sum(),x);gb,=torch.autograd.grad(vb.sum(),x)
        assert torch.allclose(va,vb,rtol=1e-10,atol=1e-10)
        assert torch.allclose(ga,gb,rtol=1e-9,atol=1e-9)
        assert torch.equal(da['dose_gate'],db['dose_gate'])


def test_scheduled_dose_is_actually_consumed():
    from evomolsteer.elite_alignment.control import frozen_reward
    cfg=read_json(CONFIG)
    p,r=validate_condition(ROOT,next(c for c in cfg['conditions'] if c['id']=='W1'))
    reward=frozen_reward(p,r)
    x=torch.tensor(r['frames'][0]['teacher_endpoint_A'][0],dtype=torch.float64)[None]+.1
    atoms=torch.zeros(1,25,dtype=torch.long);mask=torch.ones(1,25,dtype=torch.bool)
    early=reward(x,atoms,mask,0.,x)[1]
    late=reward(x,atoms,mask,.49,x)[1]
    assert float(early['temporal_dose_factor'][0])<float(late['temporal_dose_factor'][0])
    assert float(early['temporal_dose_factor'][0])==pytest.approx(.35)


def test_empty_report_preserves_planned_counts_and_failure_coverage(tmp_path):
    from evomolsteer.elite_alignment.report import report
    from evomolsteer.elite_alignment.common import failure
    cfg=read_json(CONFIG)
    failure(tmp_path,cfg,cfg['conditions'][0],cfg['attempts'][0],'injected test error')
    report(tmp_path,cfg,bootstrap=False)
    summary=read_json(tmp_path/'analysis/summary.json')
    row=next(x for x in summary['metrics'] if x['cohort']=='top05' and x['condition']=='unguided')
    assert row['n_planned']==50 and row['n_failed']==1 and row['n_pending']==49
    assert row['score_coverage']==0 and row['mean_pic50'] is None
