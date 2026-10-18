"""Scientific invariants of historical random-input replay; no FLOWR GPU needed."""
import copy
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import torch

from evomolsteer.io import read_json
from evomolsteer.path_replay.common import ancestry,identity,validate,program_identity
from evomolsteer.path_replay.tape import ReplayRandom,verify_native_primitive
from evomolsteer.path_replay.analysis import paired_cluster,distribution
from evomolsteer.path_replay.collection import collect,unpack_snapshot

REPO=Path(__file__).resolve().parents[1]


def config(): return read_json(REPO/'configs/experiments/path_replay_b50_20261018/config.json')


def test_ancestry_follows_each_preselection_parent_not_final_or_root_slot():
    selected=np.array([[2,0,2],[1,1,0],[2,1,0]])
    before,roots=ancestry(selected)
    assert before.tolist()==[[2,0,0],[0,1,1],[2,1,0]]
    assert roots.tolist()==[2,0,0]


def test_invalid_parent_rejected():
    with pytest.raises(ValueError): ancestry(np.array([[0,3]]))


def test_protocol_has_b50_common_top200_and_all_1000_paths():
    c=config();validate(c)
    assert sum(x['n'] for x in c['conditions'] if not x.get('alias_of'))==2600
    assert len(set(k for g in c['groups'] for k in g['ids']))==1000
    first={k for g in c['groups'][:4] for k in g['ids']}
    assert first=={a['id'] for a in c['attempts'] if a['rank']<=200}
    assert all(len(g['ids'])==50 for g in c['groups'])


def test_saved_singleton_com_axis_is_flattened_without_changing_values():
    from evomolsteer.path_replay.native import InitialPool
    c={'remote_dataset':'unused','sources':{'0':{'relative':'unused'}}}
    pool=InitialPool(c)
    xyz=torch.arange(50*25*3,dtype=torch.float32).reshape(50,25,3)
    centers=np.arange(150,dtype=np.float32).reshape(50,1,3)
    pool.cache[0]={'prior':{'coords':xyz},'pocket_target':{'coords':xyz},'target_com':centers,'coord_scale':1.}
    rows=[{'batch':0,'root_slot':49-i} for i in range(50)]
    prior,pocket,com,scale=pool.load(rows,device='cpu')
    assert com.shape==(50,3) and torch.equal(com,torch.from_numpy(centers[::-1].copy().reshape(50,3)))
    assert prior['coords'].shape==(50,25,3) and pocket['complex'][0].com.shape==(3,)


def test_scoring_rejects_broadcasted_coordinate_axes_before_model_call():
    from evomolsteer.path_replay.worker import score
    with pytest.raises(ValueError,match='broadcast dimension'):
        score(None,{'coords':torch.zeros(50,25,3)},None,torch.zeros(50,1,3))


def test_protocol_rejects_smaller_batch_even_with_recomputed_hash():
    c=config();c['batch_size']=24;c['protocol_sha256']=identity({k:v for k,v in c.items() if k!='protocol_sha256'})
    with pytest.raises(ValueError): validate(c)


def test_numeric_identity_ignores_provenance_but_not_strength():
    a={'program_id':'A','dose':{'rho':.66},'analyst_sha256':'old'}
    b=copy.deepcopy(a);b.update(program_id='B',analyst_sha256='new')
    assert program_identity(a)==program_identity(b)
    b['dose']['rho']=.3
    assert program_identity(a)!=program_identity(b)


def test_native_categorical_primitive_and_rng_advancement():
    assert all(x['passed'] for x in verify_native_primitive('cpu'))


@pytest.fixture
def cpu_tapes(tmp_path):
    directory=tmp_path/'tapes';directory.mkdir();expected={};before={}
    shapes=[(50,25,15),(50,25,8),(50,25,25,5)]
    for b in range(2):
        generator=torch.Generator().manual_seed(500+b);states=[];draws=[]
        before[b]=np.stack([np.roll(np.arange(50),b+3),np.roll(np.arange(50),b+7)])
        for step in range(2):
            calls=[generator.get_state().numpy().copy()]
            normal=torch.randn((50,25,3),generator=generator);samples=[]
            for shape in shapes:
                calls.append(generator.get_state().numpy().copy())
                p=torch.ones((int(np.prod(shape[:-1])),shape[-1]))
                samples.append(torch.multinomial(p,1,True,generator=generator).reshape(shape[:-1]))
            states.append(calls);draws.append((normal,samples))
        np.savez(directory/f'batch_{b:03d}.npz',states=np.array(states),before=before[b],
                 shape_coords=np.array([50,25,3]),**{'shape_'+k:np.array(s) for k,s in zip(('atomics','charges','bonds'),shapes)})
        expected[b]=draws
    attempts=[{'batch':i%2,'final_slot':i} for i in range(50)]
    return tmp_path,attempts,expected,before,shapes


def test_replay_is_independent_of_new_batch_grouping_and_preserves_global_rng(cpu_tapes):
    work,attempts,expected,before,shapes=cpu_tapes
    replay=ReplayRandom({'batch_size':50},work,attempts,device='cpu');state=torch.get_rng_state()
    with replay.activate():
        for step in range(2):
            replay.begin_step(step)
            xyz=torch.randn((50,25,3))
            cats=[torch.distributions.Categorical(torch.ones(s)).sample() for s in shapes]
            replay.end_step()
            for i,a in enumerate(attempts):
                b=a['batch'];j=before[b][step,a['final_slot']]
                assert torch.equal(xyz[i],expected[b][step][0][j])
                for k in range(3): assert torch.equal(cats[k][i],expected[b][step][1][k][j])
    assert torch.equal(state,torch.get_rng_state())


def test_counterfactual_sampling_uses_new_probabilities_not_historical_categories(cpu_tapes):
    work,attempts,expected,before,shapes=cpu_tapes
    replay=ReplayRandom({'batch_size':50},work,attempts,device='cpu')
    with replay.activate():
        replay.begin_step(0);torch.randn((50,25,3))
        for shape in shapes:
            probs=torch.zeros(shape);probs[...,2]=1
            sampled=torch.distributions.Categorical(probs).sample()
            assert bool((sampled==2).all())
        replay.end_step()
    assert bool((expected[0][0][1][0]!=2).any())


def test_missing_or_reordered_random_calls_fail_closed(cpu_tapes):
    work,attempts,*_=cpu_tapes;replay=ReplayRandom({'batch_size':50},work,attempts,device='cpu')
    replay.begin_step(0)
    with pytest.raises(ValueError): replay.end_step()
    with pytest.raises(ValueError): replay.draw_categorical(torch.ones(1250,15),1,True)


def test_paired_ci_clusters_original_source_batches_and_is_reproducible():
    base=[{'id':str(i),'batch':i//3,'pic50':7.} for i in range(12)]
    candidate=[dict(r,pic50=r['pic50']+.2) for r in base]
    result=paired_cluster(candidate,base,1000,42)
    assert result==paired_cluster(candidate,base,1000,42)
    assert result['source_batches']==4 and result['n_paired']==12
    assert result['mean_delta_ci_low']==pytest.approx(.2)


def test_distribution_reports_failure_coverage_without_zero_imputation():
    rows=[{'pic50':8.,'valid_connected':True},{'pic50':None,'status':'failed'}]
    out=distribution(rows,50,8.1)
    assert out['mean']==8. and out['score_coverage']==.02 and out['n_records']==2
    assert out['best_pb'] is None


def test_compact_snapshot_roundtrip(tmp_path):
    work=tmp_path/'work';(work/'methods').mkdir(parents=True)
    (work/'methods'/'example.txt').write_bytes(b'molecule result')
    (work/'queue_status.json').write_text('{"status":"running"}')
    (work/'runtime').mkdir();(work/'runtime'/'not_a_result.txt').write_text('do not duplicate source')
    archive=tmp_path/'snapshot.tar.gz';collect({'protocol':'test'},work,archive)
    result=unpack_snapshot(archive,tmp_path/'unpacked')
    assert result['files']==2
    assert (tmp_path/'unpacked/methods/example.txt').read_bytes()==b'molecule result'
    assert not (tmp_path/'unpacked/runtime').exists()


def test_snapshot_rejects_path_traversal(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for name,data in [('collection_manifest.json',b'{}'),('../bad',b'bad')]:
            m=tarfile.TarInfo(name);m.size=len(data);f.addfile(m,io.BytesIO(data))
    with pytest.raises(ValueError): unpack_snapshot(archive,tmp_path/'safe')


def test_frozen_r15_reward_gradient_and_dose_are_identical():
    from evomolsteer.path_replay.native import r15_equivalence
    result=r15_equivalence(REPO,config(),device='cpu')
    assert result['passed'] and result['gradient_equal'] and result['injected_state_equal']


def test_alias_cannot_start_a_second_inference_worker(tmp_path):
    from evomolsteer.path_replay.worker import run
    path=tmp_path/'protocol.json';path.write_text(json.dumps(config()))
    with pytest.raises(ValueError,match='Aliased condition'):
        run(REPO,path,tmp_path,'motif_R15_v2')


def test_missing_zero_replay_results_block_guidance(tmp_path):
    from evomolsteer.path_replay.queue import zero_gate
    with pytest.raises(ValueError,match='200 zero-guidance'):
        zero_gate(tmp_path,config())


def test_crashed_guided_condition_is_skipped_without_aborting_peers(tmp_path,monkeypatch):
    from evomolsteer.path_replay import queue
    cfg=config();c=next(c for c in cfg['conditions'] if c['id']=='NC_R11')
    class Crashed:
        pid=123
        def poll(self): return 1
        def wait(self,**kw): return 1
    monkeypatch.setattr(queue.subprocess,'Popen',lambda *a,**kw:Crashed())
    queue.run_condition(REPO,tmp_path/'config.json',cfg,c,tmp_path)
    assert not queue.pending(tmp_path,cfg,c)
    rows=queue.summary(tmp_path,cfg)
    assert next(r for r in rows if r['id']=='NC_R11')['skipped_condition_error']==200
    assert next(r for r in rows if r['id']=='M2')['pending']==200
