import pytest
from evomolsteer.adaptive_schedule.io import read, write
from evomolsteer.adaptive_schedule.grid import CHECKPOINT_STEPS, HEAD_STEPS


def test_mixed_snapshot_keeps_prefix_suffix_and_is_restartable_after_cleanup(tmp_path):
    torch = pytest.importorskip('torch')
    h5py = pytest.importorskip('h5py')
    pytest.importorskip('rdkit.Chem')
    from evomolsteer.adaptive_schedule.flowr import save_pt, load_pt
    from evomolsteer.adaptive_schedule.wave_retention import materialize_snapshot, verify_snapshot, prune_payload
    import numpy as np
    parent, wave, retained = [tmp_path/name for name in ['parent','wave','retained']]
    for folder, size in [(parent,100),(wave,400)]:
        folder.mkdir()
        value = torch.arange(size,dtype=torch.float32).reshape(size,1,1).expand(size,1,3).clone()
        for step in CHECKPOINT_STEPS:
            state = {'rows':size,'step':step,'context_sha256':'context','rng_sha256':'streams',
                'current':{'coords':value+step},'cond':{'coords':value},
                'times':[torch.full((size,),step/100)]*3, 'previous_assignment':torch.zeros(size,1,dtype=torch.long),
                'integrator':{'guard_warned':False},'streams':[]}
            save_pt(folder/f'checkpoint_{step:03d}.pt.gz',state)
        for step in HEAD_STEPS:
            save_pt(folder/f'head_{step:03d}.pt.gz',dict(prediction={'coords':value},raw_head={'coords':value},
                                                      assignment=value[:,:,0],gradient_world=value))
        save_pt(folder/'terminal.pt.gz',{'coords':value})
        with h5py.File(folder/'trace.h5','w') as f:
            f.create_dataset('coords',data=np.stack([(value+s).numpy() for s in range(100)]),chunks=(1,size,1,3))
        write(folder/'nodes.json',{str(s):{'step':s,'local_loss':list(range(size))} for s in CHECKPOINT_STEPS})
        write(folder/'steps.json',[{'step':s,'loss':list(range(size))} for s in range(100)])
        (folder/'molecules_all_built.sdf').touch()
        for i in range(size): write(folder/'profiles'/f'slot_{i:03d}'/'failure.json',{'reason':'builder_failed'})
    write(wave/'manifest.json',{'start_step':26})
    winners = [i+100 if i%2 else None for i in range(100)]
    best = [{'record':{'build_success':False},'id':str(i)} for i in range(100)]
    materialize_snapshot(parent,wave,retained,best,winners)
    verify_snapshot(retained)
    assert load_pt(retained/'checkpoint_010.pt.gz')['current']['coords'][1,0,0] == 11
    assert load_pt(retained/'checkpoint_051.pt.gz')['current']['coords'][1,0,0] == 152
    assert load_pt(retained/'checkpoint_051.pt.gz')['current']['coords'][0,0,0] == 51
    with h5py.File(retained/'trace.h5','r') as f:
        assert f['coords'][10,1,0,0] == 11 and f['coords'][50,1,0,0] == 151
    prune_payload(parent); prune_payload(wave)
    verify_snapshot(retained)
    assert read(retained/'nodes.json')['51']['local_loss'][1] == 101


def test_checkpoint_rejects_different_random_streams():
    pytest.importorskip('torch');pytest.importorskip('h5py')
    from evomolsteer.adaptive_schedule.wave_native import mix_checkpoint
    a={'step':26,'context_sha256':'x','rng_sha256':'a','integrator':{}}
    b=dict(a,rng_sha256='b')
    with pytest.raises(ValueError,match='streams'):
        mix_checkpoint(a,b,[None]*100)
