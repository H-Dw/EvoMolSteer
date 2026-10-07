import json,importlib.util
from pathlib import Path
import numpy as np
import pytest
from evomolsteer.io import write_json,digest
from evomolsteer.storage.trajectory import pack_arrays
from evomolsteer.storage.evaluation_view import export_evaluation_view,execution_arrays,state_signature
from evomolsteer.storage.generation_archive import verify_generation_archive
from evomolsteer.generation.path_evaluation import execution_audit

spec=importlib.util.spec_from_file_location('archive_cli',Path(__file__).parents[1]/'scripts/archive_generation.py')
archiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(archiver)

def fixture(tmp_path):
    root=tmp_path/'source';run=root/'results/test';folder=run/'gradient/batch_030';folder.mkdir(parents=True)
    rng=np.random.default_rng(42)
    arrays={'current_coords':rng.normal(size=(100,2,3,3)).astype('f4'),
        'current_atomics':rng.integers(1,8,(100,2,3),dtype='i2'),
        'current_bonds':rng.integers(0,3,(100,2,3,3),dtype='u1'),
        'selected_indices':np.tile(np.arange(2,dtype='i4'),(100,1)),
        'resampled':np.zeros(100,dtype=bool)}
    pack_arrays(arrays,folder/'trajectory.h5',normalize=False)
    write_json(run/'config.json',{'experiment':{'seed':42,'arms':'gradient','n':2,'batch':2},'extension':{'code_commit':'fixture'}})
    write_json(run/'reward_program.json',{'window':[.2,.6]});write_json(run/'COMPLETE.json',{'status':'complete'})
    write_json(folder/'COMPLETE.json',{'n':2});write_json(run/'coordinate_gradient_preflight.json',{'passed':True,'affinity_head_gradient':False})
    trace=[{'step':i,'particle_resampled':False,'score_time':i/100,'state_time':(i+1)/100,
        'injection_l2_A':[0.,0.],'production_target_forward_calls':1,'affinity_outputs_detached':True,'reward_evaluated':False} for i in range(100)]
    (folder/'guidance_trace.jsonl').write_text('\n'.join(json.dumps(v) for v in trace))
    (folder/'molecules_raw_decodable.sdf').write_bytes(b'unchanged terminal structure fixture')
    (root/'inputs').mkdir();(root/'inputs/protein.pdb').write_bytes(b'unchanged input fixture')
    return root,folder,arrays

def test_full_and_view_execution_audits_are_identical(tmp_path):
    root,folder,arrays=fixture(tmp_path);out=tmp_path/'view';m=export_evaluation_view(root,'test',out)
    assert execution_audit(root,'test')==execution_audit(out,'test')
    snapshot=out/'results/test/gradient/batch_030/execution_state.npz'
    with np.load(snapshot) as z:
        for key in ['current_coords','current_atomics','current_bonds']:
            assert z[key].dtype==arrays[key].dtype and z[key].tobytes()==arrays[key][:1].tobytes()
    for rel,sha in m['copied_file_sha256'].items():assert digest(out/rel)==sha==digest(root/rel)
    assert (folder/'trajectory.h5').exists() and not (out/'results/test/gradient/batch_030/trajectory.h5').exists()

def test_snapshot_tampering_and_real_selection_are_detected(tmp_path):
    root,folder,arrays=fixture(tmp_path);out=tmp_path/'view';export_evaluation_view(root,'test',out)
    p=out/'results/test/gradient/batch_030/execution_state.npz';p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):execution_audit(out,'test')
    arrays['resampled'][10]=True;np.savez_compressed(tmp_path/'selected.npz',**arrays)
    a=execution_arrays(tmp_path/'selected.npz');assert a['resampled'][10]
    assert state_signature(a)==state_signature(arrays)

def test_normalized_ancestor_aliases_keep_exact_initial_bytes(tmp_path):
    _,_,arrays=fixture(tmp_path)
    for key in ['current_coords','current_atomics','current_bonds']:
        arrays[key.replace('current_','proposal_')]=np.roll(arrays[key],-1,axis=0)
    source=tmp_path/'aliased.h5';report=pack_arrays(arrays,source,normalize=True)
    assert len(report['lineage_aliases'])==3
    capsule=execution_arrays(source)
    for key in ['current_coords','current_atomics','current_bonds']:
        assert capsule[key].tobytes()==arrays[key][:1].tobytes()

def test_compact_archive_verifies_every_payload_and_preserves_original(tmp_path):
    root,folder,_=fixture(tmp_path);archive=tmp_path/'transport.tar.gz';original=digest(folder/'trajectory.h5')
    meta=archiver.archive(root,'test',archive,evaluation_only=True)
    restored=tmp_path/'restored';verify_generation_archive(archive,meta,restored)
    assert execution_audit(restored,'test')==execution_audit(root,'test')
    assert digest(folder/'trajectory.h5')==original and not meta['source_deleted']
