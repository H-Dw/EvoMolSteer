import pytest
import json,gzip
import numpy as np
import pyarrow.parquet as pq
from evomolsteer.generation.head_response import compare


def sample(arm,t,scores,seed=42):
    return dict(arm=arm,batch=0,seed=seed,score_time=t,scores=scores)


def test_small_mean_with_large_individual_effect_is_detected_as_cancellation():
    native=[sample('unguided',0,[7,7]),sample('unguided',.01,[7,7])]
    guided=[sample('gradient',0,[7,7]),sample('gradient',.01,[7.2,6.8])]
    result=compare(guided,native)
    assert result.iloc[-1].mean_delta==pytest.approx(0)
    assert result.iloc[-1].mean_absolute_delta==pytest.approx(.2)
    assert result.iloc[-1].positive_fraction==.5
    assert result.iloc[-1].net_signal_fraction==pytest.approx(0)
    assert result.iloc[-1].cancellation_fraction==pytest.approx(1)


def test_trajectory_rate_uses_actual_time_and_rejects_seed_or_missing_coverage():
    native=[sample('unguided',.1,[7,7]),sample('unguided',.2,[7,7])]
    guided=[sample('gradient',.1,[7,7]),sample('gradient',.2,[7.1,7.1])]
    assert compare(guided,native).d_dt_mean_delta.tolist()==pytest.approx([1,1])
    guided[-1]['seed']=43
    with pytest.raises(ValueError,match='seed'):compare(guided,native)
    with pytest.raises(ValueError,match='Missing'):compare(guided,native[:1])


def tiny_campaign(root,arms,checkpoint='same'):
    from evomolsteer.io import write_json
    from evomolsteer.storage.trajectory import pack_arrays
    root.mkdir(parents=True)
    write_json(root/'config.json',{'experiment':{'arms':','.join(arms)},'extension':{
        'checkpoint_sha256':checkpoint,'native_integrator_parameters':{'steps':3},'code_commit':'fixture'}})
    write_json(root/'COMPLETE.json',{'status':'complete'})
    (root/'reference.json.gz').write_bytes(gzip.compress(json.dumps({'required_input_sha256':{'pocket':'same'}}).encode()))
    for arm in arms:
        path=root/arm/'batch_000';path.mkdir(parents=True)
        write_json(path/'final_records.json',[{'seed':42},{'seed':42}])
        scores=np.array([[7,7],[7.2,6.8],[100,100]],np.float32) if arm=='gradient' else np.full((3,2),7,np.float32)
        pack_arrays({'score_time':np.tile(np.array([.1,.2,.3],np.float32)[:,None],(1,2)),
                     'state_time':np.array([.2,.3,.4],np.float32),'resampled':np.zeros(3,bool),
                     'pic50_on':scores},path/'trajectory.h5')


def test_compact_labels_support_native_reuse_and_atomic_completion(tmp_path):
    from evomolsteer.generation.head_response import analyze,verify_report
    first=tmp_path/'first';tiny_campaign(first,['unguided','gradient'])
    out=tmp_path/'reports';r=analyze(first,out,[.1,.3],tmp_path/'unused_reference')
    assert r['used_native_source']=='current_native_arm' and r['inference_commit']=='fixture'
    assert len(pq.read_table(out/'head_scores_window.parquet').to_pylist())==4
    assert all(v['score_time']<.3 for v in pq.read_table(out/'head_scores_window.parquet').to_pylist())
    assert verify_report(out)==r
    second=tmp_path/'second';tiny_campaign(second,['gradient'])
    result=analyze(second,tmp_path/'second_report',[.1,.3],out/'head_scores_window.parquet')
    assert result['window_end_nodes'][0]['cancellation_fraction']==pytest.approx(1)
    wrong=tmp_path/'wrong';tiny_campaign(wrong,['gradient'],'different')
    with pytest.raises(ValueError,match='contract mismatch'):analyze(wrong,tmp_path/'bad_report',[.1,.3],out/'head_scores_window.parquet')
    assert not (tmp_path/'bad_report/head_window_response.json').exists()
    (out/'paired_head_window.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):verify_report(out)
