import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evomolsteer.ablation.collection import (
    merge_observations, historical_difference, tool_outcome, evidence_coverage,
    capture_state,
    compare_snapshots,
)


def frame():
    return pd.DataFrame({'batch':[1,1,2,2],'slot':[0,1,0,1],
        'pic50_on_rescore':[7.,8.,7.5,8.5],'valid_connected':[True,False,True,True],
        'pb_fast_pass':[True,False,True,True]})


def test_shared_rewards_do_not_multiply_attempt_counts():
    f=frame()
    assert merge_observations([f,f.copy(),f.copy()]).equals(merge_observations([f]))


def test_different_outputs_for_shared_attempt_are_not_silently_deduplicated():
    a=frame();b=frame();b.loc[0,'pic50_on_rescore']=7.1
    with pytest.raises(ValueError,match='Conflicting'):
        merge_observations([a,b])


def test_missing_scores_are_never_excluded_from_analysis():
    a=frame();a.loc[1,'pic50_on_rescore']=np.nan
    with pytest.raises(ValueError,match='All-attempt'):
        merge_observations([a])


def test_historical_comparison_is_batch_based_and_marks_unequal_budget():
    a=frame();b=frame();b.pic50_on_rescore-=.1
    result=historical_difference(a,b,replicates=1000)
    assert result['delta_pic50']==pytest.approx(.1)
    assert result['candidate_batches']==2
    assert result['equal_online_budget'] is False
    assert result['not_independent_final_recipe_confirmation'] is True


def test_failed_curve_request_is_not_recorded_as_a_delivered_curve():
    action={'action':'tool','tool':'get_curve'}
    assert tool_outcome(action,{'ok':False,'error':{'type':'KeyError'}})=='error'
    assert tool_outcome(action,None)=='missing_observation'
    assert tool_outcome(action,{'ok':True,'curve':{'times':[0.,.1]}})=='delivered'
    assert tool_outcome({'action':'finish'},None) is None


def test_coverage_keeps_unread_tables_and_rejects_fabricated_references():
    packet={'tables':{
        'trends':{'columns':['evidence_id','feature'],'rows':[['t:0','a'],['t:1','b']]},
        'regional':{'columns':['evidence_id'],'rows':[['r:0']]},
    }}
    result=evidence_coverage(packet,{'trends':{'t:0'}})
    assert result['trends']['fraction']==.5
    assert result['regional']['delivered_unique_rows']==0
    assert result['regional']['available_rows']==1
    with pytest.raises(ValueError,match='not in the supplied'):
        evidence_coverage(packet,{'trends':{'t:999'}})


def test_live_state_truncation_retries_and_preserves_exact_captured_bytes(monkeypatch):
    complete=b'{"status": "running", "n": 3}\n'
    snapshots=iter([b'', b'{"status":', complete])
    monkeypatch.setattr(Path,'read_bytes',lambda _:next(snapshots))
    blob,state=capture_state('state.json',delay=0)
    assert blob==complete
    assert state=={'status':'running','n':3}


def test_persistently_incomplete_state_is_rejected_instead_of_imputed(monkeypatch):
    monkeypatch.setattr(Path,'read_bytes',lambda _:b'{')
    with pytest.raises(json.JSONDecodeError):
        capture_state('state.json',delay=0)


def test_snapshot_comparison_marks_overlapping_data_and_rejects_other_study(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    for folder,n,cohorts,modules in [(old,2,['c0'],['A1']), (new,4,['c0','c1'],['A1','D1'])]:
        folder.mkdir()
        (folder/'source_manifest.json').write_text(json.dumps({
            'study':'same','collection_time_Asia_Shanghai':folder.name,
            'state_sha256':folder.name,'generated_attempts_deduplicated':n}))
        pd.DataFrame({'cohort':cohorts}).to_csv(folder/'cohort_quality.csv',index=False)
        pd.DataFrame({'module':modules}).to_csv(folder/'module_results.csv',index=False)
        pd.DataFrame({'program':['p'],'label':['M2'],'attempts':[n],
                      'mean_pic50':[7. if n==2 else 7.1]}).to_csv(folder/'program_quality.csv',index=False)
    change=compare_snapshots(old,new)
    assert change['additional_attempts']==2
    assert change['added_cohorts']==['c1']
    assert change['added_completed_modules']==['D1']
    assert change['overlapping_snapshots_not_independent_experiments'] is True
    with pytest.raises(ValueError,match='preserve'):
        compare_snapshots(new,old)
    manifest=json.loads((new/'source_manifest.json').read_text())
    manifest['study']='different'
    (new/'source_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='same study'):
        compare_snapshots(old,new)
