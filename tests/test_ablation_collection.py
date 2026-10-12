import numpy as np
import pandas as pd
import pytest

from evomolsteer.ablation.collection import (
    merge_observations, historical_difference, tool_outcome, evidence_coverage,
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
