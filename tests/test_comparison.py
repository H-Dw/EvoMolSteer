import numpy as np
import pytest
from evomolsteer.generation.comparison import paired_effect,summarize_records


def test_exact_small_batch_inference():
    r=paired_effect([1,1,1,1])
    assert r['sign_flip_p']==.125
    assert r['t_ci_low']==r['t_ci_high']==1
    assert paired_effect([0,0,0,0])['sign_flip_p']==1
    assert paired_effect([1])['sign_flip_p'] is None


def test_candidates_are_not_independent_replicates():
    records=[]
    for arm in ['unguided','gradient_region']:
        for batch in range(4):
            for slot in range(5):
                records.append(dict(arm=arm,batch=batch,slot=slot,build_success=True,connected=True,
                    pic50_on_rescore=7+batch*.1+(arm=='gradient_region')*.01,smiles='C',pairs_below_1_2A=0,qed=.5))
    b,s,p=summarize_records(records,'unguided')
    score=p[p.metric=='ck2_rescore_all'].iloc[0]
    assert score.n_batches==4 and score.mean_difference==pytest.approx(.01)
    with pytest.raises(ValueError,match='Duplicated'):summarize_records(records+[records[0]],'unguided')
    with pytest.raises(ValueError,match='Unmatched'):summarize_records(records[:-1],'unguided')
