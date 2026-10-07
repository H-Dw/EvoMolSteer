import pandas as pd
import pytest
from evomolsteer.generation.path_evaluation import summarize_tail,paired_effect
def data(scores):
    return pd.DataFrame({'batch':[0,0,1,1],'slot':[0,1,0,1],'pic50_on_rescore':scores,
        'valid_connected':[True,False,True,True],'smiles':['CC','CN','CC','CO'],
        'pb_fast_pass':[True,False,True,True],'energy_status':['converged']*4,'mmff_relief_per_heavy':[.2,.3,.4,.5]})
def test_tail_counts_valid_graphs_and_keeps_failure_denominator():
    s=summarize_tail(data([8.5,10,8.4,7]),8.2)
    assert s['elite_valid_n']==2 and s['elite_unique_graphs']==1 and s['elite_yield']==.5
def test_paired_effect_and_coverage():
    s=paired_effect(data([8,8,8,8]),data([7,7,7,7]))
    assert s['paired_mean_pic50']==1 and s['batch_bootstrap_CI95']==[1.,1.]
    with pytest.raises(ValueError):paired_effect(data([8]*4),data([7]*4).iloc[:3])

def test_one_batch_does_not_claim_a_batch_uncertainty_interval():
    s=paired_effect(data([8]*4).iloc[:2],data([7]*4).iloc[:2])
    assert s['paired_mean_pic50']==1 and s['n_batches']==1
    assert s['batch_bootstrap_CI95'] is None
