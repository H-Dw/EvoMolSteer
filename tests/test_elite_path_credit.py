import numpy as np
import pandas as pd
import pytest
from evomolsteer.continuous.elite_path_credit import terminal_ancestors, credit_batch

def fixture():
    selected = np.array([[0,0,2],[0,1,2]])
    nodes = pd.DataFrame({'node_id':[0,2,4,6,8,10], 'step':[0,0,0,1,1,1],
                          'slot':[0,1,2,0,1,2], 'representation':[0]*6})
    m = pd.DataFrame({'slot':[0,1,2], 'pic50_on_rescore':[8.5,8.5,7.0],
                      'valid_connected':[True,True,True], 'smiles':['CC','CC','CN']})
    return selected,nodes,m

def test_backward_identity_and_extinction_censoring():
    selected,nodes,m=fixture()
    assert terminal_ancestors(selected).tolist()==[[0,0,2],[0,1,2],[0,1,2]]
    c=credit_batch(nodes,selected,m,8.2).set_index('node_id')
    assert c.loc[0,'n_descendants']==2 and c.loc[0,'n_unique_graphs']==1
    assert c.loc[0,'n_elite_graphs']==1
    assert not c.loc[2,'future_observed'] and np.isnan(c.loc[2,'terminal_max_pic50'])
    assert c.loc[4,'future_observed'] and not c.loc[4,'has_observed_elite']

def test_invalid_is_observed_but_not_affinity_negative():
    selected,nodes,m=fixture();m.loc[2,'valid_connected']=False
    c=credit_batch(nodes,selected,m,8.2).set_index('node_id')
    assert c.loc[4,'future_observed'] and c.loc[4,'n_valid_descendants']==0
    assert np.isnan(c.loc[4,'terminal_max_pic50'])

def test_clone_invariance_of_terminal_utility():
    selected,nodes,m=fixture()
    first=credit_batch(nodes,selected,m,8.2)
    m.loc[1,'pic50_on_rescore']=8.4
    second=credit_batch(nodes,selected,m,8.2)
    assert first.iloc[0].terminal_mean_unique_pic50==second.iloc[0].terminal_mean_unique_pic50

def test_reject_missing_terminal_slots_and_string_validity():
    selected,nodes,m=fixture()
    with pytest.raises(ValueError):credit_batch(nodes,selected,m.iloc[:2],8.2)
    m['valid_connected']=['True']*3
    with pytest.raises(ValueError):credit_batch(nodes,selected,m,8.2)
