import numpy as np
import pytest
import torch

from evomolsteer.elite_alignment.initial_audit import parent_path, tree_identity


def test_trace_true_root_not_final_slot():
    selected=np.array([[2,2,0],[1,0,1],[2,0,1]])
    before,after=parent_path(selected,0)
    assert before.tolist()==[2,1,2]
    assert after.tolist()==[1,2,0]
    assert before[0]!=0


def test_identity_retains_exact_coordinates_and_categories():
    state={'prior':{'coords':torch.tensor([0.,-0.]),'atomics':torch.tensor([1,0])},'scale':1.}
    original=tree_identity(state)
    assert original==tree_identity({'scale':1.,'prior':dict(reversed(list(state['prior'].items())))})
    state['prior']['coords'][1]=0.
    assert original!=tree_identity(state)


def test_no_invalid_parent_indices():
    with pytest.raises(ValueError):parent_path(np.array([[0,2]]),0)
    with pytest.raises(ValueError):parent_path(np.array([[0.,1.]]),0)


def test_historical_transport_overlap_is_not_extra_samples(tmp_path):
    import pandas as pd
    from evomolsteer.elite_alignment.pool_comparison import unique_rows
    a=tmp_path/'a.csv';b=tmp_path/'b.csv'
    pd.DataFrame({'batch':[1,1],'slot':[0,1],'pic50_on_rescore':[7.,8.]}).to_csv(a,index=False)
    pd.DataFrame({'batch':[1,2],'slot':[1,0],'pic50_on_rescore':[8.,9.]}).to_csv(b,index=False)
    assert len(unique_rows([a,b]))==3
    pd.DataFrame({'batch':[1],'slot':[1],'pic50_on_rescore':[8.1]}).to_csv(b,index=False)
    with pytest.raises(ValueError):unique_rows([a,b])
