import numpy as np
import pandas as pd
from evomolsteer.continuous.outcome_alias_credit import pooled_outcome


def test_copy_credit_pools_graphs_not_lucky_copy_maximum():
    labels = pd.DataFrame({'terminal_slot_ids': ['0,1', '2', '', '3']})
    metrics = pd.DataFrame({'slot': range(4), 'valid_connected': [True]*4,
                           'smiles': ['same','same','other','last'], 'pic50_on_rescore': [9., 7., 5., 8.]})
    result = pooled_outcome(labels, np.array([0,0,0,1]), 0, metrics, 8.5)
    assert result['terminal_mean'] == 6.5
    assert result['copy_n'] == result['observed_n'] == 3
    assert result['tail_fraction'] == 1/3
    assert result['unique_graph_n'] == 2
