import pandas as pd
import numpy as np
from evomolsteer.continuous.outcome_summary import summarize
from evomolsteer.io import write_json


def test_tied_final_outcomes_are_not_simultaneously_positive_and_ordinary(tmp_path):
    labels = pd.DataFrame({'batch': [0]*4, 'step': [0]*4, 'slot': range(4), 'score_time': [0.]*4,
        'online_score': [1.,2.,3.,4.], 'terminal_mean': [7.,7.,7.,np.nan],
        'observed_n': [1,2,1,0], 'unique_graph_n': [1,1,1,0],
        'valid_fraction': [1.,1.,1.,np.nan], 'tail_fraction': [0.,0.,0.,np.nan]})
    path = tmp_path/'labels.parquet'
    labels.to_parquet(path, index=False)
    packet = tmp_path/'evidence.json'
    write_json(packet, {'evidence_items': [], 'score_window': [0,.5], 'window': [0,.51]})
    result = summarize(path, packet, tmp_path/'output')
    groups = result['evidence_items'][0]['batch_equal_groups']
    assert all(g['ancestor_fraction'] == 0 for g in groups if g['group'].endswith('final_high'))
    assert result['evidence_items'][1]['censored_ancestor_observations'] == 1
