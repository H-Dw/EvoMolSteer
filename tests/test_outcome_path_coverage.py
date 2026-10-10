import gzip
import json
import numpy as np
import pandas as pd
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.continuous.outcome_path_coverage import coverage


def test_low_scoring_copy_has_same_observed_geometry_coverage(tmp_path):
    folder = tmp_path/'results/campaign/single/batch_000'
    folder.mkdir(parents=True)
    path = folder/'trajectory.npz'
    xyz = np.zeros((2, 3, 1, 3))
    xyz[:, 2] = 2
    np.savez(path, selected_indices=np.array([[0, 0, 2], [0, 1, 2]]),
        current_coords=xyz, current_atomics=np.ones((2, 3, 1), int),
        current_bonds=np.zeros((2, 3, 1, 1), int), current_charges=np.zeros((2, 3, 1), int))
    labels = tmp_path/'labels.parquet'
    pd.DataFrame({'batch': [0]*3, 'score_time': [.1]*3, 'step': [1]*3,
        'slot': [0, 1, 2], 'terminal_slot_ids': ['0', '1', '2']}).to_parquet(labels)
    metrics = tmp_path/'metrics.csv'
    pd.DataFrame({'batch': [0]*3, 'slot': [0, 1, 2], 'valid_connected': [True]*3,
        'pic50_on_rescore': [9., 7., 8.], 'smiles': ['elite', 'ordinary', 'other']}).to_csv(metrics, index=False)
    ref = tmp_path/'ref.json.gz'
    ref.write_bytes(gzip.compress(json.dumps({'sources': [{'batch': 0, 'sha256': digest(path)}],
        'frames': [{'time': .1, 'teacher_slots': [1], 'teacher_batches': [0]}]}).encode()))
    evidence = tmp_path/'evidence.json'
    write_json(evidence, {'reference_path': str(ref), 'metrics_sha256': digest(metrics),
        'donor_batches': [0], 'threshold_pic50': 8.5, 'evidence_items': []})
    # Slot 1 has only an ordinary descendant, yet its identical generation state
    # has the elite descendant through copy 0. Coverage is geometry, not reward.
    coverage(tmp_path, 'campaign', labels, metrics, evidence, tmp_path/'out')
    result = read_json(tmp_path/'out/evidence.json')['evidence_items'][0]
    assert result['batch_equal_event_mean_slot_coverage'] == 1.
    table = pd.read_csv(tmp_path/'out/coverage_by_batch_event.csv')
    assert table.observed_final_slots_covered.iloc[0] == 2
    assert table.teacher_copy_families.iloc[0] == 1
