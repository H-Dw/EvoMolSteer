"""Observed final-tail coverage of teacher geometries, without max-child labels."""
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths


def coverage(dataset, campaign, labels, metrics, evidence, output):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    packet = read_json(evidence)
    if digest(metrics) != packet['metrics_sha256']:
        raise ValueError('Common decoded evaluation required')
    ref = json.loads(gzip.decompress(Path(packet['reference_path']).read_bytes()))
    lab, m = pd.read_parquet(labels), pd.read_csv(metrics)
    paths = {int(p.parent.name.split('_')[1]): p for p in trajectory_paths(Path(dataset)/'results'/campaign)
             if p.parent.parent.name == 'single'}
    rows = []
    for batch in packet['donor_batches']:
        if digest(paths[batch]) != next(s['sha256'] for s in ref['sources'] if s['batch'] == batch):
            raise ValueError('Teacher ancestry source changed')
        finals = m[m.batch == batch].set_index('slot')
        good = finals[finals.valid_connected & finals.pic50_on_rescore.notna()]
        elite = good[good.pic50_on_rescore >= packet['threshold_pic50']]
        with open_trajectory(paths[batch]) as tr:
            selected = np.asarray(tr['selected_indices'])
            for frame in ref['frames']:
                event = lab[(lab.batch == batch)&(abs(lab.score_time-frame['time']) <= 2e-6)].sort_values('slot')
                if not np.array_equal(event.slot, np.arange(selected.shape[1])):
                    raise ValueError('Observed label event alignment required')
                step = int(event.step.iloc[0])
                parents = selected[step-1] if step else np.arange(len(event))
                teachers = np.array(frame['teacher_slots'])[np.array(frame['teacher_batches']) == batch]
                family = np.isin(parents, parents[teachers])
                # Copied states must have identical coordinates and chemistry.
                for p in np.unique(parents[teachers]):
                    for field in ('current_coords', 'current_atomics', 'current_bonds', 'current_charges'):
                        values = np.asarray(tr[field][step])[parents == p]
                        if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                            raise ValueError('Teacher copy family is not an identical state')
                ids = {int(v) for s in event.loc[family, 'terminal_slot_ids']
                       for v in str(s).split(',') if v and v != 'nan'}
                covered = elite[elite.index.isin(ids)]
                rows.append({'batch': batch, 'time': frame['time'], 'teachers': len(teachers),
                    'teacher_copy_families': len(np.unique(parents[teachers])),
                    'observed_final_slots_covered': len(ids), 'elite_final_slots': len(elite),
                    'elite_slots_covered': len(covered), 'elite_graphs': elite.smiles.nunique(),
                    'elite_graphs_covered': covered.smiles.nunique(),
                    'elite_slot_coverage': len(covered)/len(elite) if len(elite) else np.nan,
                    'elite_graph_coverage': covered.smiles.nunique()/elite.smiles.nunique() if len(elite) else np.nan})
    out.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/'coverage_by_batch_event.csv', index=False)
    average = table.groupby('time')[['elite_slot_coverage', 'elite_graph_coverage']].mean()
    average.to_csv(out/'coverage_by_event.csv')
    packet['evidence_items'].append({'id': 'outcome/teacher_tail_coverage',
        'batch_equal_event_mean_slot_coverage': float(average.elite_slot_coverage.mean()),
        'batch_equal_event_mean_graph_coverage': float(average.elite_graph_coverage.mean()),
        'first_event_graph_coverage': float(average.elite_graph_coverage.iloc[0]),
        'last_event_graph_coverage': float(average.elite_graph_coverage.iloc[-1]),
        'semantics': 'Union observed final identities of exact-copy teacher geometry families; no maximum-child reward or native-success inference'})
    packet['path_coverage_source_sha256'] = digest(__file__)
    write_json(out/'evidence.json', packet)
    return packet
