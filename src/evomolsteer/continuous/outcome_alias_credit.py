"""Pool descendant distributions of exact-copy states before assigning priors.

This calculator changes credit, not point-cloud selection. Keeping coordinates
fixed isolates lucky-copy prior bias from discovery of different teachers.
"""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import clean, digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths


def pooled_outcome(frame_labels, parents, slot, metrics, threshold):
    group = parents == parents[slot]
    ids = sorted({int(v) for s in frame_labels.loc[group, 'terminal_slot_ids']
                  for v in str(s).split(',') if v and v != 'nan'})
    observed = metrics.set_index('slot').loc[ids]
    good = observed[observed.valid_connected & observed.pic50_on_rescore.notna()]
    graphs = good.groupby('smiles', sort=True).pic50_on_rescore.mean()
    if not len(graphs):
        raise ValueError('Teacher copy family has no observed valid final quality')
    return {'terminal_mean': float(graphs.mean()), 'terminal_p75': float(graphs.quantile(.75)),
        'tail_fraction': float((good.pic50_on_rescore >= threshold).sum()/len(ids)),
        'valid_fraction': len(good)/len(ids), 'observed_n': len(ids), 'unique_graph_n': len(graphs),
        'copy_n': int(group.sum()), 'terminal_slot_ids': ','.join(map(str, ids))}


def pool(dataset, campaign, labels, metrics, evidence, output, tail_weight=0.):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    if not np.isfinite(tail_weight) or tail_weight < 0:
        raise ValueError('Nonnegative final-tail utility weight required')
    packet = read_json(evidence)
    if digest(metrics) != packet['metrics_sha256']:
        raise ValueError('Common final evaluation metrics changed')
    reference = copy.deepcopy(json.loads(gzip.decompress(Path(packet['reference_path']).read_bytes())))
    all_labels, all_metrics = pd.read_parquet(labels), pd.read_csv(metrics)
    source = Path(dataset)/'results'/campaign
    paths = {int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name == 'single'}
    rows = []
    for batch in packet['donor_batches']:
        batch_metrics = all_metrics[all_metrics.batch == batch]
        expected = next(v['sha256'] for v in reference['sources'] if v['batch'] == batch)
        if digest(paths[batch]) != expected:
            raise ValueError('Copy-family ancestry is not the bound original source')
        with open_trajectory(paths[batch]) as tr:
            selected = np.asarray(tr['selected_indices'])
            for frame in reference['frames']:
                labels_frame = all_labels[(all_labels.batch == batch)&(np.abs(all_labels.score_time-frame['time']) <= 2e-6)].sort_values('slot').reset_index(drop=True)
                if not len(labels_frame):
                    raise ValueError('Teacher event outside supplied observed label window')
                step = int(labels_frame.step.iloc[0])
                parents = selected[step-1] if step else np.arange(len(labels_frame))
                for k in np.flatnonzero(np.array(frame['teacher_batches']) == batch):
                    slot = frame['teacher_slots'][k]
                    # Parent identity is an exact-copy group only if the
                    # pre-forward state really agrees, including chemistry.
                    group = parents == parents[slot]
                    for field in ('current_coords', 'current_atomics', 'current_bonds', 'current_charges'):
                        values = np.asarray(tr[field][step])[group]
                        if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
                            raise ValueError('Parent group is not an exact copied generation state')
                    credit = pooled_outcome(labels_frame, parents, slot, batch_metrics, packet['threshold_pic50'])
                    old_score = float(frame['teacher_scores'][k])
                    new_score = credit['terminal_mean']+tail_weight*credit['tail_fraction']
                    frame['teacher_scores'][k] = new_score
                    frame['teacher_outcomes'][k]['alias_family_credit'] = credit
                    rows.append({'batch': batch, 'time': frame['time'], 'teacher': int(k),
                        'slot': slot, 'parent_slot': int(parents[slot]), 'old_node_quality': old_score,
                        'pooled_quality': new_score, 'mean_credit_change': credit['terminal_mean']-old_score,
                        **{key:credit[key] for key in ('copy_n', 'observed_n', 'unique_graph_n', 'tail_fraction', 'valid_fraction')}})
    reference.update(reference_variant='decoded-copy-family-outcome-1.0',
        label_semantics='Decoded final outcomes pooled over exactly copied coordinate/chemical states; per-graph mean then equal-graph mean; no best-copy score',
        teacher_selection={**reference['teacher_selection'], 'prior_mode': 'copy_family_mean_plus_tail', 'tail_weight': tail_weight,
            'point_selection_unchanged': True})
    out.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(out/'alias_credit.csv', index=False)
    path = out/'reference.json.gz'
    path.write_bytes(gzip.compress(json.dumps(clean(reference), separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    batch = table.groupby('batch').mean_credit_change.mean()
    packet['evidence_items'] += [{'id': 'outcome/alias_credit', 'teacher_records': len(table),
        'mean_quality_change_batch_equal': float(batch.mean()),
        'quality_change_quantiles': table.mean_credit_change.quantile([0,.25,.5,.75,1]).tolist(),
        'copy_n_quantiles': table.copy_n.quantile([.25,.5,.75]).tolist(),
        'tail_weight': tail_weight, 'coordinate_selection_unchanged': True,
        'semantics': 'Observed exact-state copy-family distribution, not native success probability; no best-child maximum'}]
    packet.update(reference_path=str(path.resolve()), reference_sha256=digest(path), mode='copy_family_distribution',
        alias_source_sha256=digest(__file__), alias_labels_sha256=digest(labels), label_semantics=reference['label_semantics'])
    write_json(out/'evidence.json', packet)
    return packet
