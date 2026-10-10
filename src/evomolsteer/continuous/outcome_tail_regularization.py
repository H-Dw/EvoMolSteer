"""Count-regularized observed tail credit; no probability-calibration claim."""
import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import clean, digest, read_json, write_json


def regularized_fraction(raw, observed_n, batch_fraction, strength):
    if isinstance(observed_n, bool) or int(observed_n) != observed_n or observed_n <= 0:
        raise ValueError('Observed nonempty descendant support required; extinct futures remain unknown')
    if not all(math.isfinite(float(v)) for v in (raw, batch_fraction, strength)):
        raise ValueError('Finite fraction and shrinkage required')
    if not 0 <= raw <= 1 or not 0 <= batch_fraction <= 1 or not 0 <= strength <= 10:
        raise ValueError('Bounded fractions and shrinkage required')
    return (observed_n * raw + strength * batch_fraction) / (observed_n + strength)


def build(evidence, metrics, output, strength=2.):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    packet = read_json(evidence)
    if digest(metrics) != packet['metrics_sha256']:
        raise ValueError('The original common final evaluation is required')
    ref_path = Path(packet['reference_path'])
    if digest(ref_path) != packet['reference_sha256']:
        raise ValueError('Original outcome reference changed')
    ref = copy.deepcopy(json.loads(gzip.decompress(ref_path.read_bytes())))
    weight = float(ref['teacher_selection']['tail_weight'])
    if ref['teacher_selection']['prior_mode'] != 'copy_family_mean_plus_tail' or not 0 <= weight <= 1:
        raise ValueError('Unregularized, explicitly weighted pooled tail library required')
    table = pd.read_csv(metrics)
    if table.valid_connected.dtype != bool or table.duplicated(['batch', 'slot']).any():
        raise ValueError('Typed and uniquely aligned final outcomes required')
    threshold = packet['threshold_pic50']
    batch_rates, batch_counts = {}, {}
    for batch in packet['donor_batches']:
        group = table[table.batch.eq(batch)]
        if not len(group):
            raise ValueError('Donor final outcomes missing')
        elite = group.valid_connected & np.isfinite(group.pic50_on_rescore) & group.pic50_on_rescore.ge(threshold)
        batch_rates[batch] = float(elite.sum() / len(group))
        batch_counts[batch] = len(group)
    rows = []
    for frame in ref['frames']:
        for k, batch in enumerate(frame['teacher_batches']):
            credit = frame['teacher_outcomes'][k]['alias_family_credit']
            mean, raw, count = credit['terminal_mean'], credit['tail_fraction'], credit['observed_n']
            if count > batch_counts[batch]:
                raise ValueError('Family support exceeds observed batch outcomes')
            if not math.isfinite(mean):
                raise ValueError('Teacher must have observed valid final quality')
            if not np.isclose(frame['teacher_scores'][k], mean + weight * raw, atol=1e-12, rtol=0):
                raise ValueError('Source teacher is not the unregularized mean-plus-tail utility')
            value = regularized_fraction(raw, count, batch_rates[batch], strength)
            old = frame['teacher_scores'][k]
            frame['teacher_scores'][k] = mean + weight * value
            frame['teacher_outcomes'][k]['tail_regularization'] = {
                'raw_fraction': raw, 'regularized_fraction': value, 'observed_n': count,
                'batch_fraction': batch_rates[batch], 'strength': strength,
                'semantics': 'Count shrinkage of a selected-descendant observation; not a native probability posterior'}
            rows.append({'batch': batch, 'time': frame['time'], 'teacher': k,
                         'observed_n': count, 'valid_fraction': credit['valid_fraction'],
                         'raw_fraction': raw, 'regularized_fraction': value,
                         'batch_fraction': batch_rates[batch], 'strength': strength,
                         'mean_component_change': 0., 'utility_change': frame['teacher_scores'][k] - old})
    ref['teacher_selection'].update(prior_mode='copy_family_mean_plus_regularized_tail',
                                   tail_shrinkage=strength, point_selection_unchanged=True)
    ref['tail_regularization'] = {'source_reference_sha256': packet['reference_sha256'],
        'formula': '(observed_n * raw_tail_fraction + strength * donor_batch_tail_fraction) / (observed_n + strength)',
        'denominator': 'All observed final slots, including invalid decoding; donor baseline uses the same denominator',
        'limitation': 'Related descendants are not independent; shrinkage is a heuristic, not Bayesian/native-success calibration'}
    out.mkdir(parents=True)
    diagnostic = pd.DataFrame(rows)
    diagnostic.to_csv(out/'support_credit.csv', index=False)
    target = out/'reference.json.gz'
    target.write_bytes(gzip.compress(json.dumps(clean(ref), separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    packet['evidence_items'].append({'id': 'outcome/tail_regularization',
        'formula': ref['tail_regularization']['formula'], 'strength': strength,
        'teacher_records': len(rows), 'changed_records': int(np.abs(diagnostic.utility_change).gt(1e-12).sum()),
        'singleton_records': int(diagnostic.observed_n.eq(1).sum()),
        'batch_equal_utility_change': float(diagnostic.groupby('batch').utility_change.mean().mean()),
        'batch_equal_raw_fraction': float(diagnostic.groupby('batch').raw_fraction.mean().mean()),
        'batch_equal_regularized_fraction': float(diagnostic.groupby('batch').regularized_fraction.mean().mean()),
        'mean_component_change': 0., 'point_selection_unchanged': True,
        'denominator': ref['tail_regularization']['denominator'],
        'limitation': ref['tail_regularization']['limitation']})
    packet.update(reference_path=str(target.resolve()), reference_sha256=digest(target),
                  tail_regularization_source_sha256=digest(__file__))
    write_json(out/'evidence.json', packet)
    return packet
