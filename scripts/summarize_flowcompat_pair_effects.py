"""Describe paired terminal-score heterogeneity without changing selection.

This post-generation report reads compact metrics, not molecular trajectories.
Concentration is descriptive and is not a new gate or independent sample count.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json


def summarize(input_dataset, output_dataset, tolerance=1e-6):
    source, output = Path(input_dataset).resolve(), Path(output_dataset).resolve()
    if source == output or output.is_relative_to(source) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Separate output dataset and nonnegative finite tolerance required')
    output.mkdir(parents=True, exist_ok=True)
    records, inputs = [], {}
    for folder in sorted(source.glob('round[0-9][0-9]')):
        if not (folder / 'retention.json').exists():
            continue
        number = int(folder.name[-2:])
        candidate_path = folder / 'candidate_metrics.csv'
        candidate = pd.read_csv(candidate_path)
        if number in [1, 2, 25, 27, 29]:
            control_path, control = candidate_path, candidate[candidate.arm.eq('unguided')]
            contrast = 'R26_minus_native'
        else:
            control_number = number - 1 if number in [26, 28, 30] else 1
            control_path = source / f'round{control_number:02d}/candidate_metrics.csv'
            control = pd.read_csv(control_path)
            control = control[control.arm.eq('gradient')]
            contrast = 'candidate_minus_R26'
        left = candidate[candidate.arm.eq('gradient')]
        inputs[str(candidate_path)] = digest(candidate_path)
        inputs[str(control_path)] = digest(control_path)
        pairs = left.merge(control, on=['batch', 'slot'], suffixes=('_candidate', '_control'), validate='one_to_one')
        if len(pairs) != len(left) or len(pairs) != len(control):
            raise ValueError('All attempted paired samples must be covered')
        delta = (pairs.pic50_on_rescore_candidate - pairs.pic50_on_rescore_control).to_numpy(float)
        if not np.isfinite(delta).all():
            raise ValueError('Missing/nonfinite predictions cannot be silently excluded')
        magnitude = np.abs(delta)
        positive = delta[delta > tolerance]
        record = {
            'round': number, 'contrast': contrast, 'attempts': len(delta),
            'generation_batches': int(pairs.batch.nunique()),
            'mean_delta_pic50': float(delta.mean()), 'median_delta_pic50': float(np.median(delta)),
            'p10_delta_pic50': float(np.quantile(delta, .10)),
            'p90_delta_pic50': float(np.quantile(delta, .90)),
            'minimum_delta_pic50': float(delta.min()), 'maximum_delta_pic50': float(delta.max()),
            'positive_fraction': float((delta > tolerance).mean()),
            'negative_fraction': float((delta < -tolerance).mean()),
            'within_tolerance_fraction': float((magnitude <= tolerance).mean()),
            'top5_absolute_effect_share': float(np.sort(magnitude)[-min(5, len(delta)):].sum() / magnitude.sum()) if magnitude.sum() else 0.,
            'positive_effect_participation_ratio': float(positive.sum() ** 2 / (positive ** 2).sum()) if len(positive) else 0.,
        }
        records.append(record)
    if not records:
        raise ValueError('No retained paired comparisons')
    pd.DataFrame(records).to_csv(output / 'paired_effect_distribution.csv', index=False)
    write_json(output / 'interpretation.json', {
        'schema_version': 'flowcompat-paired-effect-heterogeneity-1.0',
        'numerical_tolerance_pic50': tolerance,
        'interpretation': 'Descriptive matched final prediction differences, not causal coordinate effects',
        'limitations': [
            'No chemical-graph filtering, affinity fitting, new p-values or candidate-selection gate.',
            'Participation ratio describes effect concentration; it is not an independent sample size.',
            'Screening repeats the same two initial batches and is adaptively examined.',
            'Per-round held-out summaries are not the pooled six-batch confirmation test.',
            'All attempts, including invalid molecules, remain in the score comparison.',
        ],
    })
    write_json(output / 'manifest.json', {
        'input_dataset': str(source), 'input_files': inputs, 'script_sha256': digest(__file__),
        'outputs': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'},
    })
    return {'retained_comparisons': len(records)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    print(summarize(args.input_dataset, args.output_dataset, args.tolerance))
