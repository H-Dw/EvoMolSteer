"""Read-only comparison of source-bound checkpoint outcomes; never run inference.

Usage: python scripts/report_post_m2_checkpoints.py --manifest INPUT --output DIR
Explicit qualification and a common tail threshold prevent mixing raw maxima,
survival, suffix probes and incompatible historical hit fractions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_case(root, case, sources, threshold):
    frames = []
    for index, relative in enumerate(case['files']):
        path = root / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        if case['format'] == 'records_json':
            frame = pd.DataFrame(json.loads(raw))
            # Slot numbers can recur across disjoint source batches.
            if 'batch' not in frame:
                frame['batch'] = index
        elif path.suffix == '.parquet':
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        for key, value in case.get('filters', {}).items():
            frame = frame.loc[frame[key].eq(value)]
        if case.get('batches') is not None:
            frame = frame.loc[frame.batch.isin(case['batches'])]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if case['format'] == 'summary_csv':
        if len(frame) != 1:
            raise ValueError(f"Summary selection must be unique: {case['id']}")
        row = frame.iloc[0]
        n = int(row[case['columns']['n']])
        value = {key: float(row[column]) for key, column in case['columns'].items()}
        value['high_count'] = int(round(value['tail_rate'] * n))
        value['scored'] = n  # All selected historical tables report full coverage.
        if not case.get('full_score_coverage_verified'):
            raise ValueError('Summary score coverage must be explicitly verified')
    else:
        if frame.empty:
            raise ValueError(f"No observations: {case['id']}")
        if frame.duplicated(['batch', 'slot']).any():
            raise ValueError(f"Duplicate attempt keys: {case['id']}")
        score = pd.to_numeric(frame[case['score_column']], errors='coerce')
        quality = frame[case['quality_column']]
        if quality.isna().any() or not quality.isin([True, False, 0, 1]).all():
            raise ValueError('Explicit boolean qualification required')
        quality = quality.astype(bool)
        finite = pd.Series(np.isfinite(score), index=frame.index)
        high = quality & finite & score.ge(threshold)
        n = len(frame)
        keys = frame[['batch', 'slot']].astype('int64').sort_values(['batch', 'slot'])
        key_bytes = keys.to_csv(index=False, lineterminator='\n').encode('utf-8')
        value = {'n': n, 'scored': int(finite.sum()),
                 'mean': float(score.loc[finite].mean()),
                 'best': float(score.loc[quality & finite].max()),
                 'tail_rate': float(high.sum() / n), 'high_count': int(high.sum()),
                 'attempt_keys_sha256': hashlib.sha256(key_bytes).hexdigest()}
        value['n_qualified'] = int(quality.sum())
    return {**case, **value, 'n': n, 'threshold': threshold}


def run(root, manifest_path, output):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    sources = {}
    rows = [read_case(root, case, sources, manifest['tail_threshold'])
            for case in manifest['cases']]
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Unique case IDs required')
    epsilon = manifest['comparison_tolerance']
    historical = by_id['historical_Steer']
    historical_m2 = by_id['M2_historical3000']
    for row in rows:
        row['score_coverage'] = row['scored'] / row['n']
        for label, reference in [('Steer', historical), ('historical_M2', historical_m2)]:
            row[f'mean_minus_{label}'] = row['mean'] - reference['mean']
            row[f'best_minus_{label}'] = row['best'] - reference['best']
            row[f'tail_pp_minus_{label}'] = 100 * (row['tail_rate'] - reference['tail_rate'])
            row[f'both_above_{label}'] = (row['mean'] > reference['mean'] + epsilon
                                        and row['best'] > reference['best'] + epsilon)
        if row.get('matched_M2'):
            reference = by_id[row['matched_M2']]
            if row['n'] != reference['n']:
                raise ValueError('Matched comparison requires equal attempted counts')
            if row.get('attempt_keys_sha256') and reference.get('attempt_keys_sha256'):
                if row['attempt_keys_sha256'] != reference['attempt_keys_sha256']:
                    raise ValueError('Matched comparison requires the same batch/slot keys')
            row['delta_mean_M2'] = row['mean'] - reference['mean']
            row['delta_best_M2'] = row['best'] - reference['best']
            row['delta_tail_pp_M2'] = 100 * (row['tail_rate'] - reference['tail_rate'])
            row['improves_M2_observed'] = ','.join(
                key for key, difference in [('mean', row['delta_mean_M2']),
                                            ('best', row['delta_best_M2']),
                                            ('tail', row['delta_tail_pp_M2'])]
                if difference > epsilon)
    output.mkdir(parents=True, exist_ok=True)
    table = [{k: v for k, v in row.items() if k not in ('columns', 'files', 'filters')}
             for row in rows]
    pd.DataFrame(table).to_csv(output / 'checkpoints.csv', index=False, lineterminator='\n')
    result = {'schema_version': 'post-m2-checkpoint-audit-1.0',
              'status': 'READ_ONLY_ANALYSIS', 'new_generation': 0, 'new_API_calls': 0,
              'manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'tail_threshold': manifest['tail_threshold'],
              'comparison_tolerance': epsilon, 'sources_sha256': sources,
              'limits': manifest['limits'], 'rows': rows}
    (output / 'results.json').write_bytes(
        (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'conditions': len(rows), 'source_files': len(sources),
                      'matched_improvements': [r['id'] for r in rows
                                               if r.get('improves_M2_observed')],
                      'new_programs_both_above_Steer': [r['id'] for r in rows
                          if r['group'] not in ('reference', 'M2_reuse')
                          and r['both_above_Steer']]}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.manifest, args.output)
