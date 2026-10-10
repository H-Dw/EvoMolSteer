"""Compact observed-dose and paired-quality evidence for endpoint combinations.

Reuses local deterministic diagnostics; no affinity calls or new generated data.
Only batch/condition summaries are retained, never expanded feature trajectories.
"""
import argparse
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json, write_json, digest
from analyze_dependency_study import metrics, paired
from analyze_guidance_directions import summarize, FIELDS


def analyze(quality_manifest, direction_manifest, output):
    quality_plan, direction_plan = read_json(quality_manifest), read_json(direction_manifest)
    frames, outcomes = {}, {}
    for item in quality_plan['cohorts']:
        frames[item['name']], outcomes[item['name']] = metrics(item['metrics'], item['arm'])
    comparisons = {}
    for left, right in [('I3', 'R26'), ('I3', 'native'), ('I3', 'I4'), ('M2', 'M1'), ('M2', 'I3')]:
        merged = frames[left].merge(frames[right], on=['batch', 'slot'], validate='one_to_one', suffixes=('_l', '_r'))
        if len(merged) != len(frames[left]) or len(merged) != len(frames[right]) or not (merged.seed_l == merged.seed_r).all():
            raise ValueError('Incomplete paired seed identities')
        comparisons[left + '_minus_' + right] = {
            **paired(frames[left], frames[right]),
            'slot_positive_fraction': float((merged.pic50_on_rescore_l > merged.pic50_on_rescore_r).mean())}
    directions = {}
    for item in direction_plan['cohorts']:
        result = summarize(Path(item['root']), item['arm'])
        if len({b['active_steps'] for b in result['batches']}) != 1:
            raise ValueError('Direction averages require common step coverage')
        result['equal_batch_aggregate'] = {
            key: {metric: float(np.mean([b[key][metric] for b in result['batches']]))
                  for metric in ['mean', 'negative_fraction']} for key in FIELDS}
        directions[item['name']] = result
    write_json(output, {'schema_version': 'static-combo-observed-evidence-1.0',
                       'outcomes': outcomes, 'paired_comparisons': comparisons, 'directions': directions,
                       'manifest_sha256': {'quality': digest(quality_manifest), 'direction': digest(direction_manifest)},
                       'limits': 'Observed two-batch screen, not independent validation or proof of instruction causality. Equal batch aggregates presume matched molecule counts, as frozen here. Different reward values/norms are not affinity measures.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-manifest', required=True)
    parser.add_argument('--direction-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(Path(args.quality_manifest), Path(args.direction_manifest), Path(args.output))
