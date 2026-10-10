"""Register a reproducible extension calculation with its actual input files."""
import argparse
import gzip
import json
from pathlib import Path

from evomolsteer.io import read_json, write_json
from evomolsteer.trajectory_source import pocket_input_path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'


def prepare(kind, evidence, key, labels, tail_weight=0., tolerance=.25):
    evidence, labels = [(ROOT/Path(v)).resolve() for v in (evidence, labels)]
    packet = read_json(evidence)
    reference = Path(packet['reference_path'])
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    dataset = ROOT/'data/raw/ck2_clk3_lineage_20261003'
    campaign = 'main1000_w050'
    source = dataset/'results'/campaign
    metrics = ROOT/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'
    output = DOC/'mining'/key
    if not output.is_relative_to(DOC/'mining') or '/' in key or '\\' in key:
        raise ValueError('One local immutable output name required')
    args = ['--dataset', str(dataset), '--campaign', campaign, '--labels', str(labels),
            '--evidence', str(evidence), '--output', str(output)]
    inputs = [labels, evidence, reference,
              *[dataset/s['path'] for s in ref['sources']]]
    files = [output/'evidence.json']
    if kind == 'matched_geometry':
        tool = 'outcome_conditioned_geometry'
        args += ['--score-tolerance', str(tolerance)]
        inputs += [source/'config.json', pocket_input_path(dataset, 'target_protein'),
                   *[source/f'frame_batch_{b:03d}.json' for b in packet['donor_batches']]]
        files += [output/'batch_event_effects.parquet', output/'matched_support.csv', output/'trends.json']
    elif kind == 'matched_field':
        tool = 'outcome_matched_reference'
        args += ['--metrics', str(metrics), '--score-tolerance', str(tolerance)]
        inputs += [metrics, source/'config.json',
                   *[source/f'frame_batch_{b:03d}.json' for b in packet['donor_batches']]]
        files += [output/'reference.json.gz', output/'branch_support.csv']
    elif kind == 'pooled_credit':
        tool = 'outcome_alias_credit'
        args += ['--metrics', str(metrics), '--tail-weight', str(tail_weight)]
        inputs += [metrics]
        files += [output/'reference.json.gz', output/'alias_credit.csv']
    else:
        raise ValueError('Registered sequential extension required')
    target = DOC/'tool_plans'/(key+'.json')
    if target.exists():
        raise FileExistsError(target)
    write_json(target, {'tool_id': tool, 'arguments': args,
        'input_files': [str(p.resolve()) for p in dict.fromkeys(inputs)],
        'output_files': list(map(str, files))})
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['matched_geometry', 'matched_field', 'pooled_credit'], required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--labels', default='docs/experiments/terminal_outcome15_20261010/mining/mean/ancestor_outcomes.parquet')
    p.add_argument('--tail-weight', type=float, default=0.)
    p.add_argument('--score-tolerance', type=float, default=.25)
    a = p.parse_args()
    prepare(a.kind, a.evidence, a.key, a.labels, a.tail_weight, a.score_tolerance)
