"""Compare retained mechanism responses without reopening structure trajectories.

Explicit input/output dataset directories; only compact report tables are read.
Dose summaries are proxies from batch-mean RMS, not molecular control energies.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.generation.path_evaluation import paired_effect


def summarize(input_dataset, output_dataset):
    source, output = Path(input_dataset).resolve(), Path(output_dataset).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError('Keep derived summaries outside their input dataset')
    output.mkdir(parents=True, exist_ok=True)
    records, inputs = [], {}
    for folder in sorted(source.glob('round[0-9][0-9]')):
        if not (folder / 'retention.json').exists():
            continue
        names = ['comparison.json', 'implementation_feedback.json', 'retention.json',
                 'flow_response.parquet', 'candidate_metrics.csv', 'inference_config/reward_program.json']
        for name in names:
            path = folder / name
            if path.is_file():
                inputs[str(path)] = digest(path)
        report = read_json(folder / 'comparison.json')
        feedback = read_json(folder / 'implementation_feedback.json')
        program = read_json(folder / 'inference_config/reward_program.json')
        window = np.asarray(program['window'], float)
        if window.shape != (2,) or not window[0] < window[1]:
            raise ValueError('Each source must declare its actual guidance window')
        number = int(folder.name[-2:])
        response = folder / 'flow_response.parquet'
        doses = []
        if response.exists():
            table = pd.read_parquet(response)
            table = table[table.arm.eq('gradient')]
            if table.duplicated(['batch', 'time']).any():
                raise ValueError('Batch-time response rows must be unique')
            tolerance = 2e-7
            if ((table.time < window[0] - tolerance) | (table.time >= window[1] + tolerance)).any():
                raise ValueError('Response includes times outside learned support')
            for batch, rows in table.groupby('batch', sort=True):
                rows = rows.sort_values('time')
                if 'injection_rms_A' not in rows:
                    continue
                injection = rows.injection_rms_A.to_numpy(float)
                duration = rows.state_time.to_numpy(float) - rows.time.to_numpy(float)
                if not np.isfinite(injection).all() or (injection < 0).any() or (duration <= 0).any():
                    raise ValueError('Finite nonnegative RMS and positive native durations required')
                doses.append({'batch': int(batch), 'observed_steps': len(rows),
                              'mean_RMS_path_proxy_A': float(injection.sum()),
                              'mean_RMS_squared_action_proxy_A2_per_time': float(np.sum(injection ** 2 / duration)),
                              'mean_native_flow_RMS_path_proxy_A': float(rows.predictive_flow_rms_A.sum())})
        gradient = report['results']['gradient']
        coordinate = [v['mean_RMS_A'] for v in feedback.get('window_differences_against_paired_R26', [])
                      if v['arm'] == 'gradient']
        record = {'round': number, 'reward_view': program['reward_view'], 'window': window.tolist(),
                  'mechanism_parameters': {k: program[k] for k in ['innovation', 'branch_mixture', 'flow_control', 'native_rms_ratio'] if k in program},
                  'inference_commit': read_json(folder / 'retention.json')['inference_commit'],
                  'implementation_passed': feedback.get('implementation_passed'),
                  'exact_null_parity': feedback.get('null_control_parity'),
                  'paired_window_RMS_A': float(np.mean(coordinate)) if coordinate else None,
                  'mean_pic50': gradient['all_mean_pic50'],
                  'versus_R26': report.get('versus_gradient'),
                  'versus_native': report.get('versus_unguided'),
                  'batch_dose_proxies': doses,
                  'mean_RMS_path_proxy_A': float(np.mean([d['mean_RMS_path_proxy_A'] for d in doses])) if doses else None}
        records.append(record)
    if not records:
        raise ValueError('No fully retained rounds found')
    by_round = {r['round']: r for r in records}
    contrasts = []
    for test, control, label in [(9, 10, 'natural_branch_positive_minus_reversed'),
                                (22, 23, 'regional_positive_minus_reversed'),
                                (18, 21, 'VJP_gain_schedule_minus_uniform_dose_proxy')]:
        if test not in by_round or control not in by_round:
            continue
        a, b = by_round[test], by_round[control]
        if a['versus_R26'] is None or b['versus_R26'] is None:
            continue
        left = pd.read_csv(source / f'round{test:02d}' / 'candidate_metrics.csv')
        right = pd.read_csv(source / f'round{control:02d}' / 'candidate_metrics.csv')
        paired = paired_effect(left[left.arm.eq('gradient')], right[right.arm.eq('gradient')])
        dose_a, dose_b = a['mean_RMS_path_proxy_A'], b['mean_RMS_path_proxy_A']
        contrasts.append({'contrast': label, 'rounds': [test, control],
                          'batch_delta_pic50': paired['batch_means'],
                          'mean_delta_pic50': paired['paired_mean_pic50'],
                          'dose_proxy_relative_difference': (dose_a / dose_b - 1) if dose_a is not None and dose_b else None,
                          'interpretation': 'Adaptive screening contrast, not independent confirmation or a causal effect estimate'})
    write_json(output / 'mechanism_comparisons.json', {
        'schema_version': 'flowcompat-compact-mechanism-comparison-1.0', 'rounds': records,
        'paired_mechanism_contrasts': contrasts,
        'dose_semantics': 'Sums of batch-mean RMS and squared batch-mean RMS/dt; not mean per-particle squared control energy. Same parameter budget need not imply same injected dose.',
        'input_files': inputs})
    pd.DataFrame([{k: r[k] for k in ['round', 'reward_view', 'inference_commit', 'implementation_passed',
                                   'exact_null_parity', 'paired_window_RMS_A', 'mean_pic50', 'mean_RMS_path_proxy_A']}
                  for r in records]).to_csv(output / 'mechanism_summary.csv', index=False)
    write_json(output / 'manifest.json', {'input_dataset': str(source), 'rounds': len(records),
               'input_files': inputs, 'script_sha256': digest(__file__),
               'outputs': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}})
    return {'retained_rounds': len(records), 'contrasts': len(contrasts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    args = parser.parse_args()
    print(summarize(args.input_dataset, args.output_dataset))
