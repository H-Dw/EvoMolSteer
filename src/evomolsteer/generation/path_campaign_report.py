"""Report-only synthesis of a sequential path campaign and frozen confirmation.

Read compact retained terminal records, not coordinates or retired trajectories.
Repeated adaptive hits are deduplicated for reporting, never gated in inference.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..io import digest, read_json, write_json, write_table
from .path_evaluation import paired_effect, summarize_tail


def verified_retention(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / 'retention.json')
    if manifest.get('status') != 'complete' or not manifest.get('files'):
        raise ValueError('Complete retained reports required')
    for entry in manifest['files']:
        path = (folder / entry['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or digest(path) != entry['sha256']:
            raise ValueError('Retained report checksum mismatch')
    return manifest


def qualified_summary(data, threshold):
    summary = summarize_tail(data, threshold)
    elite = data[data.valid_connected & np.isfinite(data.pic50_on_rescore)
                 & data.pic50_on_rescore.ge(threshold)]
    qualified = elite[elite.pb_fast_pass]
    energy = qualified[qualified.energy_status.eq('converged')].mmff_relief_per_heavy.dropna()
    summary.update(elite_pb_fast_n=len(qualified),
                   elite_pb_fast_unique_graphs=int(qualified.smiles.nunique()),
                   elite_pb_fast_yield=len(qualified) / len(data),
                   elite_pb_strain_converged_n=len(energy),
                   elite_pb_strain_median_per_heavy=float(energy.median()))
    return summary


def verify_initial_pair(candidate_execution, control_execution, candidate_arm='gradient', control_arm='unguided'):
    left = candidate_execution['initial_state_signatures']
    right = control_execution['initial_state_signatures']
    selected = [(key, value) for key, value in left.items() if key.startswith(candidate_arm + '/')]
    if not selected:
        raise ValueError('Missing candidate initial states')
    for key, value in selected:
        other = control_arm + '/' + key.split('/', 1)[1]
        if other not in right or right[other] != value:
            raise ValueError('Confirmation initial states differ')


def confirmation_pair(candidate, control):
    result = paired_effect(candidate, control)
    result['limitation'] = ('Independent frozen batch labels; few fixed-seed batches still limit uncertainty. '
                            'This is not measured affinity or a causal regional mechanism.')
    per_batch = []
    for batch in sorted(candidate.batch.unique()):
        a = candidate[candidate.batch.eq(batch)]
        b = control[control.batch.eq(batch)]
        delta = a[['slot', 'pic50_on_rescore']].merge(
            b[['slot', 'pic50_on_rescore']], on='slot', validate='one_to_one', suffixes=('_a', '_b'))
        per_batch.append({'batch': int(batch), 'n': len(delta),
                          'paired_mean_pic50': float((delta.pic50_on_rescore_a - delta.pic50_on_rescore_b).mean())})
    result['batch_details'] = per_batch
    return result


def load_round(reports, item):
    # The old PB output is an evaluator failure, not a biological negative.
    suffix = 'round08_replay' if item.get('repaired_retention_report') else f"round{item['round']:02d}"
    folder = Path(reports) / suffix
    verified_retention(folder)
    data = pd.read_csv(folder / 'candidate_metrics.csv')
    if data[['arm', 'batch', 'slot']].duplicated().any():
        raise ValueError('Duplicate terminal attempt identifiers')
    if 'pb_error' in data and data.pb_error.notna().any():
        raise ValueError('Unresolved evaluator failure')
    return folder, data, read_json(folder / 'execution_report.json')


def build_campaign_report(reports, campaign_config, output, steer_metrics=None, donor_batches=None):
    reports, output = Path(reports).resolve(), Path(output).resolve()
    campaign = read_json(campaign_config)
    count = campaign['rounds_completed']
    if count != campaign['maximum_rounds'] or [r['round'] for r in campaign['rounds']] != list(range(1, count + 1)):
        raise ValueError('Finish every registered round before final synthesis')
    if any(r['status'] != 'complete' for r in campaign['rounds']):
        raise ValueError('Unresolved campaign round')
    if output.exists():
        raise FileExistsError(output)
    threshold = read_json(reports / 'round02_credit_manifest.json')['threshold_pic50']
    frozen = read_json(reports / 'frozen_validation.json')
    frames, executions, rounds, elites, sources = {}, {}, [], [], []
    for item in campaign['rounds']:
        number = item['round']
        row = {'round': number, 'kind': item['kind'], 'module': json.dumps(item['module'], ensure_ascii=False)}
        if item['kind'] == 'inference':
            folder, data, execution = load_round(reports, item)
            frames[number], executions[number] = data, execution
            plan = reports / f'round{number:02d}_plan.json'
            program = read_json(folder / 'inference_config/reward_program.json')
            row.update(parent=read_json(plan)['parent'] if plan.exists() else 4 if number == 5 else None,
                       code_commit=execution['code_commit'], reward_view=program['reward_view'],
                       window=json.dumps(program['window']),
                       **qualified_summary(data[data.arm.eq('gradient')], threshold))
            hit = data[data.arm.eq('gradient') & data.valid_connected & data.pic50_on_rescore.ge(threshold)].copy()
            hit['round'] = number
            elites.append(hit[['round', 'batch', 'slot', 'pic50_on_rescore', 'pb_fast_pass',
                              'energy_status', 'mmff_relief_per_heavy', 'smiles']])
            sources.append({'round': number, 'retention': folder.name + '/retention.json',
                            'retention_sha256': digest(folder / 'retention.json')})
        rounds.append(row)

    first, second = frames[18], frames[20]
    selected_a, native_a = first[first.arm.eq('gradient')], first[first.arm.eq('unguided')]
    selected_b, native_b = second[second.arm.eq('gradient')], second[second.arm.eq('unguided')]
    old = frames[19][frames[19].arm.eq('gradient')]
    for number in [18, 20]:
        verify_initial_pair(executions[number], executions[number])
    verify_initial_pair(executions[18], executions[19], control_arm='gradient')
    selected = pd.concat([selected_a, selected_b], ignore_index=True)
    native = pd.concat([native_a, native_b], ignore_index=True)
    if selected[['batch', 'slot']].duplicated().any():
        raise ValueError('Confirmation panels overlap')
    program_a = read_json(reports / 'round18/inference_config/reward_program.json')
    program_b = read_json(reports / 'round20/inference_config/reward_program.json')
    administrative = {'round', 'program_id', 'derivation'}
    if {k: v for k, v in program_a.items() if k not in administrative} != {k: v for k, v in program_b.items() if k not in administrative}:
        raise ValueError('Validation reward was tuned between panels')
    validation = {
        'winner_round': frozen['winner_round'], 'threshold_pic50': threshold,
        'selected': qualified_summary(selected, threshold), 'native': qualified_summary(native, threshold),
        'old_incumbent_panel_a': qualified_summary(old, threshold),
        'selected_vs_native_pooled': confirmation_pair(selected, native),
        'selected_vs_native_panel_a': confirmation_pair(selected_a, native_a),
        'selected_vs_native_panel_b': confirmation_pair(selected_b, native_b),
        'selected_vs_old_panel_a': confirmation_pair(selected_a, old),
        'old_vs_native_panel_a': confirmation_pair(old, native_a),
        'selection': frozen,
    }
    if steer_metrics:
        data = pd.read_csv(steer_metrics)
        donors = set(donor_batches or [])
        validation['historical_steer'] = {'all': qualified_summary(data, threshold),
                                        'limitation': 'Unpaired historical reference; donor labels were used in reward design. Compute differs.'}
        if donors:
            validation['historical_steer']['donors'] = qualified_summary(data[data.batch.isin(donors)], threshold)
            validation['historical_steer']['non_donors'] = qualified_summary(data[~data.batch.isin(donors)], threshold)
        sources.append({'steer_metrics_sha256': digest(steer_metrics)})
    all_elites = pd.concat(elites, ignore_index=True)
    adaptive = all_elites[all_elites['round'].le(17)]
    validation['adaptive_elites'] = {'records': len(adaptive), 'unique_graphs': int(adaptive.smiles.nunique()),
                                     'limitation': 'Repeated adaptive hits are not independent discoveries.'}
    output.mkdir(parents=True)
    write_table(output / 'rounds.csv', rounds)
    write_table(output / 'elite_records.csv', all_elites)
    write_json(output / 'validation.json', validation)
    write_json(output / 'input_provenance.json', {'campaign_sha256': digest(campaign_config),
                                                'frozen_validation_sha256': digest(reports / 'frozen_validation.json'),
                                                'inputs': sources})
    lines = ['# Sequential path campaign: 20 rounds', '',
             f"Frozen winner: round {frozen['winner_round']}; master seed {campaign['master_seed']}; threshold {threshold:.6f} pIC50.", '',
             'Rounds 1–3 are local prerequisites; rounds 4–17 are inference controls/adaptive screens; rounds 18–20 are frozen confirmation.',
             'All-output head scores include failed molecular decoding and are shown beside valid-output scores. PB is the dock_fast subset. Strain is MMFF local relaxation relief, not binding energy.', '',
             '| Round | Module | Parent | N | Mean | Valid mean | Max | Elite | PB | Strain median |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rounds:
        if row['kind'] != 'inference':
            lines.append(f"| {row['round']} | {row['module']} | — | — | — | — | — | — | — | — |")
        else:
            lines.append(f"| {row['round']} | {row['module']} | {row['parent']} | {row['n']} | {row['all_mean_pic50']:.4f} | {row['valid_mean_pic50']:.4f} | {row['valid_max_pic50']:.4f} | {row['elite_valid_n']} | {row['pb_fast_rate']:.2f} | {row['strain_median_per_heavy']:.4f} |")
    lines += ['', 'Pooled independent confirmation:', '',
              f"Selected mean {validation['selected']['all_mean_pic50']:.6f}; native mean {validation['native']['all_mean_pic50']:.6f}.",
              f"Paired gain {validation['selected_vs_native_pooled']['paired_mean_pic50']:.6f}; batch-bootstrap CI95 {validation['selected_vs_native_pooled']['batch_bootstrap_CI95']}.",
              f"Adaptive elite records {len(adaptive)}, distinct graphs {adaptive.smiles.nunique()}. No region-specific causal mechanism is established by these counts.", '',
              'Round 8 uses the checksum-retained evaluator repair replay. Reported physical failures are never replaced by an evaluator exception.',
              'See validation.json for per-panel, incumbent, PB-qualified tail, energy and historical Steer comparisons. Few deterministic batches do not establish broad generalization.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return validation
