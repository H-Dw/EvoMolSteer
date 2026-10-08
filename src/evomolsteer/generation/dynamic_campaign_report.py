"""Summarize a completed dynamic-cohort campaign from retained evidence only.

The confirmation candidate is frozen before its batch labels are observed.
Screening repeats are never counted as independent discoveries. No molecule
coordinates or retired transport archives are needed to rebuild this report.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..io import digest, read_json, write_json, write_table
from .path_campaign_report import (
    confirmation_pair, qualified_summary, verified_retention, verify_initial_pair,
)


ADMINISTRATIVE_FIELDS = {'round', 'program_id', 'derivation'}


def reward_parameters(program):
    return {k: v for k, v in program.items() if k not in ADMINISTRATIVE_FIELDS}


def load_evidence(reports, number):
    folder = Path(reports) / f'round{number:02d}'
    retention = verified_retention(folder)
    expected = f'dynamic_contrast_r{number:02d}'
    if retention['campaign'] != expected:
        raise ValueError('Retained campaign identity differs')
    data = pd.read_csv(folder / 'candidate_metrics.csv')
    if data[['arm', 'batch', 'slot']].duplicated().any():
        raise ValueError('Duplicate terminal attempts')
    if not np.isfinite(data.pic50_on_rescore).all():
        raise ValueError('All-attempt affinity coverage required')
    if 'pb_error' in data and data.pb_error.notna().any():
        raise ValueError('Unresolved evaluator error')
    execution = read_json(folder / 'execution_report.json')
    if (execution['steps'] != 100 or not execution['no_particle_resampling']
            or execution['outside_window_injection'] or execution['affinity_head_gradient']
            or execution['additional_production_calls_per_step'] != 0
            or not execution['preflight']['passed']):
        raise ValueError('Inference execution contract differs')
    config = read_json(folder / 'inference_config/config.json')
    if config['experiment']['seed'] != 42 or config['extension']['code_commit'] != execution['code_commit']:
        raise ValueError('Seed or code provenance differs')
    program = read_json(folder / 'inference_config/reward_program.json')
    if program['window'] != execution['window']:
        raise ValueError('Program and execution support differ')
    return data, execution, program, config


def paired_valid_summary(candidate, control):
    """Descriptive sensitivity on the intersection, not a validity-adjusted effect."""
    merged = candidate[['batch', 'slot', 'pic50_on_rescore', 'valid_connected']].merge(
        control[['batch', 'slot', 'pic50_on_rescore', 'valid_connected']],
        on=['batch', 'slot'], suffixes=('_a', '_b'), validate='one_to_one')
    common = merged.valid_connected_a & merged.valid_connected_b
    delta = merged.loc[common, 'pic50_on_rescore_a'] - merged.loc[common, 'pic50_on_rescore_b']
    return {'common_valid_n': int(common.sum()),
            'common_valid_mean_difference_pic50': float(delta.mean()),
            'candidate_valid_only_n': int((merged.valid_connected_a & ~merged.valid_connected_b).sum()),
            'control_valid_only_n': int((~merged.valid_connected_a & merged.valid_connected_b).sum()),
            'limitation': 'Descriptive intersection; excluding failed decoding changes the selected population.'}


def paired_comparison(candidate, control):
    result = confirmation_pair(candidate, control)
    result['valid_intersection'] = paired_valid_summary(candidate, control)
    return result


def qualified_peak(candidate, native, incumbent):
    eligible = candidate[candidate.valid_connected & candidate.pb_fast_pass]
    if eligible.empty:
        return None
    best = eligible.sort_values('pic50_on_rescore', ascending=False, kind='stable').iloc[0]
    fields = ['batch', 'slot', 'seed', 'pic50_on_rescore', 'pic50_off_rescore', 'valid_connected',
              'pb_fast_pass', 'energy_status', 'mmff_relief_per_heavy', 'smiles',
              'min_protein_distance_A', 'severe_pairs_below_1_2A']
    def record(row):
        return {name: row[name] for name in fields if name in row}
    result = {'candidate': record(best)}
    for name, control in [('native_same_initial_slot', native), ('incumbent_same_initial_slot', incumbent)]:
        matched = control[control.batch.eq(best.batch) & control.slot.eq(best.slot)]
        result[name] = record(matched.iloc[0]) if len(matched) else None
    result['limitation'] = 'Post-hoc best qualified molecule, not an independent average effect or a regional causal attribution.'
    return result


def build_report(reports, config_dir, output, steer_metrics=None, donor_batches=range(14)):
    reports, config_dir, output = map(lambda p: Path(p).resolve(), (reports, config_dir, output))
    campaign = read_json(config_dir / 'campaign.json')
    if (campaign['maximum_rounds'] != 10 or campaign['rounds_completed'] != 10
            or [r['round'] for r in campaign['rounds']] != list(range(1, 11))
            or any(r['status'] != 'complete' for r in campaign['rounds'])):
        raise ValueError('Every registered round must finish before final synthesis')
    if output.exists():
        raise FileExistsError(output)
    protocol = read_json(reports / 'protocol.json')
    threshold = protocol['tail_threshold_pic50']
    frozen = read_json(reports / 'frozen_validation.json')
    chosen = frozen['candidate_round']
    if chosen not in range(2, 8) or digest(config_dir / f'round{chosen:02d}.json') != frozen['program_sha256']:
        raise ValueError('Frozen screening candidate changed')
    expected_parameters = reward_parameters(read_json(config_dir / f'round{chosen:02d}.json'))
    frames, executions, programs, sources, rows, hits = {}, {}, {}, [], [], []
    checkpoint_hashes = set()
    for number in range(1, 11):
        data, execution, program, config = load_evidence(reports, number)
        plan = read_json(reports / f'round{number:02d}_plan.json')
        if (sorted(data.batch.unique().tolist()) != list(map(int, plan['batches'].split(',')))
                or sorted(data.arm.unique().tolist()) != sorted(plan['arms'].split(','))
                or any(len(data[data.arm.eq(arm)]) != plan['n_per_arm'] for arm in data.arm.unique())):
            raise ValueError('Observed generation panel differs from frozen plan')
        source_program = config_dir / f'round{number:02d}.json'
        if digest(source_program) != plan['program_sha256'] or read_json(source_program) != program:
            raise ValueError('Executed program differs from committed plan')
        if config['extension']['reference_sha256'] != program['reference_sha256']:
            raise ValueError('Executed reference provenance differs')
        checkpoint_hashes.add(config['extension']['checkpoint_sha256'])
        frames[number], executions[number], programs[number] = data, execution, program
        g = data[data.arm.eq('gradient')]
        row = {'round': number, 'parent': str(plan['parent']), 'reason': plan['reason'],
               'single_module_change': json.dumps(plan['change'], ensure_ascii=False, sort_keys=True),
               'batches': plan['batches'], 'code_commit': execution['code_commit'],
               'reward_view': program['reward_view'], 'regional_weight': program.get('regional_weight', 0.),
               'regional_response': program.get('regional_response', 'none'),
               'window': json.dumps(program['window']), **qualified_summary(g, threshold)}
        dose = [r for r in execution.get('batch_results', []) if r['arm'] == 'gradient']
        if dose:
            row.update(controlled_steps_min=min(r['controlled_steps'] for r in dose),
                       nonzero_steps_min=min(r['nonzero_steps'] for r in dose),
                       mean_active_injection_rms_A=float(np.mean([r['mean_active_injection_rms_A'] for r in dose])),
                       mean_cumulative_path_rms_A=float(np.mean([r['mean_cumulative_rms_A'] for r in dose])))
        control = frames[1] if 2 <= number <= 7 else frames[8] if number == 9 else data
        for arm in ['gradient', 'unguided']:
            if arm == 'gradient' and control is data:
                continue
            c = control[control.arm.eq(arm) & control.batch.isin(g.batch.unique())]
            if len(c) == len(g):
                row['paired_gain_vs_' + arm] = float((g[['batch', 'slot', 'pic50_on_rescore']].merge(
                    c[['batch', 'slot', 'pic50_on_rescore']], on=['batch', 'slot'],
                    validate='one_to_one', suffixes=('_g', '_c')).eval('pic50_on_rescore_g - pic50_on_rescore_c')).mean())
        rows.append(row)
        elite = g[g.valid_connected & g.pic50_on_rescore.ge(threshold)].copy()
        elite['round'] = number
        hits.append(elite[['round', 'batch', 'slot', 'pic50_on_rescore', 'pb_fast_pass',
                           'energy_status', 'mmff_relief_per_heavy', 'smiles']])
        sources.append({'round': number, 'retention_sha256': digest(reports / f'round{number:02d}/retention.json'),
                        'code_commit': execution['code_commit']})
    if len(checkpoint_hashes) != 1:
        raise ValueError('Checkpoint changed across the campaign')
    if reward_parameters(programs[8]) != reward_parameters(programs[1]):
        raise ValueError('Incumbent confirmation was changed')
    for n in [9, 10]:
        if reward_parameters(programs[n]) != expected_parameters:
            raise ValueError('Reward tuned after confirmation labels')
    verify_initial_pair(executions[8], executions[8])
    verify_initial_pair(executions[9], executions[8], control_arm='gradient')
    verify_initial_pair(executions[9], executions[8], control_arm='unguided')
    verify_initial_pair(executions[10], executions[10])
    for n in range(2, 8):
        verify_initial_pair(executions[n], executions[1], control_arm='gradient')
    candidate_a = frames[9][frames[9].arm.eq('gradient')]
    candidate_b = frames[10][frames[10].arm.eq('gradient')]
    incumbent = frames[8][frames[8].arm.eq('gradient')]
    native_a = frames[8][frames[8].arm.eq('unguided')]
    native_b = frames[10][frames[10].arm.eq('unguided')]
    candidate = pd.concat([candidate_a, candidate_b], ignore_index=True)
    native = pd.concat([native_a, native_b], ignore_index=True)
    if candidate[['batch', 'slot']].duplicated().any():
        raise ValueError('Confirmation panels overlap')
    validation = {'schema_version': 'dynamic-cohort-confirmation-1.0',
                  'candidate_round': chosen, 'master_seed': 42, 'threshold_pic50': threshold,
                  'selected': qualified_summary(candidate, threshold),
                  'native': qualified_summary(native, threshold),
                  'screening_controls': {a: qualified_summary(frames[1][frames[1].arm.eq(a)], threshold)
                                         for a in ['gradient', 'unguided']},
                  'incumbent_panel_a': qualified_summary(incumbent, threshold),
                  'selected_vs_native_pooled': paired_comparison(candidate, native),
                  'selected_vs_native_panel_a': paired_comparison(candidate_a, native_a),
                  'selected_vs_native_panel_b': paired_comparison(candidate_b, native_b),
                  'selected_vs_incumbent_panel_a': paired_comparison(candidate_a, incumbent),
                  'incumbent_vs_native_panel_a': paired_comparison(incumbent, native_a),
                  'best_qualified_candidate': qualified_peak(candidate, native, incumbent),
                  'frozen_selection': frozen,
                  'attempted_records': sum(len(d) for d in frames.values()),
                  'screening_independent_batches': 1,
                  'confirmation_independent_batches': int(candidate.batch.nunique()),
                  'checkpoint_sha256': next(iter(checkpoint_hashes))}
    seed_rows = [d[['batch', 'seed']] for d in frames.values() if 'seed' in d]
    if seed_rows:
        seeds = pd.concat(seed_rows).drop_duplicates()
        if seeds.batch.duplicated().any():
            raise ValueError('Batch RNG stream changed between paired rounds')
        validation['observed_inference_rng_seeds_by_batch'] = {str(int(r.batch)): int(r.seed)
                                                              for r in seeds.itertuples()}
    all_hits = pd.concat(hits, ignore_index=True)
    adaptive = all_hits[all_hits['round'].between(2, 7)]
    validation['adaptive_elites'] = {'records': len(adaptive), 'unique_graphs': int(adaptive.smiles.nunique()),
                                     'limitation': 'Repeated screening initial states and adaptive labels are not independent discoveries.'}
    if steer_metrics is not None:
        steer = pd.read_csv(steer_metrics)
        donor_mask = steer.batch.isin(list(donor_batches))
        validation['historical_steer'] = {'all': qualified_summary(steer, threshold),
                                        'donors': qualified_summary(steer[donor_mask], threshold),
                                        'non_donors': qualified_summary(steer[~donor_mask], threshold),
                                        'limitation': 'Unpaired historical reference with different selection compute; donor labels informed design.'}
        sources.append({'steer_metrics_sha256': digest(steer_metrics)})
    output.mkdir(parents=True)
    write_table(output / 'rounds.csv', rows)
    write_table(output / 'elite_records.csv', all_hits)
    write_json(output / 'validation.json', validation)
    write_json(output / 'input_provenance.json', {'campaign_sha256': digest(config_dir / 'campaign.json'),
                'protocol_sha256': digest(reports / 'protocol.json'),
                'frozen_validation_sha256': digest(reports / 'frozen_validation.json'), 'inputs': sources})
    return validation
