"""Read-only collection analysis: numerical trials, actual dose and API evidence.

Writes compact tables and source-bound reports. No API, SSH or model generation.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from evomolsteer.ablation.statistics import compare
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.policy import program_signature
from evomolsteer.temporal_guidance.workflow import compile_recipe, render_instructions


def quality(frame, threshold):
    scored = pd.to_numeric(frame.pic50_on_rescore, errors='coerce')
    energy = pd.to_numeric(frame.loc[frame.energy_status.eq('converged'), 'mmff_relief_per_heavy'], errors='coerce')
    energy = energy.loc[np.isfinite(energy)]
    return {'attempts': len(frame), 'scored': int(np.isfinite(scored).sum()),
            'mean_pic50': float(scored.mean()), 'p95_pic50': float(scored.quantile(.95)),
            'p99_pic50': float(scored.quantile(.99)), 'valid': int(frame.valid_connected.sum()),
            'PB_pass': int(frame.pb_fast_pass.sum()), 'converged_energy': len(energy),
            'strain_median': float(energy.median()), 'strain_p90': float(energy.quantile(.9)),
            'elite_valid': int((frame.valid_connected & (scored >= threshold)).sum())}


def collect(repo, config_path, output):
    config = read_json(config_path)
    root = repo / config['output']
    state = read_json(root / 'run_status.json')
    bank = read_json(repo / config['control_manifest'])
    threshold = config['evaluation']['elite_threshold']
    inputs = {str(p.relative_to(repo)): digest(p) for p in [config_path, root / 'run_status.json', repo / config['control_manifest']]}
    inputs[Path(__file__).resolve().relative_to(repo).as_posix()] = digest(__file__)
    rows, deliveries, frames = [], [], {}
    for recipe in config['recipes']:
        for stage in ['screen', 'confirmation']:
            job = state['jobs'].get('control_' + recipe + '_' + stage)
            if not job or job['status'] != 'complete':
                continue
            result = job['result']['result']
            report = job['result']['comparison']
            path = repo / result['outcomes']['gradient']['metrics']
            inputs[path.relative_to(repo).as_posix()] = digest(path)
            frame = pd.read_csv(path)
            signature = result['effective_signature']
            assert signature == program_signature(compile_recipe(repo, config, recipe)[0])
            q = quality(frame, threshold)
            assert q['scored'] == q['attempts']
            assert np.isclose(q['mean_pic50'], report['summary']['candidate']['policy_mean']['mean_pic50'])
            c = report['comparisons']
            row = {'recipe': recipe, 'stage': stage, **q,
                   'delta_vs_M2': c['M2']['delta_pic50'],
                   'M2_simultaneous_CI_low': c['M2']['simultaneous_CI'][0],
                   'M2_simultaneous_CI_high': c['M2']['simultaneous_CI'][1],
                   'delta_vs_unguided': c['unguided']['delta_pic50'],
                   'delta_vs_Steer': c['Steer']['delta_pic50'],
                   'Steer_CI_low': c['Steer']['CI95'][0], 'Steer_CI_high': c['Steer']['CI95'][1],
                   'quality_envelope_passed': report['secondary_envelope']['passed'],
                   'quality_envelope_reasons': ';'.join(report['secondary_envelope']['reasons']),
                   'passes_confirmation_rule': report['eligible_for_400'],
                   'independent_confirmation': False}
            rows.append(row)
            frames[(recipe, stage)] = frame
            batches = report['batches']
            tables = []
            for job_name, value in state['jobs'].items():
                piece = value.get('result') or {}
                if not job_name.startswith('generation_') or piece.get('effective_signature') != signature:
                    continue
                campaign = piece['name']
                path = root / 'transport_records' / (campaign + '.temporal_delivery') / 'batch_node_delivery.parquet'
                inputs[path.relative_to(repo).as_posix()] = digest(path)
                tables.append(pd.read_parquet(path))
            nodes = pd.concat(tables, ignore_index=True)
            nodes = nodes.loc[nodes.batch.isin(batches)]
            assert not nodes.duplicated(['batch', 'score_time']).any()
            assert len(nodes) == len(batches) * len(compile_recipe(repo, config, recipe)[0]['temporal_policy']['score_times'])
            per_node = nodes.groupby('score_time').mean(numeric_only=True)
            eta = bank['program']
            base_eta = read_json(repo / eta)['native_rms_ratio']
            error = np.abs(nodes.delivered_ratio_mean - base_eta * nodes.dose_factors)
            deliveries.append({'recipe': recipe, 'stage': stage, 'independent_batches': len(batches),
                'cumulative_step_RMS_mean_A': float(nodes.groupby('batch').delivered_rms_mean_A.sum().mean()),
                'actual_ratio_first_node': float(per_node.delivered_ratio_mean.iloc[0]),
                'actual_ratio_last_node': float(per_node.delivered_ratio_mean.iloc[-1]),
                'mean_ratio': float(nodes.delivered_ratio_mean.mean()),
                'max_abs_ratio_policy_error': float(error.max()),
                'mean_cap_fraction': float(nodes.cap_triggered_fraction.mean()),
                'mean_backtracked_fraction': float(nodes.backtracked_fraction.mean()),
                'mean_zero_injection_fraction': float(nodes.zero_injection_fraction.mean())})
    numeric = pd.DataFrame(rows)
    controls, control_frames = {}, {}
    for label, arm in [('M2', 'gradient'), ('unguided', 'unguided')]:
        path = repo / bank['result']['outcomes'][arm]['metrics']
        inputs[path.relative_to(repo).as_posix()] = digest(path)
        frame = pd.read_csv(path)
        control_frames[label] = frame
        controls[label] = quality(frame, threshold)
    path = repo / config['historical_metrics']
    inputs[path.relative_to(repo).as_posix()] = digest(path)
    control_frames['historical_Steer'] = pd.read_parquet(path)
    controls['historical_Steer'] = quality(control_frames['historical_Steer'], threshold)
    role_rows = []
    for condition in config['conditions']:
        folder = root / 'calls' / condition['id'] / 'chain_00'
        a, d = read_json(folder / 'Analyst.json'), read_json(folder / 'Designer.json')
        provenance = read_json(folder / 'Analyst/references.json')['references']
        delivered = list(provenance.values())
        steps = [read_json(p) for p in sorted((folder / 'Analyst').glob('step_*.json'))
                 if p.name.count('.') == 1]
        for role in ['Analyst', 'Designer']:
            text, sources = render_instructions(repo, config, condition, role)
            assert sources == read_json(folder / (role + '.skills.json'))
            assert hashlib.sha256(text.encode()).hexdigest() == read_json(folder / role / 'binding.json')['instruction_sha256']
            assert read_json(folder / role / 'completed.json')['backend'] == 'real_api'
        role_rows.append({'condition': condition['id'], 'executed_chain': 0,
            'planned_chains': config['chains'], 'valid_condition': False,
            'Analyst_hypotheses': len(a['hypotheses']), 'Designer_decision': d['decision'],
            'proposed_recipe': (d.get('edit') or {}).get('value'),
            'get_curve_calls': sum(x.get('tool') == 'get_curve' for x in steps),
            'get_stage_profile_calls': sum(x.get('tool') == 'get_stage_profile' for x in steps),
            'coordinate_rows_read': sum(x['kind'] == 'row' for x in delivered),
            'coordinate_curves_read': sum(x['kind'] in {'curve', 'stage_profile'} for x in delivered),
            'actual_instruction_binding_matches': True,
            'failure': state['jobs']['roles_' + condition['id']]['failure']['error']})
    receipts = [read_json(p) for p in (root / 'calls').glob('*/chain_*/**/step_*.receipt.json')]
    remote = read_json(root / 'remote_collection_20261013.json')
    inputs[(root / 'remote_collection_20261013.json').relative_to(repo).as_posix()] = digest(root / 'remote_collection_20261013.json')
    assert remote['statuses'] == {'complete': 12} and not remote['worker_pids']
    pairs = []
    for left, right in [('S3_static_to_observed_control', 'S1_observed_to_static'),
                        ('S3_static_to_observed_control', 'S2_constant_blend'),
                        ('C3_base_to_robust_control', 'C2_constant_curvature'),
                        ('C3_base_to_robust_control', 'C1_robust_to_base')]:
        result = compare([frames[(left, 'screen')]], [frames[(right, 'screen')]], family=4, seed=42)
        pairs.append({'left': left, 'right': right, 'scope': 'exploratory screen contrast; not independent validation', **result})
    s3 = frames[('S3_static_to_observed_control', 'confirmation')]
    m2 = control_frames['M2']
    quantiles = pd.DataFrame([{'quantile': q, 'M2_pic50': m2.pic50_on_rescore.quantile(q),
        'S3_pic50': s3.pic50_on_rescore.quantile(q),
        'delta': s3.pic50_on_rescore.quantile(q) - m2.pic50_on_rescore.quantile(q)}
        for q in [.01, .05, .1, .25, .5, .75, .9, .95, .99]])
    joined = m2.merge(s3, on=['batch', 'slot'], suffixes=['_M2', '_S3'], validate='one_to_one')
    assert len(joined) == len(m2) == len(s3)
    delta = joined.pic50_on_rescore_S3 - joined.pic50_on_rescore_M2
    low = joined.pic50_on_rescore_M2 <= m2.pic50_on_rescore.quantile(.25)
    response = {'improved_above_1e_minus6': int((delta > 1e-6).sum()),
                'worsened_below_minus_1e_minus6': int((delta < -1e-6).sum()),
                'median_paired_delta': float(delta.median()),
                'lower_M2_quartile_delta_mean': float(delta.loc[low].mean()),
                'remaining_delta_mean': float(delta.loc[~low].mean()),
                'lower_M2_quartile_contribution_to_all_mean': float(delta.loc[low].sum() / len(delta)),
                'remaining_contribution_to_all_mean': float(delta.loc[~low].sum() / len(delta)),
                'invalid_M2_to_valid_S3': int((~joined.valid_connected_M2 & joined.valid_connected_S3).sum()),
                'valid_M2_to_invalid_S3': int((joined.valid_connected_M2 & ~joined.valid_connected_S3).sum()),
                'scope': 'post-hoc paired response conditioned on baseline scores; descriptive, susceptible to regression to mean; not causal geometry evidence'}
    completed_generations = [v['result'] for k, v in state['jobs'].items() if k.startswith('generation_') and v['status'] == 'complete']
    audits = [read_json(p) for p in (root / 'transport_records').glob('*.execution_audit.json')]
    assert len(audits) == len(completed_generations) == 12 and all(a['passed'] for a in audits)
    actual_batches = [b for a in audits for b in a['batches']]
    assert all(b['steps'] == 100 and b['controlled_steps'] == 50 and b['post_window_nonzero_injections'] == 0 for b in actual_batches)
    started = state['started_unix']
    report = {'schema_version': 'temporal-collection-report-1.0', 'collected_at': remote['collected_at'],
        'status': state['status'], 'started_Asia_Shanghai': datetime.fromtimestamp(started, ZoneInfo('Asia/Shanghai')).isoformat(),
        'finished_Asia_Shanghai': datetime.fromtimestamp(state['updated_unix'], ZoneInfo('Asia/Shanghai')).isoformat(),
        'wall_hours': (state['updated_unix'] - started) / 3600,
        'numerical_candidates': len(config['recipes']), 'completed_remote_cohorts': len(completed_generations),
        'new_attempts': sum(q['outcomes']['gradient']['summary']['n'] for q in completed_generations),
        'valid_Agent_conditions': 0, 'planned_Agent_conditions': len(config['conditions']),
        'completed_wire_roles': len(role_rows) * 2, 'planned_wire_roles': len(config['conditions']) * config['chains'] * 2,
        'API_action_calls': len(receipts), 'API_wire_attempts': sum(d['attempts'] for d in receipts),
        'requested_models': dict(Counter(d['requested_model'] for d in receipts)),
        'returned_models': dict(Counter(d['returned_model'] for d in receipts)),
        'API_elapsed_seconds': sum(d['completed_unix'] - d['started_unix'] for d in receipts),
        'controls': controls, 'elite_threshold': threshold, 'numerical_results': rows,
        'delivery': deliveries, 'role_evidence_audit': role_rows, 'exploratory_timing_contrasts': pairs,
        'S3_vs_M2_quantiles': quantiles.to_dict(orient='records'), 'S3_paired_response': response,
        'remote': remote, 'initial_states_and_no_SMC_audits_passed': True,
        'learning_window': read_json(repo / config['base_program'])['window'],
        'controlled_steps': 50, 'native_suffix_steps': 50, 'source_sha256': inputs,
        'limitations': ['Numerical controls were host preregistered, not valid Agent-designed rewards.',
                        '400-attempt extension includes the 200-attempt screen; not independent confirmation.',
                        'Extreme-tail inference remains low power; unequal Steer budget and dependent clones.',
                        'Temporal associations do not identify causal neural modules or optimal dose.']}
    output.mkdir(parents=True, exist_ok=True)
    numeric.to_csv(output / 'numerical_results.csv', index=False)
    pd.DataFrame(deliveries).to_csv(output / 'actual_delivery.csv', index=False)
    pd.DataFrame(role_rows).to_csv(output / 'agent_evidence_audit.csv', index=False)
    quantiles.to_csv(output / 'S3_vs_M2_quantiles.csv', index=False)
    write_json(output / 'collection_report.json', report)
    # Scientific plot: empirical upper quantiles, with cohort sizes explicit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    q = np.linspace(.5, .995, 100)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cohorts = {**control_frames, 'S3': frames[('S3_static_to_observed_control', 'confirmation')]}
    for label, frame in cohorts.items():
        ax.plot(q, frame.pic50_on_rescore.quantile(q), label=f'{label} (n={len(frame)})')
    ax.set(xlabel='Empirical quantile', ylabel='Final predicted pIC50',
           title='Mean improvement does not imply improved extreme tail')
    ax.axhline(threshold, color='gray', ls='--', lw=1, label='Frozen elite threshold')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'upper_tail_quantiles.png', dpi=180)
    plt.close(fig)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = collect(repo, repo / args.config, repo / args.output)
    print({k: report[k] for k in ['status', 'completed_remote_cohorts', 'new_attempts', 'valid_Agent_conditions', 'API_action_calls']})
