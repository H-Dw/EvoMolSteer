"""Audit completed terminal-region experiments without model or API inference.

Inputs are the verified execution archives, frozen control bank and regional
mining results. Only batch/node summaries, best unique structures and provenance
are written; individual trace events and coordinate arrays remain in archives.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import read_json, write_json, write_table, digest
from evomolsteer.storage.generation_archive import restoration_destination


def summarize_trace(rows):
    active = [r for r in rows if r['reward_evaluated']]
    if not active:
        return {'controlled_steps': 0, 'injection_rms_A': 0.}

    def values(key):
        return np.asarray([r[key] for r in active], dtype=float)

    injection = values('injection_rms_A')
    calibration = values('calibration_rms_A')
    square = values('injection_l2_A') ** 2
    result = {
        'controlled_steps': len(active), 'seed_step_events': injection.size,
        'injection_rms_A': float(injection.mean()),
        'realized_over_calibration': float((injection / calibration).mean()),
        'zero_injection_events': int((injection <= 1e-12).sum()),
        'cap_active_fraction': float((values('cap_factor') < 1 - 1e-6).mean()),
        'backtrack_fraction': float((values('backtrack_factor') < 1 - 1e-6).mean()),
        'max_atom_injection_A': float(values('max_atom_injection_A').max()),
        'positive_first_order_response_fraction': float((values('first_order_reward_change') > 0).mean()),
        'mean_gradient_native_cosine': float(values('flowcompat_native_cosine_before').mean()),
        'final_cumulative_rms_A': float(np.mean(active[-1]['cumulative_rms_A'])),
        'first_mean_nearest_regional_or_global_rms_A': float(np.mean(active[0]['nearest_standardized_rms'])),
        'last_mean_nearest_regional_or_global_rms_A': float(np.mean(active[-1]['nearest_standardized_rms'])),
    }
    if 'regional_core_fraction' in active[0]:
        result.update(
            core_fraction=float(values('regional_core_fraction').mean()),
            noncore_fraction_of_total_squared_injection=float(
                (values('injection_noncore_fraction') * square).sum() / square.sum()),
            zero_background_verified=bool((values('regional_zero_background') == 1).all()),
        )
    return result


def analyze(repo, study, output):
    repo, study, output = repo.resolve(), study.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = read_json(study / 'collection_status.json')
    fresh = read_json(study / 'remote_completion_check.json')
    if state['status'] != 'complete' or not fresh['all_complete_and_matching']:
        raise ValueError('Complete remote/local archive verification is required')
    config_path = repo / 'configs/experiments/m2_terminal_regions_20261014/analysis.json'
    config = read_json(config_path)
    bank_path = repo / 'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json'
    bank = read_json(bank_path)
    transport = repo / 'data/generated/m2_terminal_regions_20261014/transport'
    metrics = {name: repo / bank['result']['outcomes'][arm]['metrics']
               for name, arm in [('M2', 'gradient'), ('unguided', 'unguided')]}
    metrics['historical_Steer'] = repo / 'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv'
    audit, batch_rows, time_rows, source_hashes = {}, [], [], {}
    candidates = sorted(n for n in state['completed'] if n.startswith('M2REG_terminal_'))
    for name in candidates:
        archive = transport / (name + '.tar.gz')
        if digest(archive) != fresh['archives'][name]['remote_sha256']:
            raise ValueError('Archive changed after remote verification: ' + name)
        root = restoration_destination(transport, name) / 'results' / name
        program = read_json(root / 'reward_program.json')
        preflight = read_json(root / 'coordinate_gradient_preflight.json')
        execution = read_json(study / 'transport_records' / (name + '.execution_audit.json'))
        pairing = read_json(study / 'transport_records' / (name + '.input_pairing.json'))
        if not (execution['passed'] and pairing['passed'] and preflight['passed']):
            raise ValueError('Execution/input/gradient audit did not pass: ' + name)
        if program['affinity_head_gradient'] or program['additional_per_step_affinity_calls']:
            raise ValueError('Unexpected affinity-based guidance')
        all_rows, checked_batches = [], []
        for trace in sorted(root.glob('gradient/batch_*/guidance_trace.jsonl')):
            batch = int(trace.parent.name.split('_')[-1])
            rows = [read_json_line(line) for line in trace.read_text().splitlines() if line.strip()]
            window = program['window']
            for row in rows:
                expected = window[0] - 1e-7 <= row['score_time'] < window[1] - 1e-7
                if row['reward_evaluated'] != expected:
                    raise ValueError('Guidance clock differs from configured window')
                if row['particle_resampled'] or row['production_target_forward_calls'] != 1:
                    raise ValueError('Unexpected resampling or additional production forward')
                if not expected and np.asarray(row['injection_l2_A']).max() > 0:
                    raise ValueError('Injection outside configured window')
                scalar = summarize_trace([row])
                time_rows.append({'condition': name, 'batch': batch, 'step': row['step'],
                                  'score_time': row['score_time'], **scalar})
            with np.load(trace.parent / 'execution_state.npz') as execution_arrays:
                if execution_arrays['resampled'].any():
                    raise ValueError('Recorded particle resampling')
                indices = execution_arrays['selected_indices']
                if not np.array_equal(indices, np.broadcast_to(np.arange(config['batch_size']), indices.shape)):
                    raise ValueError('Recorded particle selection')
            if len(rows) != config['steps'] or abs(rows[-1]['state_time'] - 1.) > 2e-6:
                raise ValueError('Incomplete final inference')
            summary = summarize_trace(rows)
            batch_rows.append({'condition': name, 'batch': batch, **summary})
            source_hashes[str(trace.relative_to(repo))] = digest(trace)
            checked_batches.append(batch)
            all_rows.extend(rows)
        if checked_batches != config['development_batches']:
            raise ValueError('Missing generated batches')
        # Eight equally sized batches: averages of the event summaries have
        # equal batch weights. squared-energy spatial share uses measured dose.
        summary = summarize_trace(all_rows)
        summary['controlled_steps'] //= len(checked_batches)
        for key in ['final_cumulative_rms_A', 'first_mean_nearest_regional_or_global_rms_A',
                    'last_mean_nearest_regional_or_global_rms_A']:
            summary[key] = float(np.mean([r[key] for r in batch_rows if r['condition'] == name]))
        audit[name] = {'program_sha256': digest(root / 'reward_program.json'),
                       'reward_view': program['reward_view'], 'window': program['window'],
                       'source_learning_window': program['source_learning_window'],
                       'preflight_min_relative_error': min(r['relative_error'] for r in preflight['checks']),
                       'paired_inputs': True, 'inference_complete': True, 'no_SMC': True,
                       'production_forwards_per_step': 1, 'guidance': summary}
        metrics[name] = study / 'evaluated' / name / 'candidate_metrics.csv'

    write_table(output / 'guidance_batch_summary.csv', batch_rows)
    per_node = pd.DataFrame(time_rows)
    write_table(output / 'guidance_exact_node_summary.csv', per_node.groupby(
        ['condition', 'step', 'score_time'], as_index=False).mean(numeric_only=True).drop(columns='batch'))
    best, tails, failures = [], [], {}
    fields = ['batch', 'slot', 'pic50_on_rescore', 'pb_fast_pass', 'energy_status',
              'mmff_relief_per_heavy', 'relax_rms_surround_A', 'min_protein_distance_A', 'smiles']
    for name, path in metrics.items():
        frame = pd.read_csv(path)
        source_hashes[str(path.relative_to(repo))] = digest(path)
        if name in ('M2', 'unguided'):
            # Add existing full control records, without re-evaluating any
            # molecule. Check the frozen score table before using metadata.
            full_path = (repo / bank['all_attempt_tensor_and_decode_archive']).parent / 'candidate_metrics.csv'
            full = pd.read_csv(full_path)
            arm = 'gradient' if name == 'M2' else 'unguided'
            full = full[full.arm.eq(arm)].copy()
            matched = full.merge(frame[['batch', 'slot', 'pic50_on_rescore']],
                                 on=['batch', 'slot'], validate='one_to_one', suffixes=('', '_frozen'))
            if len(matched) != len(frame) or not np.allclose(
                    matched.pic50_on_rescore, matched.pic50_on_rescore_frozen, atol=1e-12, rtol=0):
                raise ValueError('Full control metadata differs from frozen metrics')
            frame = full
            source_hashes[str(full_path.relative_to(repo))] = digest(full_path)
        valid = frame[frame.valid_connected]
        top = valid.sort_values('pic50_on_rescore', ascending=False, kind='stable').drop_duplicates('smiles').head(5)
        best.extend({'condition': name, 'rank': rank, **row.reindex(fields).to_dict()}
                    for rank, (_, row) in enumerate(top.iterrows(), 1))
        high = valid[valid.pic50_on_rescore >= config['evaluation']['elite_threshold_pic50']]
        tails.append({'condition': name, 'attempts': len(frame), 'high_valid_count': len(high),
                      'high_valid_unique_graphs': high.smiles.nunique(),
                      'p10_pic50': frame.pic50_on_rescore.quantile(.1),
                      'valid_unique_graphs': valid.smiles.nunique()})
        failures[name] = {
            'decode_status': frame['decode_status'].fillna('missing').value_counts().to_dict()
                if 'decode_status' in frame else {'not_recorded_in_source_metrics': len(frame)},
            'energy_status': frame.energy_status.fillna('missing').value_counts().to_dict(),
            'invalid_count': int((~frame.valid_connected).sum()),
            'PB_failed_count': int((~frame.pb_fast_pass).sum()),
            'failure_reason': frame.failure_reason.fillna('none').value_counts().to_dict(),
        }
    write_table(output / 'best_unique_candidates.csv', best)
    write_table(output / 'tail_and_diversity.csv', tails)
    write_json(output / 'failure_counts.json', failures)

    selection = read_json(study / 'calls_v1/selection.json')
    mining = read_json(study / 'mining_v4/manifest.json')
    effects = pd.read_parquet(study / 'mining_v4/whole_window_effects.parquet')
    with gzip.open(study / 'mining_v4/terminal_teacher_reference.json.gz', 'rt') as stream:
        import json
        reference = json.load(stream)
    selected = [r['region_index'] for r in selection['design']['selected_regions']]
    write_table(output / 'selected_region_discovery_validation.csv', effects[effects.region_index.isin(selected)])
    trends = read_json(study / 'mining_v4/empirical_trends.json')
    times = np.asarray(trends['times'])
    selected_curves = {}
    for handle in selection['actual_curves_read']:
        key = handle.removeprefix('curve/')
        curve = trends['functions'][key]
        if curve['region_index'] not in selected:
            continue
        values = np.array([np.nan if x is None else x for x in curve['values']])
        finite = np.isfinite(values)
        selected_curves[key] = {'observed_times': times[finite],
                               'observed_range': [values[finite].min(), values[finite].max()]}
    point = np.array(reference['landmarks_A'])[selected]
    teachers = np.asarray(reference['frames'][0]['teacher_endpoint_A'])
    distance = np.linalg.norm(teachers[:, :, None] - point[None, None], axis=-1)
    coverage = {str(radius): ((distance < radius).any(-1).sum(1)).tolist()
                for radius in config['agent']['allowed_radii_A']}
    receipts = [read_json(p) for p in sorted((study / 'calls_v1/Analyst').glob('step_*.receipt.json'))]
    receipts.append(read_json(study / 'calls_v1/Designer.receipt.json'))
    source_hashes.update({str(p.relative_to(repo)): digest(p) for p in [config_path, bank_path,
        study / 'mining_v4/manifest.json', study / 'calls_v1/selection.json']})
    record = {
        'schema': 'terminal-region-completed-analysis-1.0', 'audit': audit,
        'source_sha256': source_hashes, 'selected_regions': selected,
        'teacher_atoms_inside_radii_A': coverage,
        'teacher_unique_ancestors': mining['unique_teacher_ancestors'],
        'selected_curves': selected_curves,
        'API': {'calls': len(receipts), 'models': sorted(set(r['returned_model'] for r in receipts)),
                'all_finished_stop': all(r['finish_reason'] == 'stop' for r in receipts),
                'provisional_scientific_finish_rejections': len(list((study / 'calls_v1/Analyst').glob('*.finish_rejected.json')))},
        'limitations': ['One Agent region decision; not an individual Skill ablation',
            'Historical Steer has a different search budget and dependent offspring',
            'Retrospective partial associations are not causal intervention directions',
            'Delayed window reduces update count; integrated dose is not matched',
            'Noncore refers to matched endpoint atom indices, not a fixed geometric boundary',
            'First-order reward response is not post-update reward or affinity gain'],
    }
    write_json(output / 'execution_and_evidence_audit.json', record)
    return record


def read_json_line(line):
    import json
    return json.loads(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.repo, args.study, args.output)
    print({'audited_candidates': len(result['audit']), 'API': result['API']})
