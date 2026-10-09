"""Read-only audit of deployed teacher labels and post-window generation.

Teacher matching respects original atom-slot order and the receptor frame.
All matches within serialization precision are retained, never assigned an
invented unique genealogy ID. Reported coverage is alias-inclusive coverage,
not a statement that nearby excluded coordinates receive no gradient.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.continuous.path_graph import validate_lineage
from evomolsteer.io import digest, read_json, write_json, write_table


def audit_suffix_rows(rows, window, guided):
    start, end = map(float, window)
    if not 0 <= start < end <= 1:
        raise ValueError('Dynamic valid window required')
    outside, controlled, last, first_native, calls = 0, 0, None, None, set()
    for row in rows:
        inside = row['score_time'] >= start-2e-6 and row['state_time'] <= end+2e-6
        injection = np.asarray(row['injection_l2_A'], float)
        if not np.isfinite(injection).all():
            raise ValueError('Nonfinite injection')
        if row['particle_resampled']:
            raise ValueError('Gradient/native arm unexpectedly resampled')
        if bool(row['reward_evaluated']) != (guided and inside):
            raise ValueError('Recorded reward support differs from the learned window')
        if not inside or not guided:
            if np.any(injection != 0) or row['active']:
                raise ValueError('Native suffix or control contains an injection')
            outside += 1
        if row['reward_evaluated']:
            controlled += 1
            last = (row['score_time'], row['state_time'])
        if row['score_time'] >= end-2e-6 and first_native is None:
            first_native = (row['score_time'], row['state_time'])
        calls.add(row['production_target_forward_calls'])
    if calls != {1}:
        raise ValueError('Unexpected production target forward count')
    return {'steps': len(rows), 'controlled_steps': controlled,
            'native_rows': outside, 'last_controlled_clocks': last,
            'first_postwindow_native_clocks': first_native,
            'suffix_injection_is_exact_zero': True, 'particle_resampling': False}


def selection_boundary(a, batch):
    clock, state, _, _, _, resampled = validate_lineage(a)
    active = np.flatnonzero(resampled)
    if not len(active):
        raise ValueError('Original Steer trajectory has no selection events')
    return {'batch': int(batch), 'selection_events': len(active),
            'last_selection_score_time': clock[active[-1]],
            'last_selection_state_time': state[active[-1]],
            'post_selection_resampling': bool(resampled[active[-1]+1:].any())}


def read_bound_references(completed, paths):
    libraries, metadata, configs = {}, {}, {}
    for name, folder in [('R26', 'controls_snapshot'), ('R11', 'R11_full')]:
        program_path = completed/folder/'inference_config/reward_program.json'
        program = read_json(program_path)
        path = paths[name]
        if digest(path) != program['reference_sha256']:
            raise ValueError('Deployed reference SHA mismatch: '+name)
        library = json.loads(gzip.decompress(path.read_bytes()))
        libraries[name] = library
        config = read_json(completed/folder/'inference_config/config.json')
        configs[name] = config
        metadata[name] = {
            'program_id': program['program_id'], 'program_sha256': digest(program_path),
            'reference_sha256': digest(path), 'reward_view': program['reward_view'],
            'window': library['window'], 'label_semantics': library['label_semantics'],
            'reference_variant': library.get('reference_variant'),
            'teacher_selection': library.get('teacher_selection'),
            'teacher_label_source': library.get('teacher_label_source'),
            'teacher_library_metadata': library.get('teacher_library'),
            'frame_count': len(library['frames']),
            'terminal_credit_fields_present': all('teacher_descendant_counts' in f for f in library['frames']),
            'explicit_teacher_slots_present': all('teacher_slots' in f for f in library['frames']),
            'branch_virtual_mass': program.get('branch_mixture', {}).get('virtual_mass'),
            'runtime_integrator': config['runtime_integrator'],
            'checkpoint_sha256': config['extension']['checkpoint_sha256'],
            'inference_code_commit': config['extension']['code_commit'],
        }
    equal = len(libraries['R26']['frames']) == len(libraries['R11']['frames']) and all(
        all(np.array_equal(a[k], b[k]) for k in ['time', 'teacher_endpoint_A', 'teacher_scores', 'teacher_batches'])
        for a, b in zip(libraries['R26']['frames'], libraries['R11']['frames']))
    return libraries, metadata, configs, equal


def teacher_coverage(root, campaign, metrics, reference, threshold, tolerance_A=1e-5):
    """Retrospective original-tail coverage, allowing serialization aliases."""
    config = read_json(root/'results'/campaign/'config.json')
    metrics = metrics.copy()
    metrics['elite'] = metrics.valid_connected & (metrics.pic50_on_rescore >= threshold)
    rows, inputs, boundaries = [], [], []
    batches = sorted(set(b for f in reference['frames'] for b in f['teacher_batches']))
    for batch in batches:
        m = metrics[metrics.batch == batch].sort_values('slot').reset_index(drop=True)
        path = root/'results'/campaign/'single'/f'batch_{batch:03d}'/'trajectory.npz'
        with np.load(path) as t:
            clock, state, selected, _, _, resampled = validate_lineage(t)
            ancestors = terminal_ancestors(selected)
            if not np.array_equal(m.slot, np.arange(selected.shape[1])):
                raise ValueError('Incomplete original terminal slot map')
            com = np.asarray(read_json(root/'results'/campaign/f'frame_batch_{batch:03d}.json')['target_com'])
            boundaries.append(selection_boundary(t, batch))
            for f in reference['frames']:
                index = np.flatnonzero(np.isclose(clock, f['time'], rtol=0, atol=2e-6))
                if len(index) != 1:
                    raise ValueError('Reference does not bind to one original event')
                step = int(index[0])
                ids = np.flatnonzero(np.asarray(f['teacher_batches']) == batch)
                xyz = t['predicted_coords'][step].astype(float)*config['coord_scale']+com[:, None]
                matched, ambiguity, largest_score_error, largest_min_rms = set(), 0, 0., 0.
                for teacher in ids:
                    cloud = np.asarray(f['teacher_endpoint_A'][teacher], float)
                    rms = np.sqrt(np.mean((xyz-cloud)**2, axis=(1, 2)))
                    matches = np.flatnonzero(rms <= tolerance_A)
                    if not len(matches):
                        raise ValueError('Teacher not recovered from recorded coordinates')
                    error = float(np.max(np.abs(t['pic50_on'][step, matches]-f['teacher_scores'][teacher])))
                    if error > 4e-6:
                        raise ValueError('Teacher label is not the recorded instantaneous score')
                    matched.update(matches.tolist())
                    ambiguity += int(len(matches) > 1)
                    largest_score_error = max(largest_score_error, error)
                    largest_min_rms = max(largest_min_rms, float(rms.min()))
                slots = np.flatnonzero(m.elite.to_numpy())
                covered = [int(s) for s in slots if ancestors[step, s] in matched]
                observed = set(ancestors[step].tolist())
                rows.append({'batch': int(batch), 'step': step, 'score_time': clock[step],
                    'state_time': state[step], 'teachers': len(ids), 'matched_node_aliases': len(matched),
                    'ambiguous_teachers': ambiguity, 'teacher_min_rms_max_A': largest_min_rms,
                    'teacher_score_max_error': largest_score_error,
                    'censored_matched_nodes': len(matched-observed),
                    'elite_terminal_slots': len(slots), 'covered_elite_slots': len(covered),
                    'elite_terminal_graphs': int(m.iloc[slots].smiles.nunique()),
                    'covered_elite_graphs': int(m.iloc[covered].smiles.nunique()),
                    'elite_ancestor_nodes': len(np.unique(ancestors[step, slots])),
                    'covered_elite_ancestor_nodes': len(set(ancestors[step, slots]).intersection(matched)),
                    'covered_final_slots': ','.join(map(str, covered))})
        inputs.append({'batch': int(batch), 'trajectory_sha256': digest(path)})
    frame = pd.DataFrame(rows)
    counts = ['teachers', 'matched_node_aliases', 'ambiguous_teachers', 'censored_matched_nodes',
              'elite_terminal_slots', 'covered_elite_slots', 'elite_ancestor_nodes', 'covered_elite_ancestor_nodes']
    curves = frame.groupby('step', sort=True).agg({**{k: 'sum' for k in counts}, 'score_time': 'mean'}).reset_index()
    curves['slot_coverage_fraction'] = curves.covered_elite_slots/curves.elite_terminal_slots
    curves['ancestor_coverage_fraction'] = curves.covered_elite_ancestor_nodes/curves.elite_ancestor_nodes
    return frame, curves, inputs, boundaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed-report', type=Path, required=True)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--original-campaign', required=True)
    p.add_argument('--steer-metrics', type=Path, required=True)
    p.add_argument('--r11-dataset', type=Path, required=True)
    p.add_argument('--r26-dataset', type=Path, required=True)
    p.add_argument('--r11-reference', type=Path, required=True)
    p.add_argument('--r26-reference', type=Path, required=True)
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    libraries, metadata, configs, same_teachers = read_bound_references(args.completed_report,
        {'R11': args.r11_reference, 'R26': args.r26_reference})
    metrics = pd.read_csv(args.steer_metrics)
    frame, curves, source_audit, boundaries = teacher_coverage(args.original_root, args.original_campaign,
                                                             metrics, libraries['R26'], args.threshold)
    donor_batches = {v['batch'] for v in source_audit}
    # Audit continuation boundaries on held-out original batches too, without
    # expanding the teacher-discovery cohort or saving their coordinates.
    for batch in sorted(set(metrics.batch)-donor_batches):
        path = args.original_root/'results'/args.original_campaign/'single'/f'batch_{batch:03d}'/'trajectory.npz'
        with np.load(path) as t:
            boundaries.append(selection_boundary(t, batch))
        source_audit.append({'batch': int(batch), 'trajectory_sha256': digest(path)})
    source_audit.sort(key=lambda v: v['batch'])
    boundaries.sort(key=lambda v: v['batch'])
    suffix = []
    trace_inputs = []
    for label, dataset, config_key, arm in [('R11', args.r11_dataset, 'R11', 'gradient'),
                                           ('R26', args.r26_dataset, 'R26', 'gradient'),
                                           ('native', args.r26_dataset, 'R26', 'unguided')]:
        config = configs[config_key]
        campaign = config['experiment']['campaign']
        for batch in config['experiment']['batch_indices']:
            folder = dataset/'results'/campaign/arm/f'batch_{batch:03d}'
            trace = folder/'guidance_trace.jsonl'
            data = [json.loads(line) for line in trace.read_text().splitlines()]
            result = audit_suffix_rows(data, libraries[config_key]['window'], arm == 'gradient')
            with np.load(folder/'execution_state.npz') as state:
                a = state['selected_indices']
                if state['resampled'].any() or not np.array_equal(a, np.broadcast_to(np.arange(a.shape[1]), a.shape)):
                    raise ValueError('Actual selected indices are not native identity')
            suffix.append({'cohort': label, 'batch': int(batch), **result})
            trace_inputs.append({'cohort': label, 'batch': int(batch), 'trace_sha256': digest(trace),
                                 'execution_state_sha256': digest(folder/'execution_state.npz')})
    original_manifest = read_json(args.original_root/'provenance/run_manifest.json')
    if any(m['checkpoint_sha256'] != original_manifest['checkpoint_sha256'] for m in metadata.values()):
        raise ValueError('Original and current FLOWR checkpoints differ')
    source_equality = {}
    for name, config in configs.items():
        files = config['upstream_source_attestation']['after']['files']
        source_equality[name] = {k: files[k]['sha256'] == original_manifest['upstream_files'][k]
                               for k in ['flowr/models/fm_pocket.py', 'flowr/models/integrator.py']}
        if not all(source_equality[name].values()):
            raise ValueError('Original and current upstream core source differ')
    params = ['integration_steps', 'ode_sampling_strategy', 'categorical_strategy', 'cat_sampling_noise_level',
              'cat_noise_euler_guard', 'corrector_iters', 'use_cosine_scheduler', 'rotation_alignment', 'permutation_alignment']
    original_config = read_json(args.original_root/'results'/args.original_campaign/'config.json')
    parameter_equality = {name: {k: c['flowr_args'][k] == original_config['flowr_args'][k] for k in params}
                          for name, c in configs.items()}
    if not all(all(v.values()) for v in parameter_equality.values()):
        raise ValueError('Native generation parameters differ')
    args.output.mkdir(parents=True, exist_ok=True)
    write_table(args.output/'teacher_coverage_by_batch.csv', frame)
    write_table(args.output/'teacher_coverage_by_event.csv', curves)
    write_json(args.output/'audit.json', {'schema': 'analyst-label-and-native-suffix-audit-1.0',
        'code_sha256': digest(__file__), 'threshold': args.threshold, 'deployed': metadata,
        'same_original_teacher_clouds_and_scores': same_teachers, 'native_core_source_equality': source_equality,
        'native_parameter_equality': parameter_equality, 'native_parameters': {k: original_config['flowr_args'][k] for k in params},
        'suffix_execution': suffix, 'source_audit': source_audit, 'original_selection_boundaries': boundaries,
        'teacher_donor_batches': sorted(donor_batches),
        'original_generation_source_sha256': digest(args.original_root/'provenance/instrumented_generate_selective.py'),
        'trace_inputs': trace_inputs, 'metrics_sha256': digest(args.steer_metrics),
        'matching_semantics': 'Original atom-slot order, receptor-fixed coordinates; all aliases within 1e-5 A retained; labels match recorded head within 4e-6; coverage does not measure the attraction from nearby unselected clouds',
        'statistical_semantics': 'Descriptive retrospective coverage; final slots, coordinate aliases and repeated stages are not independent replicates'})
    print({'original_batches': len(source_audit), 'suffix_runs': len(suffix), 'coverage_events': len(curves),
           'same_R11_R26_teachers': same_teachers,
           'coverage_first': float(curves.slot_coverage_fraction.iloc[0]),
           'coverage_last': float(curves.slot_coverage_fraction.iloc[-1])})


if __name__ == '__main__':
    main()
