"""Decoded-outcome labels and clock-explicit compact coordinate teachers.

No predictor is fitted. Final slots are observed Steer descendants, not
independent native rollouts. Extinction is censored. Coordinates stay in the
immutable trajectory; only the small executable teacher library duplicates them.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from ..io import clean, digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths
from .affinity_geometry import names, numpy_geometry
from .branch_mutation import conditional_teacher_contrast, compose_ancestors
from .coordinate_mining import bh
from .elite_path_credit import terminal_ancestors
from .path_graph import validate_lineage


def outcome_labels(selected, metrics, threshold):
    n = np.asarray(selected).shape[1]
    m = metrics.sort_values('slot').reset_index(drop=True)
    if len(m) != n or not np.array_equal(m.slot, np.arange(n)) or m.valid_connected.dtype != bool:
        raise ValueError('One typed decoded terminal observation per slot required')
    score = m.pic50_on_rescore.to_numpy(float)
    valid = m.valid_connected.to_numpy() & np.isfinite(score)
    ancestor = terminal_ancestors(selected)
    result = []
    for step, slots in enumerate(ancestor[:-1]):
        rows = []
        for node in range(n):
            ids = np.flatnonzero(slots == node)
            good = ids[valid[ids]]
            graph_mean = m.iloc[good].groupby('smiles', sort=True).pic50_on_rescore.mean()
            rows.append({'slot': node, 'observed_n': len(ids), 'valid_n': len(good),
                'unique_graph_n': len(graph_mean), 'future_observed': bool(len(ids)),
                'terminal_mean': float(graph_mean.mean()) if len(graph_mean) else np.nan,
                'terminal_p75': float(graph_mean.quantile(.75)) if len(graph_mean) else np.nan,
                'tail_fraction': float((score[good] >= threshold).sum()/len(ids)) if len(ids) else np.nan,
                'valid_fraction': len(good)/len(ids) if len(ids) else np.nan,
                'terminal_slot_ids': ','.join(map(str, ids))})
        result.append(pd.DataFrame(rows))
    return result


def observed_support(clock, state, resampled, score_window):
    lo, hi = map(float, score_window)
    if not 0 <= lo < hi < 1:
        raise ValueError('Dynamic nonempty score window required')
    steps = np.flatnonzero(resampled & (clock >= lo-2e-6) & (clock <= hi+2e-6))
    if not len(steps) or not np.isclose(clock[steps[0]], lo, atol=2e-6) or not np.isclose(clock[steps[-1]], hi, atol=2e-6):
        raise ValueError('Window must bind to actual selection endpoints')
    if np.any(np.diff(steps) != 1):
        raise ValueError('Contiguous observed support required')
    return steps, [lo, round(float(state[steps[-1]]), 6)]


def utility(labels, mode, tail_weight=0., parent_quality=None, shrinkage=2.):
    if mode in ('mean', 'instantaneous'):
        return labels.terminal_mean.to_numpy(float)
    if mode == 'distribution':
        return labels.terminal_mean.to_numpy(float)+tail_weight*labels.tail_fraction.to_numpy(float)
    if mode == 'p75':
        return labels.terminal_p75.to_numpy(float)
    if mode == 'hierarchical':
        value = labels.terminal_mean.to_numpy(float)
        count = labels.unique_graph_n.to_numpy(float)
        parent = np.asarray(parent_quality, float)
        known = np.isfinite(value)
        if parent.shape != value.shape or not np.isfinite(shrinkage) or shrinkage < 0 or not np.isfinite(parent[known]).all():
            raise ValueError('Known parent outcomes and finite nonnegative heuristic shrinkage required')
        result = np.full_like(value, np.nan)
        result[known] = (count[known]*value[known]+shrinkage*parent[known])/(count[known]+shrinkage)
        return result
    raise ValueError('Registered decoded-outcome label required')


def choose_teachers(xyz, scores, budget):
    """Score-ranked spatial coverage; never rank by a best terminal descendant."""
    if not 1 <= budget <= 50:
        raise ValueError('Bounded per-batch teacher budget required')
    ordered = np.argsort(-scores, kind='stable')
    chosen = []
    for node in ordered:
        if not np.isfinite(scores[node]):
            continue
        if not chosen or min(np.sqrt(np.mean((xyz[node]-xyz[k])**2)) for k in chosen) > .01:
            chosen.append(int(node))
        if len(chosen) == budget:
            break
    if not chosen:
        raise ValueError('No observed valid terminal outcome in donor frame')
    return chosen


def build(dataset, campaign, metrics_path, baseline_path, output, score_window=(0., .5),
          mode='mean', budget=2, threshold=8.258901977539063, tail_weight=.5,
          branch_mode='instantaneous', score_tolerance=.25, shrinkage=2.):
    root, out = Path(dataset), Path(output)
    if out.exists():
        raise FileExistsError(out)
    base = json.loads(gzip.decompress(Path(baseline_path).read_bytes()))
    metrics = pd.read_csv(metrics_path)
    if metrics.duplicated(['batch', 'slot']).any():
        raise ValueError('Ambiguous terminal metrics')
    source = root/'results'/campaign
    cfg = read_json(source/'config.json')
    paths = {int(p.parent.name.split('_')[1]): p for p in trajectory_paths(source) if p.parent.parent.name == 'single'}
    batches = list(base['discovery_batches'])
    features = names(base['landmarks_A'])+['coordinate_displacement_RMS_A', 'coordinate_speed_A_per_t']
    frames, label_rows, events, sources, effect_curves = {}, [], [], [], []
    control_window = None
    coverage = []
    for batch in batches:
        path = paths[batch]
        com = np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:, None, :]
        with open_trajectory(path) as tr:
            clock, state, selected, _, _, resampled = validate_lineage(tr)
            steps, window = observed_support(clock, state, resampled, score_window)
            if control_window is not None and window != control_window:
                raise ValueError('Donor clocks disagree')
            control_window = window
            labels = outcome_labels(selected, metrics[metrics.batch == batch], threshold)
            for step in steps:
                time = round(float(clock[step]), 6)
                f = frames.setdefault(time, {'time': time, 'state_time': float(state[step]),
                    'teacher_endpoint_A': [], 'teacher_scores': [], 'teacher_batches': [],
                    'teacher_slots': [], 'teacher_outcomes': [], 'teacher_base_log_weight': [],
                    'teacher_contrast_direction_unit': [], 'teacher_contrast_confidence': [],
                    'teacher_contrast_atom_weight': [], 'teacher_contrast_provenance': []})
                xyz = np.asarray(tr['predicted_coords'][step], float)*cfg['coord_scale']+com
                current = np.asarray(tr['current_coords'][step], float)*cfg['coord_scale']+com
                proposal = np.asarray(tr['proposal_coords'][step], float)*cfg['coord_scale']+com
                online = np.asarray(tr['pic50_on'][step], float)
                lab = labels[step].copy()
                if step:
                    parent_quality = labels[step-1].terminal_mean.to_numpy(float)[selected[step-1]]
                else:
                    bm = metrics[(metrics.batch == batch)&metrics.valid_connected&metrics.pic50_on_rescore.notna()]
                    parent_quality = np.full(len(lab), bm.groupby('smiles').pic50_on_rescore.mean().mean())
                quality = utility(lab, mode, tail_weight, parent_quality, shrinkage)
                rank_score = online if mode == 'instantaneous' else quality
                chosen = choose_teachers(xyz, rank_score, budget)
                saved = next((v for v in base['frames'] if abs(v['time']-time) <= 2e-6), None) if mode == 'instantaneous' else None
                original_indices = []
                if saved is not None:
                    chosen = []
                    for k in np.flatnonzero(np.asarray(saved['teacher_batches']) == batch):
                        error = np.sqrt(np.mean((xyz-np.asarray(saved['teacher_endpoint_A'][k]))**2, axis=(1, 2)))
                        if error.min() > 1e-5:
                            raise ValueError('Original control teacher is not a bound source coordinate')
                        chosen.append(int(error.argmin()))
                        original_indices.append(int(k))
                lab['batch'], lab['step'], lab['score_time'], lab['state_time'] = batch, int(step), time, float(state[step])
                lab['online_score'] = online
                lab['utility'] = quality
                label_rows.append(lab)
                displacement = np.sqrt(np.sum((proposal-current)**2, axis=-1).mean(1))
                z = np.column_stack([numpy_geometry(xyz, np.asarray(base['landmarks_A']), np.asarray(base['origin_A'])),
                                     displacement, displacement/(state[step]-clock[step])])
                known = np.isfinite(quality)
                effect = np.full(len(features), np.nan)
                if known.sum() >= 2 and np.ptp(quality[known]) > 1e-12:
                    lo, hi = np.quantile(quality[known], [.25, .75])
                    high, low = known & (quality >= hi), known & (quality <= lo)
                    if hi > lo:
                        effect = z[high].mean(0)-z[low].mean(0)
                effect_curves.extend({'batch': batch, 'time': time, 'feature': name, 'effect': value}
                                     for name, value in zip(features, effect))
                elite_slots = lab[lab.tail_fraction > 0].slot.to_numpy(int)
                coverage.append({'batch': batch, 'time': time, 'known_nodes': int(known.sum()),
                    'elite_ancestor_nodes': len(elite_slots), 'covered_elite_ancestors': len(set(chosen).intersection(elite_slots))})
                rho = float(spearmanr(online[known], quality[known]).statistic) if known.sum() >= 3 and np.ptp(quality[known]) else np.nan
                events.append({'batch': batch, 'time': time, 'known': int(known.sum()), 'censored': int((lab.observed_n == 0).sum()),
                    'instant_final_spearman': rho, 'teachers': len(chosen)})
                for teacher_index, slot in enumerate(chosen):
                    old_k = original_indices[teacher_index] if original_indices else None
                    f['teacher_endpoint_A'].append(saved['teacher_endpoint_A'][old_k] if old_k is not None else xyz[slot].tolist())
                    f['teacher_scores'].append(saved['teacher_scores'][old_k] if old_k is not None else float(rank_score[slot]))
                    f['teacher_batches'].append(batch)
                    f['teacher_slots'].append(int(slot))
                    outcome = lab.iloc[slot].to_dict()
                    f['teacher_outcomes'].append(outcome)
                    f['teacher_base_log_weight'].append(float(-np.log(len(chosen))))
                    direction, confidence, weight = np.zeros_like(xyz[slot]), 0., np.ones(xyz.shape[1])
                    provenance = {'lag2_observed': False, 'raw_direction_RMS_A': 0.}
                    if step >= 2:
                        parents = selected[step-1]
                        ancestors = compose_ancestors(parents, selected[step-2])
                        branch_score = online if branch_mode == 'instantaneous' else quality
                        eligible = np.isfinite(branch_score)
                        if branch_mode == 'outcome_matched':
                            eligible &= np.abs(online-online[slot]) <= score_tolerance
                        contrast = conditional_teacher_contrast(xyz[slot], branch_score[slot], xyz[eligible],
                            branch_score[eligible], parents[eligible], ancestors[eligible], int(parents[slot]), int(ancestors[slot]))
                        direction, confidence, weight = contrast['direction_unit'], contrast['confidence'], contrast['atom_weight']
                        provenance = {'lag2_observed': True, 'raw_direction_RMS_A': contrast['raw_rms_A'],
                            'common_grandparent_slot': int(ancestors[slot]), 'immediate_parent_slot': int(parents[slot]),
                            'lower_immediate_parent_slots': contrast['lower_parent_ids'],
                            'distinct_observed_mutations': contrast['distinct_observed_mutations'],
                            'score_label': branch_mode, 'instant_score_tolerance': score_tolerance if branch_mode == 'outcome_matched' else None}
                    if old_k is not None and 'teacher_contrast_direction_unit' in saved:
                        direction = np.asarray(saved['teacher_contrast_direction_unit'][old_k])
                        confidence = saved['teacher_contrast_confidence'][old_k]
                        weight = np.asarray(saved['teacher_contrast_atom_weight'][old_k])
                        provenance = saved['teacher_contrast_provenance'][old_k]
                    f['teacher_contrast_direction_unit'].append(direction.tolist())
                    f['teacher_contrast_confidence'].append(float(confidence))
                    f['teacher_contrast_atom_weight'].append(weight.tolist())
                    f['teacher_contrast_provenance'].append(provenance)
        sources.append({'batch': batch, 'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    ref = copy.deepcopy(base)
    ref.pop('branch_mutation', None)
    ref.update(window=control_window, score_window=list(score_window), times=sorted(frames), frames=[frames[t] for t in sorted(frames)],
        sources=sources, reference_variant='decoded-outcome-distribution-1.0',
        label_semantics=('Recorded instantaneous on-target affinity head; boundary-only matched control' if mode == 'instantaneous' else
                         'Decoded final valid molecule mean per chemical graph, then equal graph mean per observed ancestor'+
                         ('; heuristic graph-count-weighted shrinkage toward observed parent mean, not a calibrated posterior' if mode == 'hierarchical' else '')),
        teacher_selection={'mode': mode, 'per_batch_budget': budget, 'tail_weight': tail_weight, 'branch_mode': branch_mode,
                           'shrinkage': shrinkage if mode == 'hierarchical' else 0.},
        label_clock=1., threshold_pic50=threshold, future_semantics='Observed Steer descendants; extinct future censored',
        teacher_library={'terminal_metrics_sha256': digest(metrics_path), 'baseline_sha256': digest(baseline_path)},
        time_alignment='Endpoint forecast at score t; control ends at last observed proposal state t+dt')
    out.mkdir(parents=True)
    # No expanded geometry cache, edge table or raw coordinate copies.
    pd.concat(label_rows, ignore_index=True).to_parquet(out/'ancestor_outcomes.parquet', compression=None, index=False)
    curves = pd.DataFrame(effect_curves)
    curves.to_parquet(out/'batch_event_effects.parquet', compression=None, index=False)
    pd.DataFrame(events).to_csv(out/'event_support.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'teacher_coverage.csv', index=False)
    whole = curves.groupby(['batch', 'feature'], sort=True).effect.mean().unstack('batch')
    evidence = []
    for feature, row in whole.iterrows():
        v = row.dropna().to_numpy()
        p = float(ttest_1samp(v, 0).pvalue) if len(v) >= 3 and np.ptp(v) > 1e-14 else (1. if len(v) >= 3 else np.nan)
        evidence.append({'id': 'outcome/geometry/'+feature, 'feature': feature, 'effect': float(v.mean()) if len(v) else None,
                         'n_batches': len(v), 'p': p, 'same_sign_fraction': float(np.mean(np.sign(v) == np.sign(v.mean()))) if len(v) else None})
    q = bh(np.asarray([np.nan if e['p'] is None else e['p'] for e in evidence]))
    for e, value in zip(evidence, q):
        e['q'] = float(value)
    trends = []
    for feature, part in curves.groupby('feature', sort=True):
        average = part.groupby('time').effect.mean().dropna()
        if len(average) >= 5:
            coef = np.polyfit(average.index, average.to_numpy(), 2)
            trends.append({'feature': feature, 'times': average.index.tolist(), 'means': average.tolist(),
                           'quadratic_coefficients': coef.tolist(), 'derivative_coefficients': np.polyder(coef).tolist(),
                           'semantics': 'Descriptive temporal curve; derivative is not a coordinate force'})
    write_json(out/'trends.json', trends)
    reference = out/'reference.json.gz'
    reference.write_bytes(gzip.compress(json.dumps(clean(ref), separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    packet = {'schema_version': 'terminal-outcome-evidence-1.0', 'window': control_window, 'score_window': list(score_window),
        'label_clock': 1., 'mode': mode, 'branch_mode': branch_mode, 'donor_batches': batches,
        'label_semantics': ref['label_semantics'], 'censoring': ref['future_semantics'], 'threshold_pic50': threshold,
        'reference_path': str(reference.resolve()), 'reference_sha256': digest(reference),
        'metrics_sha256': digest(metrics_path), 'source_code_sha256': digest(__file__),
        'evidence_items': [{'id': 'outcome/labels', 'label_source': 'instantaneous_control' if mode == 'instantaneous' else 'decoded_final', 'mode': mode, 'branches': branch_mode},
            {'id': 'outcome/clocks', 'score_window': list(score_window), 'control_window': control_window, 'events': len(frames)},
            {'id': 'outcome/censoring', 'mean_censored_fraction': float(np.mean([e['censored']/50 for e in events])),
             'observations': 'Clones and descendants are not independent; final quality of extinct branches unknown'},
            *evidence],
        'limitations': ['Retrospective Steer-conditioned quality, not native rollout probability',
                        'Whole-window contrasts condition on observed future; not causal feature effects',
                        'No affinity-head derivative, physical-energy inference or chemical-graph gate']}
    write_json(out/'evidence.json', packet)
    write_json(out/'manifest.json', {**{k: packet[k] for k in ['window', 'score_window', 'reference_sha256', 'metrics_sha256', 'source_code_sha256']},
        'sources': sources, 'outputs': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    return packet
