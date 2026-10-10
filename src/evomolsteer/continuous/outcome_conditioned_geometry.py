"""Equal-family spatial contrasts conditional on final quality and online score.

Prepared extension: call only after the foundation label trial is evaluated.
No individual coordinate/feature expansion is persisted. Natural-copy parents
are pooled once and whole generation batches provide inferential replicates.
"""
from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths, pocket_input_path
from .affinity_geometry import names, numpy_geometry
from .coordinate_mining import bh
from .path_graph import validate_lineage
from .terminal_outcome import observed_support


def matched_branch_effect(features, online, parents, grandparent_map, parent_quality, tolerance):
    """One observation per immediate parent and equal total weight per ancestor."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError('Finite nonnegative online-score matching tolerance required')
    branches = []
    for parent in np.unique(parents):
        ids = parents == parent
        quality = float(parent_quality[parent])
        if not np.isfinite(quality):
            continue
        z = features[ids]
        average = z[0] if np.array_equal(z, np.broadcast_to(z[0], z.shape)) else z.mean(0)
        branches.append((int(parent), int(grandparent_map[parent]), average, float(online[ids].mean()), quality))
    effects, matched_gaps, final_gaps = [], [], []
    pair_count = 0
    zero_count = 0
    for ancestor in sorted({v[1] for v in branches}):
        family = [v for v in branches if v[1] == ancestor]
        candidates = []
        for a in range(len(family)):
            for b in range(a+1, len(family)):
                x, y = family[a], family[b]
                gap = abs(x[3]-y[3])
                if gap <= tolerance+1e-12 and abs(x[4]-y[4]) > 1e-12:
                    candidates.append((gap, x[0], y[0], a, b))
        used, local = set(), []
        # Deterministic closest-score, non-reused natural branches; pairing never
        # depends on which final-quality contrast has the greatest magnitude.
        for gap, _, _, a, b in sorted(candidates):
            if a in used or b in used:
                continue
            used.update((a, b))
            x, y = family[a], family[b]
            if x[4] < y[4]:
                x, y = y, x
            diff = x[2]-y[2]
            local.append(diff)
            zero_count += int(np.max(np.abs(diff)) <= 1e-12)
            pair_count += 1
            matched_gaps.append(gap)
            final_gaps.append(x[4]-y[4])
        if local:
            effects.append(np.mean(local, axis=0))
    effect = np.mean(effects, axis=0) if effects else np.full(features.shape[1], np.nan)
    return effect, {'matched_ancestors': len(effects), 'matched_pairs': pair_count,
        'zero_coordinate_feature_pairs': zero_count,
        'mean_online_gap': float(np.mean(matched_gaps)) if matched_gaps else None,
        'mean_final_gap': float(np.mean(final_gaps)) if final_gaps else None}


def receptor_points(path):
    return np.array([[float(s[30:38]), float(s[38:46]), float(s[46:54])]
        for s in Path(path).read_text().splitlines()
        if s[:6].strip() == 'ATOM' and s[76:78].strip() != 'H'], dtype=float)


def coordinate_observables(endpoint, current, proposal, landmarks, origin, receptor, time, dt):
    fields = names(landmarks)
    values = [numpy_geometry(endpoint, landmarks, origin)]
    displacement = proposal-current
    drift = (endpoint-current)/(1-time)
    values += [np.column_stack([np.sqrt(np.sum(displacement**2, axis=-1).mean(1)),
        np.sqrt(np.sum(displacement**2, axis=-1).mean(1))/dt,
        np.sqrt(np.sum(drift**2, axis=-1).mean(1))])]
    fields += ['proposal_displacement_RMS_A', 'proposal_speed_A_per_t', 'forecast_flow_RMS_A_per_t']
    assignment = np.argmin(np.sum((receptor[:, None]-landmarks[None])**2, axis=-1), axis=1)
    local = []
    for region in range(len(landmarks)):
        points = receptor[assignment == region]
        if not len(points):
            raise ValueError('Region requires observed receptor points')
        d = np.sqrt(np.sum((endpoint[:, :, None]-points[None, None])**2, axis=-1)+1e-20)
        occupancy = np.exp(-.5*np.sum((current-landmarks[region])**2, axis=-1)/4**2)
        radial = current-landmarks[region]
        direction = radial/np.sqrt(np.sum(radial**2, axis=-1, keepdims=True)+1e-20)
        radial_speed = np.sum(drift*direction, axis=-1)
        local.append(np.column_stack([np.exp(-.5*((d-3)/1.)**2).mean((1, 2)),
            np.exp(-.5*((d-5)/1.)**2).mean((1, 2)),
            np.sum(occupancy*radial_speed, axis=1)/np.maximum(occupancy.sum(1), 1e-12)]))
        fields += [f'region_{region:02d}_{v}' for v in ('receptor_contact3', 'receptor_contact5', 'radial_forecast_speed')]
    values.append(np.concatenate(local, axis=1))
    return np.concatenate(values, axis=1), fields


def analyze(dataset, campaign, labels, evidence, output, tolerance=.25):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    packet = read_json(evidence)
    lab = pd.read_parquet(labels)
    ref = json.loads(gzip.decompress(Path(packet['reference_path']).read_bytes()))
    root = Path(dataset)
    source = root/'results'/campaign
    cfg = read_json(source/'config.json')
    points, origin = np.array(ref['landmarks_A']), np.array(ref['origin_A'])
    receptor = receptor_points(pocket_input_path(root, 'target_protein'))
    receptor = receptor[np.min(np.sum((receptor[:, None]-points[None])**2, axis=-1), axis=1) <= 8**2]
    paths = {int(p.parent.name.split('_')[1]): p for p in trajectory_paths(source) if p.parent.parent.name == 'single'}
    rows, support = [], []
    for batch in packet['donor_batches']:
        expected = next(v['sha256'] for v in ref['sources'] if v['batch'] == batch)
        if digest(paths[batch]) != expected:
            raise ValueError('Matched ancestry is not the bound original source')
        com = np.array(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:, None]
        with open_trajectory(paths[batch]) as tr:
            clock, state, selected, _, _, flags = validate_lineage(tr)
            steps, window = observed_support(clock, state, flags, packet['score_window'])
            if window != packet['window']:
                raise ValueError('Score/state support differs from foundation')
            for step in steps:
                if step < 2:
                    continue
                raw = [np.asarray(tr[name][step], float)*cfg['coord_scale']+com
                       for name in ('predicted_coords', 'current_coords', 'proposal_coords')]
                z, fields = coordinate_observables(*raw, points, origin, receptor, float(clock[step]), float(state[step]-clock[step]))
                previous = lab[(lab.batch == batch)&(lab.step == step-1)].sort_values('slot')
                if not len(previous):
                    support.append({'batch': batch, 'time': round(float(clock[step]), 6),
                        'matched_ancestors': 0, 'matched_pairs': 0, 'zero_coordinate_feature_pairs': 0,
                        'mean_online_gap': None, 'mean_final_gap': None, 'skip_reason': 'Previous label event outside supplied feature window'})
                    continue
                if not np.array_equal(previous.slot, np.arange(z.shape[0])):
                    raise ValueError('Exact previous-event ancestor label alignment required')
                effect, diag = matched_branch_effect(z, np.asarray(tr['pic50_on'][step], float),
                    selected[step-1], selected[step-2], previous.terminal_mean.to_numpy(float), tolerance)
                support.append({'batch': batch, 'time': round(float(clock[step]), 6), **diag})
                rows.extend({'batch': batch, 'time': round(float(clock[step]), 6), 'feature': name, 'effect': value}
                            for name, value in zip(fields, effect))
    out.mkdir(parents=True)
    curves = pd.DataFrame(rows)
    curves.to_parquet(out/'batch_event_effects.parquet', compression=None, index=False)
    pd.DataFrame(support).to_csv(out/'matched_support.csv', index=False)
    whole = curves.groupby(['batch', 'feature'], sort=True).effect.mean().unstack('batch')
    results = []
    for feature, row in whole.iterrows():
        v = row.dropna().to_numpy()
        p = float(ttest_1samp(v, 0).pvalue) if len(v) >= 3 and np.ptp(v) > 1e-14 else (1. if len(v) >= 3 else None)
        results.append({'id': 'outcome/matched/'+feature, 'feature': feature, 'n_batches': len(v),
            'effect': float(v.mean()) if len(v) else None, 'p': p,
            'same_sign_fraction': float(np.mean(np.sign(v) == np.sign(v.mean()))) if len(v) else None})
    for item, q in zip(results, bh(np.array([np.nan if e['p'] is None else e['p'] for e in results]))):
        item['q'] = q
    trends = []
    for feature, group in curves.groupby('feature', sort=True):
        series = group.groupby('time').effect.mean().dropna()
        if len(series) >= 5:
            coef = np.polyfit(series.index, series.values, 2)
            trends.append({'feature': feature, 'coefficients': coef.tolist(), 'derivative_coefficients': np.polyder(coef).tolist(),
                'supported_times': series.index.tolist(), 'semantics': 'Sparse observed matched-family trends; missing events not imputed'})
    write_json(out/'trends.json', trends)
    packet['evidence_items'] += [{'id': 'outcome/matched_support', 'total_pairs': sum(s['matched_pairs'] for s in support),
        'supported_batch_events': sum(s['matched_pairs'] > 0 for s in support), 'online_tolerance': tolerance,
        'matching': 'Distinct immediate parents, common grandparent, nearest online score, no pair reuse; equal families and batches'}, *results]
    packet['matched_source_sha256'] = digest(__file__)
    packet['matched_labels_sha256'] = digest(labels)
    packet['limitations'] += ['Matched sparse survivor branches remain observational; contact features are spatial proxies, not interaction energies']
    write_json(out/'evidence.json', packet)
    return packet
