"""Discovery-only retained-ancestor references; no neural surrogate or time bins."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from ..io import read_json, write_table, digest, clean
from ..trajectory_source import open_trajectory, trajectory_paths


FEATURES = ['ck2:A:ASN117::hetero_distance_softmin',
            'ck2:A:VAL116::hetero_distance_softmin']


def write_json(path, value):
    """Canonical LF artifacts keep Git deployment provenance identical on Windows/Linux."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',
                    encoding='utf-8',newline='\n')


def measure_patch(x, atomics, mask, catalog, features=FEATURES):
    """Same normalized typed soft-min as analysis; absent types remain NaN."""
    x = np.asarray(x, dtype=float); mask = np.asarray(mask, dtype=bool)
    values = []
    for feature in features:
        spec = catalog['features'][feature]
        eligible = mask & np.isin(atomics, [catalog['atom_vocabulary'][e] for e in spec['elements']])
        points = np.array(catalog['regions'][spec['region']]['points_A'])
        distance = np.linalg.norm(x[:, :, None] - points[None, None], axis=-1)
        tau = spec['temperature_A']; n = eligible.sum(1)*len(points)
        value = -tau*(logsumexp(np.where(eligible[..., None], -distance/tau, -np.inf), axis=(1, 2))
                      - np.log(np.maximum(n, 1)))
        value[n == 0] = np.nan
        values.append(value)
    return np.stack(values, axis=-1)


def retained_copies(offspring, parents):
    copies = np.zeros_like(offspring, dtype=np.int64); copies[-1] = offspring[-1]
    for i in range(len(copies)-2, -1, -1):
        copies[i] = np.bincount(parents[i+1], weights=copies[i+1], minlength=copies.shape[1]).astype(np.int64)
    if not np.all(copies.sum(1) == copies.shape[1]):
        raise ValueError('Ancestor copy mass is not conserved')
    return copies


def regularize_covariance(covariance, shrinkage=.1, std_floor=.1):
    if not 0 <= shrinkage <= 1 or std_floor <= 0:
        raise ValueError('Invalid covariance regularization')
    diagonal = np.eye(covariance.shape[-1])*np.diagonal(covariance, axis1=-2, axis2=-1)[..., None, :]
    mixed = (1-shrinkage)*covariance + shrinkage*diagonal
    values, vectors = np.linalg.eigh(mixed)
    return (vectors*np.maximum(values, std_floor**2)[..., None, :]) @ vectors.swapaxes(-1, -2)


def fit_local_stages(times, centers, covariance, tolerance=.1):
    """Global deterministic knot search using every event, not subinterval statistics.

    Interpolation is local shape-preserving cubic. Add the event with largest
    Mahalanobis reconstruction error until all batch representatives meet the
    declared tolerance. No uncertainty/causality inference comes from interpolation.
    """
    times = np.asarray(times); centers = np.asarray(centers)
    if tolerance <= 0 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid interpolation grid')
    knots = [0, len(times)-1]; precision = np.linalg.inv(covariance)
    while True:
        fitted = PchipInterpolator(times[knots], centers[knots], axis=0)(times)
        residual = fitted-centers
        error = np.sqrt(np.maximum(np.einsum('tbf,tfg,tbg->tb', residual, precision, residual), 0))
        worst = error.max(1); worst[knots] = 0
        if worst.max() <= tolerance or len(knots) == len(times):
            break
        knots = sorted(knots + [int(worst.argmax())])
    return knots, {'maximum_whitened_error': float(error.max()),
                   'rms_whitened_error': float(np.sqrt(np.mean(error**2))),
                   'tolerance': tolerance, 'n_segments': len(knots)-1,
                   'fit_meaning': 'Discovery interpolation approximation, not heldout prediction or a learned causal law'}


def build_references(analysis, dataset, output, tolerance=.1):
    analysis, dataset, output = map(Path, (analysis, dataset, output))
    if (output/'reference_packet.json').exists():
        raise FileExistsError('Use a fresh reference output')
    cfg = read_json(analysis/'config.json'); catalog = read_json(analysis/'feature_catalog.json')
    if cfg['time_analysis'] != 'continuous_window' or cfg['evidence_arm'] != 'single':
        raise ValueError('Requires complete continuous single-target discovery')
    batches = sorted(cfg['discovery_batches']); campaign = dataset/'results'/cfg['campaign']
    source_cfg = read_json(campaign/'config.json'); scale = source_cfg['coord_scale']
    fields, sources, records = {}, [], []
    # Explicitly exclude validation, heldout and t=1 before any structural reads.
    for path in trajectory_paths(campaign):
        arm = path.parent.parent.name; batch = int(path.parent.name.split('_')[1])
        if arm not in ('single', 'unguided') or batch not in batches:
            continue
        frame = np.array(read_json(campaign/f'frame_batch_{batch:03d}.json')['target_com'])
        with open_trajectory(path) as z:
            times = np.round(z['score_time'][:, 0].astype(float), 6)
            steps = np.flatnonzero((times >= 0) & (times <= .5))
            if arm == 'single' and not np.all(z['resampled'][steps]):
                raise ValueError('Expected a selection event at every reference node')
            x = z['predicted_coords'][steps].astype(float)*scale + frame[None, :, None, :]
            atomics = z['predicted_atomics'][steps]; mask = z['mask'][steps]
            values = measure_patch(x.reshape(-1, x.shape[2], 3), atomics.reshape(-1, atomics.shape[2]),
                                   mask.reshape(-1, mask.shape[2]), catalog).reshape(len(steps), x.shape[1], -1)
            counts = retained_copies(z['offspring_count'][steps], z['parent_slot'][steps]) if arm == 'single' else None
            fields[arm, batch] = (times[steps], values, counts)
            for i, time in enumerate(times[steps]):
                for slot, value in enumerate(values[i]):
                    records.append({'arm': arm, 'batch': batch, 'step': int(steps[i]), 'time': time,
                        'slot': slot, 'window_end_copies': int(counts[i, slot]) if counts is not None else None,
                        **dict(zip(FEATURES, value))})
        sources.append({'path': path.relative_to(dataset).as_posix(), 'sha256': digest(path)})
    if set(fields) != {(a,b) for a in ('single','unguided') for b in batches}:
        raise ValueError('Incomplete paired discovery batches')
    times = fields['single', batches[0]][0]
    if not np.array_equal(times, np.round(np.arange(51)*.01, 6)):
        raise ValueError('Expected the audited 51-event selection grid')
    centers, covariances, backgrounds, coverage = [], [], [], []
    for batch in batches:
        ts, value, counts = fields['single', batch]
        tu, background, _ = fields['unguided', batch]
        if not np.array_equal(ts, times) or not np.array_equal(tu, times):
            raise ValueError('Unmatched event times')
        finite = np.isfinite(value).all(2)
        weights = counts*finite
        if np.any(weights.sum(1) == 0):
            raise ValueError('A retained prototype lacks conditional atom support')
        centers.append((np.nan_to_num(value)*weights[..., None]).sum(1)/weights.sum(1)[:, None])
        batch_cov, batch_mean = [], []
        for bg in background:
            usable = bg[np.isfinite(bg).all(1)]
            if len(usable) < 3: raise ValueError('Insufficient background covariance support')
            batch_cov.append(np.cov(usable, rowvar=False, ddof=1)); batch_mean.append(usable.mean(0))
        covariances.append(batch_cov); backgrounds.append(batch_mean)
        coverage.append({'batch': batch, 'retained_copy_coverage_by_time': (weights.sum(1)/counts.sum(1)).tolist()})
    # Equal independent batch weights; no clone-dependent mixture weights.
    centers = np.stack(centers, axis=1)
    covariance_raw = np.mean(covariances, axis=0)
    covariance = regularize_covariance(covariance_raw)
    knots, audit = fit_local_stages(times, centers, covariance, tolerance)
    from .launcher import INPUT_FILES
    packet = {'schema_version': 'retained-reference-1.0', 'split': 'discovery',
        'features': FEATURES, 'times': times.tolist(), 'batches': batches,
        'centers_A': centers.tolist(), 'covariance_A2': covariance.tolist(),
        'background_mean_A': np.mean(backgrounds, axis=0).tolist(),
        'knots': knots, 'interpolation_audit': audit, 'coverage': coverage,
        'regularization': {'shrinkage_to_diagonal': .1, 'minimum_eigen_sd_A': .1,
            'origin': 'Pilot numerical assumptions; covariance is equal-batch mean of same-time within-unguided-batch covariances'},
        'required_input_sha256': {n: digest(dataset/'inputs'/n) for n in INPUT_FILES},
        'sources': sources, 'source_catalog_sha256': digest(analysis/'feature_catalog.json'),
        'source_analyst_sha256': digest(analysis/'agents/Analyst.continuous.response.json'),
        'limitations': ['Each prototype is one batch retained-copy mean, not an identified molecular pose mode.',
                       'Two distances cannot reconstruct all selected structures or chemical graphs.',
                       'Targets are retrospective selection associations, not physical optima.']}
    write_json(output/'reference_packet.json', packet)
    write_table(output/'discovery_patch_candidates.parquet', records)
    regions = {catalog['features'][f]['region'] for f in FEATURES}
    catalog['features'] = {f: catalog['features'][f] for f in FEATURES}
    catalog['regions'] = {r: catalog['regions'][r] for r in regions}
    write_json(output/'reward_catalog.json', catalog)
    return packet


def compile_program(reference_dir, designer_path, output):
    reference_dir, designer_path, output = map(Path, (reference_dir, designer_path, output))
    packet = read_json(reference_dir/'reference_packet.json')
    decision = read_json(designer_path)
    if decision.get('schema_version') != 'multistage-designer-decision-1.0':
        raise ValueError('Wrong Designer contract')
    if decision['provenance']['deterministic_prototype']['sha256'] != digest(reference_dir/'reference_packet.json'):
        raise ValueError('Designer reference binding mismatch')
    knots = packet['knots']
    program = {'schema_version': 'live-multistage-1.0', 'program_id': 'ck2_retained_patch_multistage_v1',
        'representation': 'predicted_endpoint_world_A', 'features': packet['features'],
        'observation_window': [0., .5], 'control_window': [0., 1.],
        'knot_times': np.array(packet['times'])[knots].tolist(),
        'knot_centers_A': np.array(packet['centers_A'])[knots].tolist(),
        'covariance_times': packet['times'], 'covariance_A2': packet['covariance_A2'],
        'mixture_weights': [1/len(packet['batches'])]*len(packet['batches']),
        'pseudo_huber_delta': 1., 'mixture_temperature': 1.,
        'continuation': 'Freeze t=.5 centers and covariance for .5<t<1; an unvalidated maintenance hypothesis',
        'interpolation_audit': packet['interpolation_audit'], 'regularization': packet['regularization'],
        'required_input_sha256': packet['required_input_sha256'],
        'source_reference_sha256': digest(reference_dir/'reference_packet.json'),
        'designer_decision_sha256': digest(designer_path),
        'source_analyst_sha256': packet['source_analyst_sha256'],
        'categorical_policy': 'Detached live endpoint argmax N/O/S mask; reevaluated each step; absent mask unavailable',
        'independent_validation': 'New generation batches, never reference fitting',
        'internal_weight_policy': 'One correlated two-observable patch; covariance handles units/correlation; equal independent batch mixture weights',
        'sensitivity': 'dR/dz=-sum_b responsibility_b * precision*(z-center_b)/sqrt(1+Mahalanobis_squared)',
        'constraints': {'max_atom_step_A': .025, 'max_pair_distance_change_A': .05,
                        'severe_receptor_clash_A': 1.2, 'max_cumulative_rms_A': 2.5,
                        'backtrack_attempts': 5},
        'pilot_native_rms_ratios': [.05, .15, .3],
        'parameter_status': 'Pilot hypotheses; no guarantee of affinity or chemical improvement'}
    if not isinstance(decision, dict) or not decision:
        raise ValueError('Missing Designer decision record')
    write_json(output/'reward_program.json', program)
    write_json(output/'reward_catalog.json', read_json(reference_dir/'reward_catalog.json'))
    return program
