"""Exact observed composition strata for compact conditional geometry references.

Counts condition the reference, never a discrete gradient. This avoids linear
extrapolation of central tensors outside their positive-semidefinite domain.
"""
import copy
import gzip
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..continuous.coordinate_features import regional_observables
from ..io import read_json, digest
from ..storage.trajectory import TrajectoryPackage
from .coordinate_shape import CoordinateShapeReward, dimensionless_moments, time_weights, COMPONENTS
from .prototypes import write_json
from .window_reference import load_reference


def tensor_min_eigenvalue(vector):
    """Unscale svec off-diagonals; each regional block has units A^2."""
    values = []
    for x in np.asarray(vector).reshape(-1, 6):
        matrix = np.diag(x[:3])
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
            matrix[i, j] = matrix[j, i] = x[k]/np.sqrt(2)
        values.append(np.linalg.eigvalsh(matrix)[0])
    return float(min(values))


def conditional_record(v, probability, roots, batch, scale):
    probability = np.asarray(probability, float)
    mass = float(probability.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Conditional moment requires positive finite selected mass')
    w = probability/mass
    record = dimensionless_moments(v, w, scale)
    record.update(source_batch=batch, selected_mass=mass, n_available=len(v),
                  unique_roots=int(np.unique(roots).size), probability_ESS=float(1/(w@w)))
    _, root_ids = np.unique(roots, return_inverse=True)
    root_mass = np.bincount(root_ids, weights=w)
    record['root_weight_ESS'] = float(1/(root_mass@root_mass))
    record['ridge_trace_fraction'] = float(.01*len(scale)/np.trace(record['covariance_dimensionless']))
    record['mean_tensor_min_eigenvalue_A2'] = tensor_min_eigenvalue(record['center_A2'])
    if record['mean_tensor_min_eigenvalue_A2'] < -1e-10:
        raise ValueError('Impossible conditional central tensor')
    return record


def build(dataset, campaign, reference, output, audit_output, min_group_size=3, min_batches=2,
          min_prior_ess=1.5, max_loo_rms=1.0):
    root, reference, output, audit_output = map(Path, (dataset, reference, output, audit_output))
    if output.exists() or audit_output.exists():
        raise FileExistsError('Use immutable new conditional outputs')
    if min_group_size < 3 or min_batches < 2:
        raise ValueError('Declared empirical support cannot be relaxed below 3 records / 2 batches')
    if not np.isfinite(min_prior_ess) or min_prior_ess < 1.5 or not np.isfinite(max_loo_rms) or not 0 < max_loo_rms <= 1.:
        raise ValueError('Finite conservative prior/stability thresholds required')
    ref = load_reference(reference)
    if ref['schema_version'] != 'regional-shape-mixture-1.0' or ref['channel'] != 'NOS':
        raise ValueError('Matched NOS directional parent required')
    source = root/'results'/campaign
    scale = np.asarray(ref['feature_scale_A2']); coord_scale = read_json(source/'config.json')['coord_scale']
    catalog = {'regions': ref['regions'], 'atom_vocabulary': ref['atom_vocabulary']}
    grouped, counts, native_counts, sources, unconditioned_check, reaggregation_check = {}, [], {}, [], [], []
    for batch in ref['batches']:
        path = source/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        native_path = source/'unguided'/f'batch_{batch:03d}'/'trajectory.h5'
        com = np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:, None]
        with TrajectoryPackage(path) as package, TrajectoryPackage(native_path) as native:
            times = np.round(package.read('score_time')[:, 0].astype(float), 6)
            if native.read('resampled').any() or not np.array_equal(times, np.round(native.read('score_time')[:, 0].astype(float), 6)):
                raise ValueError('Matching unselected native control required')
            for i in np.flatnonzero(package.read('resampled')):
                t = float(times[i])
                if t not in ref['times']:
                    continue
                x, end, proposal = [package.read(f'{view}_coords', int(i)).astype(float)*coord_scale+com for view in ('current', 'predicted', 'proposal')]
                atoms, mask = package.read('predicted_atomics', int(i)), package.read('mask', int(i)).astype(bool)
                dt = float(package.read('step_size', int(i)).reshape(-1)[0])
                v, names, _ = regional_observables(x, end, proposal, atoms, mask, catalog, t, dt,
                    ref['spatial_width_A'], 'endpoint', True, 'shape')
                values = v[:, [names.index(f) for f in ref['features']]]
                c = np.column_stack([((atoms == ref['atom_vocabulary'][a]) & mask).sum(1) for a in ('N', 'O', 'S')])
                valid = np.isfinite(values).all(1) & (c.sum(1) >= 2)
                whole_probability = package.read('selection_probability', int(i)).astype(float)
                whole_probability /= whole_probability.sum()
                original_available_mass = float(whole_probability[valid].sum())
                probability = whole_probability[valid]
                probability /= probability.sum()
                values, c = values[valid], c[valid]
                roots = package.read('root_slot', int(i))[valid]
                j = ref['times'].index(t)
                old = next(m for m in ref['frames'][j]['modes'] if m['source_batch'] == batch)
                parent = dimensionless_moments(values, probability, scale)
                difference = max(np.max(np.abs(np.asarray(parent[key])-old[key])) for key in ('center_scaled', 'covariance_dimensionless'))
                if difference > 1e-12:
                    raise ValueError('Parent geometry moments or availability changed')
                unconditioned_check.append(difference)
                combos, membership = np.unique(c, axis=0, return_inverse=True)
                strata = []
                for k, combo in enumerate(combos):
                    use = membership == k
                    record = conditional_record(values[use], probability[use], roots[use], batch, scale)
                    record['original_whole_batch_selected_mass'] = record['selected_mass']*original_available_mass
                    strata.append(record)
                    key = tuple(int(a) for a in combo)
                    counts.append({'batch': batch, 'time': t, 'N': key[0], 'O': key[1], 'S': key[2],
                        'n_available': record['n_available'], 'unique_roots': record['unique_roots'],
                        'selected_mass': record['selected_mass'], 'probability_ESS': record['probability_ESS'],
                        'original_whole_batch_selected_mass': record['original_whole_batch_selected_mass'],
                        'root_weight_ESS': record['root_weight_ESS'], 'ridge_trace_fraction': record['ridge_trace_fraction'],
                        'mean_tensor_min_eigenvalue_A2': record['mean_tensor_min_eigenvalue_A2']})
                    if use.sum() >= min_group_size:
                        grouped.setdefault(t, {}).setdefault(key, []).append(record)
                reconstructed_mean = sum(s['selected_mass']*np.asarray(s['center_A2']) for s in strata)
                reconstructed_covariance = sum(s['selected_mass']*(np.asarray(s['raw_covariance_A4'])+
                    np.outer(np.asarray(s['center_A2'])-reconstructed_mean,np.asarray(s['center_A2'])-reconstructed_mean)) for s in strata)
                reaggregation_error = max(np.max(np.abs(reconstructed_mean-parent['center_A2'])),
                                          np.max(np.abs(reconstructed_covariance-parent['raw_covariance_A4'])))
                if reaggregation_error > 1e-12:
                    raise ValueError('Count strata do not reaggregate to parent raw moments')
                reaggregation_check.append(float(reaggregation_error))
                na, nm = native.read('predicted_atomics', int(i)), native.read('mask', int(i)).astype(bool)
                native_counts[(batch, t)] = np.column_stack([((na == ref['atom_vocabulary'][a]) & nm).sum(1) for a in ('N', 'O', 'S')])
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path)})
    if sources != ref['sources']:
        raise ValueError('Parent/source trajectory hashes differ')
    frames, diagnostics, coverage = [], [], []
    for t in ref['times']:
        combinations = []
        for combo, modes in sorted(grouped.get(t, {}).items()):
            if len(modes) < min_batches:
                continue
            mass = np.asarray([m['selected_mass'] for m in modes]); prior = mass/mass.sum()
            centers = np.asarray([m['center_scaled'] for m in modes]); mean = prior@centers
            loo = np.stack([(np.delete(prior, k)@np.delete(centers, k, 0))/(1-prior[k]) for k in range(len(modes))])
            max_loo = float(np.sqrt(((loo-mean)**2).mean(1)).max())
            ess = float(1/(prior@prior)); accepted = ess >= min_prior_ess and max_loo <= max_loo_rms
            if accepted:
                combinations.append({'counts': list(combo), 'modes': modes, 'mode_prior': prior.tolist()})
            diagnostics.append({'time': t, 'N': combo[0], 'O': combo[1], 'S': combo[2],
                'n_support_batches': len(modes), 'conditional_prior_ESS': ess, 'accepted': accepted,
                'maximum_component_prior': float(prior.max()),
                'max_LOO_center_change_scaled_RMS': max_loo,
                'weighted_ridge_trace_fraction': float(prior@np.asarray([m['ridge_trace_fraction'] for m in modes])),
                'minimum_roots_in_mode': min(m['unique_roots'] for m in modes),
                'minimum_records_in_mode': min(m['n_available'] for m in modes)})
        frames.append({'time': t, 'combinations': combinations})
        supported = {tuple(v['counts']) for v in combinations}
        for batch in ref['batches']:
            c = native_counts[(batch, t)]
            retained_mass = sum(m['selected_mass'] for combination in combinations for m in combination['modes'] if m['source_batch']==batch)
            coverage.append({'batch': batch, 'time': t, 'attempted': len(c),
                'supported_fraction': float(np.mean([tuple(v) in supported for v in c])),
                'discarded_parent_available_selection_mass': float(1-retained_mass),
                'informative_fraction': float(np.mean(c.sum(1) >= 2))})
    new = copy.deepcopy(ref)
    new.update(conditioning='exact observed endpoint N/O/S counts', conditional_frames=frames,
        conditioning_parent_reference_sha256=digest(reference),
        conditional_support={'min_group_size': min_group_size, 'min_batches': min_batches,
                             'min_prior_ESS': min_prior_ess, 'max_LOO_center_change_scaled_RMS': max_loo_rms,
                             'unsupported_rule': 'zero dose; native dynamics continue'},
        conditional_mixture_definition='Empirical selected count-mass prior across equally weighted discovery batches; robust geometry mixture inside each exact count stratum')
    new['selected_mass_denominator'] = 'Per-batch selection probabilities normalized on the parent informative shape subset (NOS>=2), as in R13; original whole-batch masses also retained separately.'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress((json.dumps(new, separators=(',', ':'), allow_nan=False)+'\n').encode(), mtime=0))
    audit_output.mkdir(parents=True)
    for name, rows in [('count_support', counts), ('conditional_diagnostics', diagnostics), ('native_coverage', coverage)]:
        pd.DataFrame(rows).to_parquet(audit_output/(name+'.parquet'), index=False, compression='zstd')
    d = pd.DataFrame(coverage)
    per_time = d.groupby('time').supported_fraction.mean().reindex(ref['times']).to_numpy()
    audit = {'schema_version': 'empirical-composition-conditioning-1.0', 'reference_sha256': digest(output),
        'parent_reference_sha256': digest(reference), 'window': ref['window'], 'times': ref['times'],
        'discovery_batches': ref['batches'], 'geometry_moment_max_difference': max(unconditioned_check),
        'raw_moment_reaggregation_max_difference': max(reaggregation_check),
        'mean_native_support_fraction': float(time_weights(ref['times'])@per_time),
        'native_support_min_max_by_time': [float(per_time.min()), float(per_time.max())],
        'n_supported_time_combinations': sum(v['accepted'] for v in diagnostics),
        'n_candidate_time_combinations': len(diagnostics), 'bytes_reference': output.stat().st_size,
        'selected_sources': sources,
        'native_sources': [{'path': (source/'unguided'/f'batch_{b:03d}'/'trajectory.h5').relative_to(root).as_posix(),
                           'sha256': digest(source/'unguided'/f'batch_{b:03d}'/'trajectory.h5')} for b in ref['batches']],
        'limitations': ['Records/clone frequencies are not independent replicates; root counts are descriptive',
                        'Exact N/O/S counts do not determine bond graph or charge',
                        'Two supporting batches can still be weak and unequal; LOO stability/ESS remain visible',
                        'Composition association is observational, not a discrete force or affinity mechanism'],
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in audit_output.iterdir()}}
    write_json(audit_output/'audit.json', audit)
    return audit


class CoordinateCountConditionedShapeReward(CoordinateShapeReward):
    def __init__(self, program, reference):
        super().__init__(program, reference)
        policy = reference['conditional_support']
        if (policy['min_group_size'] < 3 or policy['min_batches'] < 2 or
            not np.isfinite(policy['min_prior_ESS']) or policy['min_prior_ESS'] < 1.5 or
            not np.isfinite(policy['max_LOO_center_change_scaled_RMS']) or
            not 0 < policy['max_LOO_center_change_scaled_RMS'] <= 1.):
            raise ValueError('Conservative conditional support policy required')
        if len(reference['conditional_frames']) != len(self.times):
            raise ValueError('Full conditional window required')
        for t, frame in zip(self.times, reference['conditional_frames']):
            if frame['time'] != t:
                raise ValueError('Matched conditional times required')
            seen = set()
            for combo in frame['combinations']:
                key = tuple(combo['counts'])
                if len(key) != 3 or any(int(c) != c or c < 0 for c in key) or key in seen:
                    raise ValueError('Distinct nonnegative exact count combinations required')
                seen.add(key)
                prior = np.asarray(combo['mode_prior'])
                if len(prior) != len(combo['modes']) or (prior <= 0).any() or not np.isclose(prior.sum(), 1.):
                    raise ValueError('Normalized positive empirical count prior required')
                modes = combo['modes']
                batches = [m['source_batch'] for m in modes]
                if (len(set(batches)) != len(batches) or len(batches) < policy['min_batches'] or
                    set(batches)-set(reference['batches']) or
                    any(m['n_available'] < policy['min_group_size'] for m in modes)):
                    raise ValueError('Same-node independent batch support required')
                masses = np.asarray([m['selected_mass'] for m in modes])
                if not np.isfinite(masses).all() or (masses <= 0).any() or not np.allclose(prior, masses/masses.sum()):
                    raise ValueError('Conditional prior must match empirical selected mass')
                centers = np.asarray([m['center_scaled'] for m in modes]); mean = prior@centers
                loo = np.stack([(np.delete(prior,k)@np.delete(centers,k,0))/(1-prior[k]) for k in range(len(modes))])
                if (1/(prior@prior) < policy['min_prior_ESS']-1e-12 or
                    np.sqrt(((loo-mean)**2).mean(1)).max() > policy['max_LOO_center_change_scaled_RMS']+1e-12):
                    raise ValueError('Conditional prior ESS or leave-one-batch-out stability failed')
                for mode in combo['modes']:
                    covariance = np.asarray(mode['covariance_dimensionless']); center = np.asarray(mode['center_scaled'])
                    if center.shape != self.scale.shape or covariance.shape != (len(self.scale),len(self.scale)) or not np.isfinite(center).all() or not np.isfinite(covariance).all() or not np.allclose(covariance,covariance.T,atol=1e-12,rtol=1e-12):
                        raise ValueError('Finite symmetric conditional shape moments required')
                    np.linalg.cholesky(covariance)
                    if tensor_min_eigenvalue(center*self.scale) < -1e-10:
                        raise ValueError('Conditional tensor mean must be physically PSD')

    def __call__(self, x, atoms, mask, time, anchor=None):
        j = int(np.abs(self.times-time).argmin())
        if abs(self.times[j]-time) > 2e-6:
            raise ValueError('No exact conditional time')
        if atoms.ndim == 3:
            atoms = atoms.detach().argmax(-1)
        counts = torch.stack([((atoms == self.reference['atom_vocabulary'][a]) & mask.bool()).sum(1) for a in ('N', 'O', 'S')], 1)
        raw, informative, core = self.observables(x.double(), atoms, mask, anchor.double() if anchor is not None else None)
        z = raw/raw.new_tensor(self.scale)
        value, nearest, gate = z.sum(1)*0, z.sum(1)*0, z.sum(1)*0
        available = torch.zeros(len(z), dtype=torch.bool, device=z.device)
        component_ess = torch.zeros_like(value)
        for combo in self.reference['conditional_frames'][j]['combinations']:
            match = (counts == counts.new_tensor(combo['counts'])).all(1) & informative
            if not bool(match.any()):
                continue
            modes = combo['modes']; v = z[match]
            center = v.new_tensor([m['center_scaled'] for m in modes])
            precision = v.new_tensor(np.linalg.inv([m['covariance_dimensionless'] for m in modes]))
            residual = v[:, None]-center[None]
            q = torch.einsum('bmi,mij,bmj->bm', residual, precision, residual).clamp_min(0)/v.shape[1]
            cost = self.delta**2*(torch.sqrt(1+q/self.delta**2)-1)
            logits = -cost/self.tau+v.new_tensor(combo['mode_prior']).log()
            value[match] = self.tau*torch.logsumexp(logits, 1)
            nearest[match] = q.amin(1).sqrt()
            gate[match] = torch.sqrt(q.amin(1)/(1+q.amin(1)))
            component_ess[match] = 1/logits.softmax(1).square().sum(1)
            available |= match
        return value, {'observables': raw, 'available': available, 'core_mask': core,
            'nearest_standardized_rms': nearest, 'dose_gate': gate,
            'conditional_component_ESS': component_ess, 'conditioning_counts': counts}
