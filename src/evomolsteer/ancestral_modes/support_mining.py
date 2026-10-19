"""Deterministic variable-size local supports, measured evidence and compilation.

No affinity inference, reward fitting, LLM call or generation occurs here. Final
group contrasts and a teacher's full-window evolution are separate evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.local_support import bounds, validate_slots


FEATURES = ('point_error_A', 'pair_error_A', 'pocket_error_A',
            'signed_dx_A', 'signed_dy_A', 'signed_dz_A', 'beyond_global_error_A')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(config):
    for key in ('bootstrap_repeats', 'max_regions_per_template', 'trend_degree'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(key + ' must be a positive integer')
    window = config['learning_window']
    if len(window) != 2 or not 0 <= window[0] < window[1] < 1:
        raise ValueError('A declared learning window within [0,1) is required')
    p = config['local_support_policy']
    if isinstance(p['pocket_sites'], bool) or not isinstance(p['pocket_sites'], int) or p['pocket_sites'] < 1:
        raise ValueError('pocket_sites must be a positive integer')
    if not np.isfinite(p['band_bond_fraction']) or p['band_bond_fraction'] <= 0:
        raise ValueError('band_bond_fraction must be positive')


def propose(mode, pocket, policy):
    """Graph neighborhoods and spatial supports; cardinality follows the data."""
    xyz = np.asarray(mode['terminal_coords_A'], dtype=float)
    low, high = bounds(len(xyz), policy)
    adjacency = {int(i): sorted(set(j)) for i, j in mode['full_adjacency'].items()}
    if set(adjacency) != set(range(len(xyz))) or any(
            j not in adjacency for neighbors in adjacency.values() for j in neighbors):
        raise ValueError('Teacher graph does not cover its atom coordinates')
    sites = pocket[['x_A', 'y_A', 'z_A']].to_numpy(dtype=float)
    if not len(sites) or not np.isfinite(xyz).all() or not np.isfinite(sites).all():
        raise ValueError('Finite teacher and pocket coordinates required')
    nearest = np.linalg.norm(xyz[:, None] - sites[None], axis=-1).min(1)
    lengths = [np.linalg.norm(xyz[i] - xyz[j]) for i in adjacency for j in adjacency[i] if i < j]
    lengths = [v for v in lengths if v > 0]
    if not lengths:
        raise ValueError('Cannot calibrate neighborhoods without nonzero graph bond lengths')
    proposals = {}

    def add(slots, reason):
        slots = sorted(set(int(v) for v in slots))
        if not low <= len(slots) <= high:
            return
        proposals.setdefault(tuple(slots), []).append(reason)

    depths = policy['graph_depths']
    if any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in depths):
        raise ValueError('graph_depths must contain positive integers')
    for center in range(len(xyz)):
        for depth in depths:
            slots = {center}
            for _ in range(depth):
                slots |= {j for i in list(slots) for j in adjacency[i]}
            add(slots, {'kind': 'bond_neighborhood', 'center': center, 'depth': depth})
        for multiple in policy['spatial_radius_bond_multiples']:
            if not np.isfinite(multiple) or multiple <= 0:
                raise ValueError('Positive neighborhood radius required')
            radius = float(np.median(lengths) * multiple)
            add(np.flatnonzero(np.linalg.norm(xyz - xyz[center], axis=1) <= radius),
                {'kind': 'spatial_neighborhood', 'center': center, 'radius_A': radius})
    for fraction in policy['interface_fractions']:
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Interface coverage fractions must be in (0,1)')
        size = max(low, int(np.floor(len(xyz) * fraction)))
        add(np.argsort(nearest, kind='stable')[:size],
            {'kind': 'interface_geometry', 'requested_fraction': fraction})
    result = []
    for slots, reasons in sorted(proposals.items()):
        indices = list(slots)
        site_distance = np.linalg.norm(xyz[indices, None] - sites[None], axis=-1).min(0)
        site_ids = np.argsort(site_distance, kind='stable')[:policy['pocket_sites']]
        pocket_sites = []
        for j in site_ids:
            row = pocket.iloc[j]
            pocket_sites.append({'coords_A': sites[j].tolist(), 'id': str(row.residue) + ':' + str(row.atom_name),
                                 'index': int(row.pocket_index)})
        identity = fingerprint([mode['template_id'], indices])[:12]
        roles = [r for r in mode.get('terminal_roles', []) if r['atom_index'] in indices]
        result.append({'region_id': mode['candidate_id'] + '__local_' + identity,
            'template_id': mode['template_id'], 'candidate_id': mode['candidate_id'],
            'atom_slots': indices, 'family': 'mined_local_support', 'proposal_reasons': reasons,
            'locality': validate_slots(indices, len(xyz), policy), 'source_score': mode['terminal_score'],
            'pocket_sites': pocket_sites, 'orientation_triplets': [
                [i, *adjacency[i][:2]] for i in indices
                if len(adjacency[i]) >= 2 and set(adjacency[i][:2]).issubset(slots)],
            'terminal_roles': roles, 'claim': 'Measured local coordinate contrast candidate; causal affinity benefit unproven',
            'base_band_A': float(np.median(lengths) * policy['band_bond_fraction']),
            'scale_source': 'Configured fraction of observed covalent-length median; not an energy'})
    return result


def descriptors(cloud, target, sites, global_error):
    if len(cloud) < len(target):
        raise ValueError('Insufficient points for a unique local correspondence')
    rr, cc = linear_sum_assignment(np.square(cloud[:, None] - target).sum(-1))
    points = cloud[rr[np.argsort(cc)]]
    delta = points - target
    error = np.sqrt(np.square(delta).sum(-1).mean())
    a, b = np.triu_indices(len(target), 1)
    pair = np.linalg.norm(points[a] - points[b], axis=-1) - np.linalg.norm(target[a] - target[b], axis=-1)
    pocket = np.linalg.norm(points[:, None] - sites[None], axis=-1) - np.linalg.norm(target[:, None] - sites[None], axis=-1)
    return np.r_[error, np.sqrt(np.square(pair).mean()), np.sqrt(np.square(pocket).mean()),
                 delta.mean(0), error - global_error]


def grouped_contrast(values, labels, roots, bootstrap_repeats, seed, batches=None):
    """Equal-root means; resample whole source batches when they are available."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    roots = np.asarray(roots)
    units = sorted(set(roots.tolist()))
    if not len(units):
        raise ValueError('At least one observed control root required')
    clusters = roots if batches is None else np.asarray(batches)
    root_clusters = []
    aggregates = np.full((len(units), 2, values.shape[1]), np.nan)
    for i, root in enumerate(units):
        cluster = set(clusters[roots == root].tolist())
        if len(cluster) != 1:
            raise ValueError('A root cannot belong to different source batches')
        root_clusters.append(next(iter(cluster)))
        for label in (False, True):
            ids = (roots == root) & (labels == label)
            if ids.any():
                aggregates[i, int(label)] = values[ids].mean(0)

    def effect(a):
        if not np.isfinite(a[:, 0]).any() or not np.isfinite(a[:, 1]).any():
            return np.full(values.shape[1], np.nan)
        return np.nanmean(a[:, 0], axis=0) - np.nanmean(a[:, 1], axis=0)

    estimate = effect(aggregates)
    rng = np.random.default_rng(seed)
    cluster_ids = sorted(set(root_clusters))
    cluster_rows = [np.flatnonzero(np.asarray(root_clusters) == c) for c in cluster_ids]
    high_clusters = {root_clusters[i] for i in range(len(units)) if np.isfinite(aggregates[i, 1, 0])}
    other_clusters = {root_clusters[i] for i in range(len(units)) if np.isfinite(aggregates[i, 0, 0])}
    samples = []
    for _ in range(bootstrap_repeats):
        ids = np.concatenate([cluster_rows[i] for i in rng.integers(0, len(cluster_ids), len(cluster_ids))])
        z = effect(aggregates[ids])
        if np.isfinite(z).all():
            samples.append(z)
    rows = {}
    for i, feature in enumerate(FEATURES):
        interval = np.quantile(np.asarray(samples)[:, i], [.025, .975]).tolist() if samples and min(len(high_clusters), len(other_clusters)) >= 2 else None
        rows[feature] = {'other_minus_high': float(estimate[i]) if np.isfinite(estimate[i]) else None,
                         'cluster_interval_95': interval}
    return {'features': rows, 'roots': len(units), 'high_roots': int(np.isfinite(aggregates[:, 1, 0]).sum()),
            'other_roots': int(np.isfinite(aggregates[:, 0, 0]).sum()), 'n_high': int(labels.sum()),
            'n_other': int((~labels).sum()), 'valid_bootstrap_repeats': len(samples),
            'bootstrap_unit': 'source_batch' if batches is not None else 'root', 'clusters': len(cluster_ids),
            'high_clusters': len(high_clusters), 'other_clusters': len(other_clusters),
            'positive_effect_meaning': 'For error features, positive means high-quality finals are closer; signed axes have no universal preferred sign',
            'limits': 'Root-balanced descriptive association in observed valid finals. Intervals are unadjusted exploratory intervals, not causal effects, independent targets or calibrated success probability'}


def teacher_trends(mode, region, learning_window, degree):
    clocks = np.asarray(mode['path']['times'], dtype=float)
    ids = np.flatnonzero((clocks >= learning_window[0] - 2e-6) & (clocks <= learning_window[1] + 2e-6))
    if len(ids) < degree + 1 or np.any(np.diff(clocks[ids]) <= 0):
        raise ValueError('Insufficient distinct teacher times for full-window fitting')
    if clocks[ids[0]] > learning_window[0] + 2e-6 or clocks[ids[-1]] < learning_window[1] - 2e-6:
        raise ValueError('Teacher records do not cover the complete requested learning window')
    slots = region['atom_slots']
    final = np.asarray(mode['terminal_coords_A'])[slots]
    result = {}
    for representation, key in [('actual_current', 'coords_A'), ('endpoint_prediction', 'endpoint_A')]:
        x = np.asarray(mode['path'][key])[ids][:, slots]
        delta = x - final
        values = np.column_stack([np.sqrt(np.square(delta).sum(-1).mean(1)), delta.mean(1)])
        coefficients = np.polynomial.polynomial.polyfit(clocks[ids], values, degree)
        fit = np.polynomial.polynomial.polyval(clocks[ids], coefficients).T
        result[representation] = {'observables': ['error_to_actual_final_A', 'signed_dx_A', 'signed_dy_A', 'signed_dz_A'],
            'coefficients_ascending': coefficients.tolist(),
            'time_derivative_coefficients': (coefficients[1:] * np.arange(1, degree + 1)[:, None]).tolist(),
            'fit_rmse': np.sqrt(np.square(fit - values).mean(0)).tolist(),
            'mean_speed_A_per_t': float(np.linalg.norm(np.diff(x, axis=0), axis=-1).mean(1).sum() / np.diff(clocks[ids]).sum())}
    return {'domain': [float(clocks[ids[0]]), float(clocks[ids[-1]])], 'n_times': len(ids), 'fits': result,
            'limits': 'Evolution of one recorded teacher lineage; not cohort enrichment or a spatial gradient. Fits are diagnostic, not substituted for online teacher predictions.'}


def prepare(bank_path, atoms_path, membership_path, pocket_path, config_path, output, templates=None):
    paths = dict(bank=Path(bank_path), atoms=Path(atoms_path), membership=Path(membership_path),
                 pocket=Path(pocket_path), config=Path(config_path))
    config = read_json(paths['config'])
    validate_config(config)
    policy = config['local_support_policy']
    source = {k: {'path': str(v.resolve()), 'sha256': digest(v)} for k, v in paths.items()}
    selected = sorted(templates or read_json(paths['bank'])['modes'])
    import evomolsteer.local_support as support_policy_module
    identity = fingerprint({'source_hashes': {k: v['sha256'] for k, v in source.items()},
                            'templates': selected, 'implementation_sha256': digest(Path(__file__)),
                            'policy_implementation_sha256': digest(Path(support_policy_module.__file__)),
                            'libraries': {'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__}})
    output = Path(output)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old['identity'] != identity:
            raise ValueError('Local support inputs/policy changed; use a new output directory')
        for relative, sha in old['artifacts'].items():
            if digest(output / relative) != sha:
                raise ValueError('Local support evidence artifact changed')
        return old
    bank = read_json(paths['bank'])
    pocket = pd.read_csv(paths['pocket'])
    atoms = pd.read_csv(paths['atoms'])
    membership = pd.read_csv(paths['membership'])
    labels = membership[config['positive_label']]
    if (membership.node_id.duplicated().any() or labels.isna().any()
            or not set(labels.tolist()).issubset({False, True, 0, 1})):
        raise ValueError('Unique membership and explicit final group labels required')
    clouds = {node: g.sort_values('atom_index')[['x_A', 'y_A', 'z_A']].to_numpy() for node, g in atoms.groupby('node_id')}
    regions, evidence_files = [], []
    for tid in selected:
        mode = bank['modes'][tid]
        controls = membership.loc[membership.node_id.ne(mode['evidence_origin'])].copy()
        if config['exclude_teacher_root']:
            controls = controls.loc[controls.root_id.ne(mode['root_id'])]
        if controls.empty or any(node not in clouds for node in controls.node_id):
            raise ValueError('Missing terminal controls / coordinates')
        full = np.asarray(mode['terminal_coords_A'])
        global_errors = []
        for node in controls.node_id:
            cloud = clouds[node]
            if len(cloud) < len(full):
                raise ValueError('Global reference contrast requires at least teacher atom count; stratify atom counts first')
            rr, cc = linear_sum_assignment(np.square(cloud[:, None] - full).sum(-1))
            global_errors.append(np.sqrt(np.square(cloud[rr[np.argsort(cc)]] - full).sum(-1).mean()))
        proposals = propose(mode, pocket, policy)
        ranked = []
        for region in proposals:
            target = full[region['atom_slots']]
            sites = np.asarray([v['coords_A'] for v in region['pocket_sites']])
            values = [descriptors(clouds[node], target, sites, error) for node, error in zip(controls.node_id, global_errors)]
            contrast = grouped_contrast(values, controls[config['positive_label']].to_numpy(), controls.root_id.to_numpy(),
                                        config['bootstrap_repeats'], config['seed'],
                                        controls.batch.to_numpy() if 'batch' in controls else None)
            ev = {'region_id': region['region_id'], 'locality': region['locality'], 'terminal_contrast': contrast,
                  'teacher_trends': teacher_trends(mode, region, config['learning_window'], config['trend_degree']),
                  'claim_status': 'exploratory', 'identity': identity,
                  'observable_to_operator': {'point_error_A': 'pointwise', 'pair_error_A': 'pair_distance', 'pocket_error_A': 'pocket_distance'},
                  'limits': ['Other observed finals are controls, not extinct-branch failure labels.',
                             'Root clustering does not establish cross-target generalization.',
                             'No affinity model call and no per-node feature table is saved.']}
            path = output / 'evidence' / (region['region_id'] + '.json')
            write_json(path, ev)
            evidence_files.append(path.relative_to(output).as_posix())
            region['local_support_evidence'] = str(path.relative_to(output).as_posix())
            effect = contrast['features']['beyond_global_error_A']
            rank = effect['cluster_interval_95'][0] if effect['cluster_interval_95'] is not None else float('-inf')
            region['shortlist_basis'] = {'beyond_global_effect': effect, 'claim': 'Exploratory rank, not proof of affinity benefit'}
            ranked.append((rank, -len(region['atom_slots']), region))
        ranked.sort(key=lambda r: (-r[0], -r[1], r[2]['region_id']))
        # Budget is a count of proposed regions, never the number of atoms per region.
        chosen = {r[2]['region_id'] for r in ranked[:config['max_regions_per_template']]}
        for _, _, region in ranked:
            region['shortlisted'] = region['region_id'] in chosen
            regions.append(region)
    write_json(output / 'regions.json', regions)
    artifact_names = ['regions.json', *evidence_files]
    manifest = {'schema_version': 'local-support-mining.v1', 'identity': identity, 'sources': source,
                'bank_identity': fingerprint(bank),
                'policy': policy, 'config': config, 'templates': selected, 'n_regions': len(regions),
                'shortlisted_regions': [r['region_id'] for r in regions if r['shortlisted']],
                'artifacts': {name: digest(output / name) for name in artifact_names}}
    write_json(manifest_path, manifest)
    return manifest


def load_registry(path, bank):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    for relative, sha in manifest['artifacts'].items():
        if digest(path / relative) != sha:
            raise ValueError('Local support evidence fingerprint mismatch')
    if fingerprint(bank) != manifest['bank_identity']:
        raise ValueError('Local support registry refers to a different teacher bank')
    return manifest, read_json(path / 'regions.json')


def feature_binding(program, region, bank, policy):
    """Recompute provenance whenever an action changes support, clock or force."""
    locality = validate_slots(region['atom_slots'], len(bank['modes'][region['template_id']]['coords_A']), policy)
    return {'region_id': region['region_id'], 'atom_slots': list(region['atom_slots']), 'locality': locality,
            'evidence': region.get('local_support_evidence'), 'operator': program['operator'],
            'origin': 'deterministic_feature_compilation; not a new Analyst/Designer decision',
            'chemical_role_used_for_selection_only': not (program['typed_weight'] or program['charge_weight']),
            'clock_policy': program['clock_policy'], 'direct_type_force': bool(program['typed_weight']),
            'active_terms': ['registered_' + program['operator']]
                + (['typed_spatial_field'] if program['typed_weight'] else [])
                + (['charge_spatial_field'] if program['charge_weight'] else [])}


def compile_program(parent, region, bank, policy, operator):
    """Bind selected support/observable to a real registered coordinate potential."""
    from .programs import spec, validate, executable_view
    mode = bank['modes'][region['template_id']]
    validate_slots(region['atom_slots'], len(mode['coords_A']), policy)
    if operator not in ('pointwise', 'pair_distance', 'pocket_distance', 'orientation'):
        raise ValueError('Unsupported local observable')
    p = copy.deepcopy(parent)
    p.update(mode_specs=[spec(region)], routing='single', operator=operator,
             local_support_policy=copy.deepcopy(policy), typed_weight=0., charge_weight=0.,
             advantage_evidence_gain=False)
    if operator in ('pair_distance', 'pocket_distance'):
        p['target_band_A'] = region['base_band_A']
    for name in ('teacher_sha256', 'analyst_sha256', 'designer_sha256', 'evidence_bundle_sha256'):
        p.pop(name, None)
    p['local_feature_binding'] = feature_binding(p, region, bank, policy)
    validate(p, bank)
    assert p['local_feature_binding']['active_terms'] == executable_view(p)['actual_terms']
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for key in ('bank', 'atoms', 'membership', 'pocket', 'config', 'output'):
        prep.add_argument('--' + key, required=True, type=Path)
    prep.add_argument('--template', action='append')
    comp = sub.add_parser('compile')
    for key in ('bank', 'registry', 'parent', 'output'):
        comp.add_argument('--' + key, required=True, type=Path)
    comp.add_argument('--region', required=True)
    comp.add_argument('--operator', choices=['pointwise', 'pair_distance', 'pocket_distance', 'orientation'], required=True)
    comp.add_argument('--program-id', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.bank, args.atoms, args.membership, args.pocket, args.config, args.output, args.template)
        print(json.dumps({'n_regions': result['n_regions'], 'shortlisted_regions': result['shortlisted_regions'], 'identity': result['identity']}))
    else:
        bank = read_json(args.bank)
        manifest, regions = load_registry(args.registry, bank)
        region = next(r for r in regions if r['region_id'] == args.region)
        program = compile_program(read_json(args.parent), region, bank, manifest['policy'], args.operator)
        program.update(program_id=args.program_id, teacher_sha256=digest(args.bank))
        write_json(args.output, program)
        write_json(args.output.with_suffix('.binding.json'), {'program_sha256': digest(args.output),
            'teacher_path': str(args.bank.resolve()), 'teacher_sha256': digest(args.bank), 'registry_identity': manifest['identity'],
            'binding': program['local_feature_binding'], 'status': 'compiled_not_generated'})
        print(json.dumps({'program': str(args.output), 'binding': program['local_feature_binding']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
