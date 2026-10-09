"""Retrospective tail anatomy with clone-aware contrasts and exact genealogy.

No generation, fitted affinity predictor, atom correspondence between molecules,
or reward edits occur here. Extinct lineages are censored, not failed binders.
Coordinates are measured in the receptor frame; graph descriptors are secondary.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from scipy.stats import rankdata, spearmanr, wilcoxon
from scipy.special import softmax

from ..geometry import build_catalog, measure
from ..io import digest, read_json, write_json, write_table
from ..generation.terminal_evaluation import load_mols
from .elite_path_credit import terminal_ancestors
from .path_graph import validate_lineage


def shape_features(xyz, mask):
    """Unweighted heavy-point moments, also valid for predicted noisy endpoints."""
    xyz, mask = np.asarray(xyz, float), np.asarray(mask, bool)
    if xyz.shape[:-1] != mask.shape or xyz.shape[-1] != 3 or not mask.any(axis=1).all():
        raise ValueError("Finite nonempty M,N,3 point clouds and M,N masks required")
    if not np.isfinite(xyz[mask]).all():
        raise ValueError("Nonfinite active coordinates")
    n = mask.sum(1)
    center = np.where(mask[..., None], xyz, 0).sum(1) / n[:, None]
    z = np.where(mask[..., None], xyz-center[:, None], 0)
    covariance = np.einsum('mni,mnj->mij', z, z)/n[:, None, None]
    eigen = np.maximum(np.linalg.eigvalsh(covariance), 0)
    total = eigen.sum(1)
    return {
        'shape::radius_gyration_A': np.sqrt(total),
        'shape::thickness_A': np.sqrt(eigen[:, 0]),
        'shape::planarity_fraction': np.divide(eigen[:, 0], total, out=np.zeros_like(total), where=total > 0),
        'shape::largest_axis_fraction': np.divide(eigen[:, 2], total, out=np.zeros_like(total), where=total > 0),
    }


def bh_adjust(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Finite probabilities required")
    order = np.argsort(p, kind='stable')
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1])
    return q


def graph_weighted_mean(frame, features):
    """Equal graph weight within each batch/group; average repeated poses first."""
    return frame.groupby('smiles', sort=True)[features].mean().mean()


def matched_contrasts(frame, features, seed=42, bootstrap=5000):
    """Paired batch effects; neither clones nor atom pairs are replicates."""
    effects = []
    for batch, f in frame.groupby('batch', sort=True):
        positive, other = f[f.elite], f[~f.elite]
        if positive.empty or other.empty:
            continue
        d = graph_weighted_mean(positive, features)-graph_weighted_mean(other, features)
        for name, value in d.items():
            effects.append({'batch': batch, 'feature': name, 'effect': value,
                            'elite_graphs': positive.smiles.nunique(), 'other_graphs': other.smiles.nunique()})
    effects = pd.DataFrame(effects)
    rows = []
    rng = np.random.default_rng(seed)
    for name in sorted(features):
        x = effects.loc[effects.feature == name, 'effect'].dropna().to_numpy()
        if len(x) < 4:
            continue
        means = x[rng.integers(0, len(x), size=(bootstrap, len(x)))].mean(1)
        p = 1. if not np.any(x) else float(wilcoxon(x, method='auto').pvalue)
        rows.append({'feature': name, 'n_batches': len(x), 'mean_effect': x.mean(),
                     'median_effect': np.median(x), 'ci_low': np.quantile(means, .025),
                     'ci_high': np.quantile(means, .975), 'p': p,
                     'positive_batches': int((x > 0).sum()), 'negative_batches': int((x < 0).sum()),
                     'family': 'graph' if name.startswith('graph::') else 'coordinates'})
    result = pd.DataFrame(rows)
    result['q'] = np.nan
    for _, ids in result.groupby('family', sort=True).groups.items():
        result.loc[ids, 'q'] = bh_adjust(result.loc[ids, 'p'])
    return effects, result.sort_values(['family', 'q', 'feature']).reset_index(drop=True)


def terminal_features(root, campaign, metrics, catalog):
    frames, provenance = [], []
    max_atoms = int(metrics.heavy_atoms.max())
    acid = Chem.MolFromSmarts('[C](=O)[O;H1,-1]')
    amide = Chem.MolFromSmarts('[C](=O)[N]')
    for batch, frame in metrics[metrics.valid_connected].groupby('batch', sort=True):
        folder = root/'results'/campaign/'single'/f'batch_{batch:03d}'
        mols = load_mols(folder)
        provenance.append({'batch': int(batch), 'sdf': str(folder/'molecules_raw_decodable.sdf'),
                           'sha256': digest(folder/'molecules_raw_decodable.sdf')})
        xyz = np.zeros((len(frame), max_atoms, 3))
        labels = np.zeros((len(frame), max_atoms), dtype=int)
        mask = np.zeros_like(labels, dtype=bool)
        descriptors = []
        for i, row in enumerate(frame.itertuples(index=False)):
            mol = Chem.Mol(mols[row.node_id])
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            if Chem.MolToSmiles(mol, isomericSmiles=True) != row.smiles:
                raise ValueError('Terminal SDF/score graph mismatch: '+row.node_id)
            n = mol.GetNumAtoms()
            xyz[i, :n] = mol.GetConformer().GetPositions()
            mask[i, :n] = True
            labels[i, :n] = [catalog['atom_vocabulary'][a.GetSymbol()] for a in mol.GetAtoms()]
            descriptors.append({
                'graph::aromatic_atom_fraction': sum(a.GetIsAromatic() for a in mol.GetAtoms())/n,
                'graph::aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'graph::rings': rdMolDescriptors.CalcNumRings(mol),
                'graph::rotatable_bonds': Lipinski.NumRotatableBonds(mol),
                'graph::HBD': Lipinski.NumHDonors(mol), 'graph::HBA': Lipinski.NumHAcceptors(mol),
                'graph::TPSA_A2': rdMolDescriptors.CalcTPSA(mol), 'graph::logP': Descriptors.MolLogP(mol),
                'graph::carboxyl': int(mol.HasSubstructMatch(acid)),
                'graph::amide': int(mol.HasSubstructMatch(amide)),
            })
        all_geometry = measure(xyz, labels, mask, catalog)
        # The regional inference family contains only all-heavy coordinate features.
        geometry = {k: v for k, v in all_geometry.items() if k.endswith(('::distance_softmin', '::contact_fraction'))}
        geometry.update(shape_features(xyz, mask))
        frames.append(pd.concat([frame.reset_index(drop=True), pd.DataFrame(geometry), pd.DataFrame(descriptors)], axis=1))
    return pd.concat(frames, ignore_index=True), provenance


def duplicate_tail_poses(root, campaign, terminal):
    """Exact graph isomorphisms, receptor-fixed RMS; no rigid superposition."""
    rows = []
    for (batch, smiles), f in terminal[terminal.elite].groupby(['batch', 'smiles'], sort=True):
        if len(f) < 2:
            continue
        high = f.loc[f.pic50_on_rescore.idxmax()]
        low = f.loc[f.pic50_on_rescore.idxmin()]
        mols = load_mols(root/'results'/campaign/'single'/f'batch_{batch:03d}')
        a, b = Chem.Mol(mols[high.node_id]), Chem.Mol(mols[low.node_id])
        Chem.SanitizeMol(a)
        Chem.SanitizeMol(b)
        a, b = Chem.RemoveHs(a), Chem.RemoveHs(b)
        maps = b.GetSubstructMatches(a, useChirality=True, maxMatches=10000)
        if not maps:
            raise ValueError('Canonical graph identity lacks an exact atom mapping')
        x, y = a.GetConformer().GetPositions(), b.GetConformer().GetPositions()
        rms = min(float(np.sqrt(np.mean(np.sum((x-y[list(mapping)])**2, axis=1)))) for mapping in maps)
        rows.append({'batch': batch, 'smiles': smiles, 'n_tail_poses': len(f),
                     'high_slot': int(high.slot), 'low_slot': int(low.slot),
                     'high_pic50': high.pic50_on_rescore, 'low_pic50': low.pic50_on_rescore,
                     'score_difference': high.pic50_on_rescore-low.pic50_on_rescore,
                     'receptor_fixed_isomorphic_rms_A': rms,
                     'high_strain': high.mmff_relief_per_heavy, 'low_strain': low.mmff_relief_per_heavy,
                     'n_isomorphisms': len(maps)})
    return pd.DataFrame(rows)


def quality_panel(named_metrics, threshold, native_name='native', quantile=.9):
    baseline = named_metrics[native_name]
    ok = baseline.valid_connected & baseline.energy_status.eq('converged')
    budget = float(baseline.loc[ok, 'mmff_relief_per_heavy'].quantile(quantile))
    rows = []
    for name, m in named_metrics.items():
        valid = m[m.valid_connected]
        elite = valid[valid.pic50_on_rescore >= threshold]
        qualified = elite[elite.energy_status.eq('converged') & elite.pb_fast_pass & (elite.mmff_relief_per_heavy <= budget)]
        rows.append({'arm': name, 'n': len(m), 'valid': len(valid), 'elite': len(elite),
                     'elite_graphs': elite.smiles.nunique(), 'qualified_elite': len(qualified),
                     'qualified_graphs': qualified.smiles.nunique(),
                     'valid_max': valid.pic50_on_rescore.max(), 'qualified_max': qualified.pic50_on_rescore.max(),
                     'elite_strain_median': elite.mmff_relief_per_heavy.median(),
                     'strain_budget': budget, 'budget_native_quantile': quantile})
    return pd.DataFrame(rows)


def comparison_metrics(spec):
    """PATH#ARM is explicit when a table contains multiple generation arms."""
    path, separator, arm = str(spec).partition('#')
    frame = pd.read_csv(path)
    if separator:
        frame = frame[frame.arm == arm].copy()
        if frame.empty:
            raise ValueError('Unknown comparison arm: '+arm)
    elif frame.arm.nunique() != 1:
        raise ValueError('Multi-arm comparison tables require PATH#ARM')
    if frame.valid_connected.dtype != bool or frame.pb_fast_pass.dtype != bool or frame[['batch', 'slot']].duplicated().any():
        raise ValueError('Comparison requires typed booleans and unique batch/slot identities')
    return frame, Path(path), arm if separator else str(frame.arm.iloc[0])


def stage_trends(group_table):
    """Continuous event curves; no arbitrary time bins, no causal fitness fit."""
    names = ['online_score', 'score_rank_fraction']+[k for k in group_table if k.startswith('shape::')]
    rows, fits = [], []
    for (group, step), frame in group_table.groupby(['group', 'step'], sort=True):
        for name in names:
            x = frame[name].dropna().to_numpy()
            rows.append({'group': group, 'step': step, 'score_time': frame.score_time.iloc[0],
                         'feature': name, 'n_batches': len(x), 'median_nodes_per_batch': frame.n_nodes.median(),
                         'mean': np.mean(x), 'median': np.median(x)})
    curves = pd.DataFrame(rows)
    for (group, feature), frame in curves.groupby(['group', 'feature'], sort=True):
        frame = frame[frame.n_batches >= 4]
        if len(frame) < 6:
            continue
        x, y = frame.score_time.to_numpy(), frame['mean'].to_numpy()
        coefficients = np.polyfit(x, y, 2)
        pred = np.polyval(coefficients, x)
        var = np.sum((y-y.mean())**2)
        fits.append({'group': group, 'feature': feature, 'polynomial_degree': 2,
                     'coefficients_descending': coefficients.tolist(), 'start': x.min(), 'end': x.max(),
                     'r_squared': 1-np.sum((pred-y)**2)/var if var else None,
                     'derivative_coefficients_descending': np.polyder(coefficients).tolist(),
                     'scope': 'Descriptive fit to batch-averaged endpoint/score curves; not an affinity reward or causal effect; shared ancestors and repeated stages are correlated'})
    return curves, fits


def analyze_genealogy(root, campaign, metrics, catalog):
    """Read one batch at a time; persist only traces and small per-event summaries."""
    config = read_json(root/'results'/campaign/'config.json')
    scale = float(config['coord_scale'])
    paths, populations, groups, batches, audits, correlations, siblings = [], [], [], [], [], [], []
    for batch, m in metrics.groupby('batch', sort=True):
        m = m.sort_values('slot').reset_index(drop=True)
        path = root/'results'/campaign/'single'/f'batch_{batch:03d}'/'trajectory.npz'
        with np.load(path) as t:
            clock, state, selected, roots, offspring, resampled = validate_lineage(t)
            n = selected.shape[1]
            if not np.array_equal(m.slot.to_numpy(), np.arange(n)):
                raise ValueError('Terminal records must cover every original slot')
            ancestors = terminal_ancestors(selected)
            if not np.array_equal(roots[0, ancestors[0]], m.root_slot):
                raise ValueError('Final root mismatch')
            active = np.flatnonzero(resampled)
            stop = int(active[-1])
            copies = t['proposal_coords'][active[:, None], selected[active]]
            if not np.array_equal(copies, t['current_coords'][active+1]):
                raise ValueError('Copy edge mismatch')
            probability = np.asarray(t['selection_probability'], float)
            softmax_error = np.max(np.abs(softmax(t['pic50_on'][active], axis=1)-probability[active]))
            if softmax_error > 2e-6:
                raise ValueError('This analyzer requires single-target softmax scores')
            com = np.asarray(read_json(root/'results'/campaign/f'frame_batch_{batch:03d}.json')['target_com'])
            # A batch can only be mapped via its original center if all pockets share it.
            if not np.allclose(com, com[0], rtol=0, atol=1e-6):
                raise ValueError('Reordered variable receptor centers require explicit trace mapping')
            valid, elite = m.valid_connected.to_numpy(bool), m.elite.to_numpy(bool)
            elite_slots = np.flatnonzero(elite)
            for parent in np.unique(ancestors[stop, elite]):
                children = np.flatnonzero(ancestors[stop] == parent)
                good = children[valid[children]]
                e = good[elite[good]]
                other = good[~elite[good]]
                high = int(e[np.argmax(m.pic50_on_rescore.to_numpy()[e])])
                low = int(good[np.argmin(m.pic50_on_rescore.to_numpy()[good])])
                state_fields = ['current_coords', 'current_atomics', 'current_bonds', 'current_charges', 'mask']
                same = all(np.array_equal(t[k][stop+1, children],
                           np.broadcast_to(t[k][stop+1, children[0]], t[k][stop+1, children].shape)) for k in state_fields)
                if not same:
                    raise ValueError('Exit siblings do not share the recorded coordinate/chemical state')
                siblings.append({'batch': batch, 'exit_parent_slot': int(parent), 'score_time': clock[stop],
                    'exit_state_time': state[stop], 'n_children': len(children), 'n_valid': len(good),
                    'n_elite': len(e), 'n_other_valid': len(other), 'n_unique_graphs': m.iloc[good].smiles.nunique(),
                    'high_child_slot': high, 'low_child_slot': low, 'high_final_pic50': m.pic50_on_rescore.iloc[high],
                    'low_final_pic50': m.pic50_on_rescore.iloc[low],
                    'score_range': m.pic50_on_rescore.iloc[high]-m.pic50_on_rescore.iloc[low],
                    'same_recorded_exit_state': same, 'checked_fields': ','.join(state_fields)})
            step_ids = sorted(set(active.tolist()+[stop+1, len(clock)*3//4, len(clock)-1]))
            for step in step_ids:
                s = np.asarray(t['pic50_on'][step], float)
                rank = rankdata(s, method='average')/n
                endpoint = t['predicted_coords'][step]*scale+com[:, None]
                shape = shape_features(endpoint, t['mask'][step])
                all_ids = np.unique(ancestors[step])
                positive = np.unique(ancestors[step, elite])
                other = np.setdiff1d(np.unique(ancestors[step, valid & ~elite]), positive)
                censored = np.setdiff1d(np.arange(n), all_ids)
                if resampled[step]:
                    populations.append({'batch': batch, 'step': step, 'score_time': clock[step],
                        'state_time': state[step], 'ess': 1/np.sum(probability[step]**2),
                        'current_roots': len(np.unique(roots[step])), 'terminal_ancestors': len(all_ids),
                        'elite_ancestors': len(positive), 'other_valid_only_ancestors': len(other),
                        'shared_elite_and_other_ancestors': len(np.intersect1d(positive, np.unique(ancestors[step, valid & ~elite]))),
                        'eliminated_now': int((offspring[step] == 0).sum()),
                        'event_max_probability': probability[step].max()})
                    for group, ids in [('elite_descendant', positive), ('other_observed_valid', other), ('extinct_censored', censored)]:
                        if not len(ids):
                            continue
                        row = {'batch': batch, 'step': step, 'score_time': clock[step], 'group': group,
                               'n_nodes': len(ids), 'online_score': s[ids].mean(),
                               'score_rank_fraction': rank[ids].mean(), 'selection_probability': probability[step, ids].mean(),
                               'offspring': offspring[step, ids].mean()}
                        row.update({k: np.mean(v[ids]) for k, v in shape.items()})
                        groups.append(row)
                for slot in elite_slots:
                    ancestor = int(ancestors[step, slot])
                    row = {'batch': batch, 'final_slot': int(slot), 'root_slot': int(m.root_slot.iloc[slot]),
                           'node_id': m.node_id.iloc[slot], 'final_pic50': float(m.pic50_on_rescore.iloc[slot]),
                           'strain': float(m.mmff_relief_per_heavy.iloc[slot]), 'step': step,
                           'ancestor_slot': ancestor, 'score_time': clock[step], 'proposal_state_time': state[step],
                           'actual_selection_event': bool(resampled[step]), 'online_pic50': s[ancestor],
                           'score_rank_fraction': rank[ancestor], 'selection_probability': probability[step, ancestor],
                           'offspring': int(offspring[step, ancestor])}
                    row.update({k: float(v[ancestor]) for k, v in shape.items()})
                    paths.append(row)
                if step in [stop, stop+1, len(clock)*3//4, len(clock)-1] and valid.sum() > 2:
                    x, y = s[ancestors[step, valid]], m.pic50_on_rescore.to_numpy()[valid]
                    correlations.append({'batch': batch, 'step': step, 'score_time': clock[step],
                                         'spearman': float(spearmanr(x, y).statistic), 'n': int(valid.sum())})
            e = m[m.elite]
            initial = np.unique(ancestors[0, elite])
            initial_rank = rankdata(t['pic50_on'][0], method='average')/n
            batches.append({'batch': batch, 'valid': int(valid.sum()), 'elite': len(e), 'elite_graphs': e.smiles.nunique(),
                            'terminal_roots': len(np.unique(ancestors[0])), 'elite_roots': len(initial),
                            'elite_initial_rank': float(initial_rank[initial].mean()) if len(initial) else np.nan,
                            'distinct_selection_exit_parents': len(np.unique(ancestors[stop])),
                            'elite_selection_exit_parents': len(np.unique(ancestors[stop, elite])),
                            'final_max': m.loc[valid, 'pic50_on_rescore'].max(),
                            'ess_mean': float((1/np.sum(probability[active]**2, axis=1)).mean())})
            audits.append({'batch': int(batch), 'trajectory_sha256': digest(path), 'steps': len(clock),
                           'selection_events': len(active), 'selection_score_start': float(clock[active[0]]),
                           'selection_score_end': float(clock[stop]), 'selection_exit_state': float(state[stop]),
                           'maximum_softmax_error': float(softmax_error), 'copy_and_root_identity_verified': True})
    return {name: pd.DataFrame(rows) for name, rows in [('elite_paths', paths), ('selection_population', populations),
            ('selection_groups', groups), ('batch_tail', batches), ('suffix_correlations', correlations),
            ('exit_sibling_fates', siblings)]}, audits


def run_tail_analysis(root, campaign, metrics_path, config_path, output, threshold, comparisons):
    root, output = Path(root), Path(output)
    if not np.isfinite(threshold):
        raise ValueError('An explicit finite frozen tail threshold is required')
    m = pd.read_csv(metrics_path)
    if m.valid_connected.dtype != bool or m.pb_fast_pass.dtype != bool or m[['batch', 'slot']].duplicated().any():
        raise ValueError('Typed booleans and unique batch/slot identities required')
    if not np.isfinite(m.pic50_on_rescore).all():
        raise ValueError('Missing target scores')
    m['elite'] = m.valid_connected & (m.pic50_on_rescore >= threshold)
    config = read_json(config_path)
    config['feature_pockets'] = ['ck2']
    original = read_json(root/'results'/campaign/'config.json')
    catalog = build_catalog(root, config, original['atom_vocabulary'])
    terminal, sdf_audit = terminal_features(root, campaign, m, catalog)
    feature_names = [k for k in terminal if k.startswith(('shape::', 'graph::')) or k.endswith(('::distance_softmin', '::contact_fraction'))]
    effects, contrast = matched_contrasts(terminal, feature_names)
    tables, audit = analyze_genealogy(root, campaign, m, catalog)
    tables.update({'terminal_contrasts': contrast, 'batch_feature_effects': effects,
                   'elite_terminal_features': terminal[terminal.elite].sort_values('pic50_on_rescore', ascending=False),
                   'same_graph_tail_poses': duplicate_tail_poses(root, campaign, terminal)})
    loaded = {name: comparison_metrics(spec) for name, spec in comparisons.items()}
    named = {name: parts[0] for name, parts in loaded.items()}
    named['steer'] = m
    if named.keys() >= {'native', 'steer'}:
        tables['quality_tail_panel'] = quality_panel(named, threshold)
        budget = float(tables['quality_tail_panel'].strain_budget.iloc[0])
        qualified = terminal[terminal.energy_status.eq('converged') & terminal.pb_fast_pass & (terminal.mmff_relief_per_heavy <= budget)]
        qe, qc = matched_contrasts(qualified, feature_names)
        tables.update({'qualified_batch_feature_effects': qe, 'qualified_terminal_contrasts': qc})
    tables['selection_trend_curves'], fits = stage_trends(tables['selection_groups'])
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        write_table(output/(name+'.csv'), table)
    write_json(output/'descriptive_trend_fits.json', fits)
    # These are compact descriptive quantiles, not independent-sample tests.
    describe = {}
    for label, group in [('elite', terminal[terminal.elite]), ('other_valid', terminal[~terminal.elite])]:
        describe[label] = {'n': len(group), 'graphs': group.smiles.nunique(),
                           'feature_medians': group[feature_names].median().to_dict(),
                           'feature_q25': group[feature_names].quantile(.25).to_dict(),
                           'feature_q75': group[feature_names].quantile(.75).to_dict()}
    write_json(output/'summary.json', {'schema': 'steer-tail-anatomy-1.0', 'threshold': threshold,
        'campaign': campaign, 'terminal_summary': describe, 'n_features': len(feature_names),
        'code_sha256': digest(__file__),
        'coordinate_parameters': {k: config[k] for k in ['pocket_radius_A', 'softmin_temperature_A', 'contact_midpoint_A', 'contact_width_A']},
        'regions': list(catalog['regions']), 'batch_audit': audit, 'sdf_audit': sdf_audit,
        'inputs': {str(p): digest(p) for p in [Path(metrics_path), Path(config_path), root/'inputs/3PE1_protein_aligned.pdb',
                                             root/'inputs/3PE1_ligand_aligned.sdf', root/'results'/campaign/'config.json']},
        'comparison_inputs': {name: {'path': str(parts[1]), 'sha256': digest(parts[1]), 'arm': parts[2]} for name, parts in loaded.items()},
        'semantics': {
            'affinity': 'FLOWR head prediction, not independently measured binding affinity',
            'endpoint': 'Model endpoint prediction; not current noisy state or decoded final molecule',
            'contrasts': 'Matched batch mean differences with equal graph weight within each label; paired Wilcoxon, BH by coordinate/graph family; deterministic batch bootstrap seed 42',
            'selection_labels': 'Retrospective observed elite descendants, other observed valid descendants, and future-censored extinct nodes; shared ancestors allocated to elite class',
            'suffix': 'Only three post-selection score diagnostics, excluded from selection feature mining',
            'contacts': 'All-heavy smooth distances/contacts, not verified hydrogen bonds or interaction energies',
            'strain': 'Converged MMFF94s local relaxation relief per heavy atom, not binding free energy',
            'quality_budget': 'Original native P90 strain, fixed before examining Steer tail qualification',
            'causality': 'Retrospective associations, no network attention or causal module evidence',
        }})
    return tables
