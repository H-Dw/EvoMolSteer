"""Offline descendant credit, with explicit extinction and last-record semantics.

No network call: only original selected-parent indices, recorded scores and
coordinates inside the learned support. A missing future label is never zero.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from scipy.optimize import linear_sum_assignment

from .coordinate_mining import bh
from .endpoint_regions import COMPONENTS, regional_moments
from ..generation.window_reference import load_reference
from ..io import digest, read_json, write_json
from ..storage.trajectory import TrajectoryPackage


def descendant_credit(selected, terminal_scores, terminal_index=None):
    """Map last observed slots backward; exclude selection at the terminal row.

    selected[j, child] maps a slot at node j+1 to its parent at node j.
    Terminal scores are node forecasts, before selection at that same node.
    """
    a = np.asarray(selected)
    score = np.asarray(terminal_scores, float).reshape(-1)
    if a.ndim != 2 or a.shape[1] != len(score) or not np.isfinite(score).all():
        raise ValueError('Finite terminal scores and T,N parent indices required')
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a >= len(score))):
        raise ValueError('Invalid parent indices')
    end = len(a)-1 if terminal_index is None else int(terminal_index)
    if not 0 <= end < len(a):
        raise ValueError('Invalid terminal index')
    counts = np.zeros((end+1, len(score)), dtype=np.int64)
    means = np.full(counts.shape, np.nan)
    ancestor = np.arange(len(score))
    for node in range(end, -1, -1):
        counts[node] = np.bincount(ancestor, minlength=len(score))
        sums = np.bincount(ancestor, weights=score, minlength=len(score))
        np.divide(sums, counts[node], out=means[node], where=counts[node] > 0)
        if node:
            ancestor = a[node-1, ancestor]
    return counts, means


def diverse_survivors(coords, labels, counts, points, origin, limit=10):
    """Equal ancestors, not descendant multiplicity; keep intact coordinate modes."""
    if limit < 1:
        raise ValueError('Positive teacher limit required')
    live = np.flatnonzero(np.asarray(counts) > 0)
    if not len(live) or not np.isfinite(np.asarray(labels)[live]).all():
        raise ValueError('Observable surviving ancestors required')
    x = np.asarray(coords)
    rank = live[np.argsort(-np.asarray(labels)[live], kind='stable')]
    chosen = []
    for slot in rank:
        distances = []
        for j in chosen:
            cost = ((x[slot, :, None]-x[j, None])**2).sum(-1)
            row, col = linear_sum_assignment(cost)
            distances.append(np.sqrt(cost[row, col].mean()))
        # Fixed receptor-frame distance in Angstrom; only exclude near copies.
        if not chosen or min(distances) > 1e-3:
            chosen.append(int(slot))
        if len(chosen) >= limit:
            break
    return chosen


def descriptive_parent_mean(values, parents, keep):
    """Permit one observed ancestor as a description, never as an n=3 test."""
    ids = np.unique(parents[keep])
    if not len(ids):
        return np.full(values.shape[1], np.nan), 0
    return np.mean([values[keep & (parents == p)].mean(0) for p in ids], axis=0), len(ids)


def mine(reference, dataset, campaign, output, radius_A=5., teachers_per_batch=2):
    module_sha = digest(__file__)
    ref = load_reference(reference)
    root = Path(dataset)
    source = root/'results'/campaign
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    cfg = read_json(source/'config.json')
    times = np.asarray(ref['times'], float)
    fields = [f'landmark_{r:02d}_{c}' for r in range(len(ref['landmarks_A'])) for c in COMPONENTS]
    alive_means, extinct_means, scales, batches, rows, teachers = [], [], [], [], [], {}
    terminal_metadata = []
    for item in ref['sources']:
        path = root/item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Immutable original source mismatch')
        batch = int(path.parent.name.split('_')[-1])
        batches.append(batch)
        com = np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:, None]
        am, em, sd = [], [], []
        with TrajectoryPackage(path) as tr:
            grid = np.round(tr.read('score_time')[:, 0].astype(float), 6)
            state = tr.read('state_time')
            selected = tr.read('selected_indices').astype(np.int64)
            end = len(grid)-1
            score = tr.read('pic50_on', end).reshape(-1)
            counts, credit = descendant_credit(selected, score)
            ids = np.flatnonzero(tr.read('resampled') & (grid >= ref['window'][0]-1e-6) & (state <= ref['window'][1]+1e-6))
            if not np.array_equal(grid[ids], times) or np.any(np.diff(ids) != 1):
                raise ValueError('Exact contiguous learned support required')
            terminal_metadata.append(dict(batch=batch, terminal_index=end,
                terminal_score_time=float(grid[end]), terminal_state_time=float(state[end]),
                future_resampling_nodes=int(tr.read('resampled')[ids[-1]+1:end].sum())))
            previous_parents = None
            for k, i in enumerate(ids):
                y = tr.read('predicted_coords', int(i)).astype(float)*cfg['coord_scale']+com
                phi = regional_moments(y, ref['landmarks_A'], radius_A)
                live = counts[i] > 0
                parents = np.arange(len(y)) if k == 0 else previous_parents
                hm, nh = descriptive_parent_mean(phi, parents, live)
                if (~live).any():
                    lm, nl = descriptive_parent_mean(phi, parents, ~live)
                else:
                    lm, nl = np.full(len(fields), np.nan), 0
                am.append(hm); em.append(lm); sd.append(phi.std(0))
                # Credit strata are descriptive only. Sparse early ancestors
                # cannot supply a full-window high/low affinity test.
                labels = credit[i, live]
                distinct = len(np.unique(labels))
                row = dict(batch=batch, score_time=float(times[k]), live_ancestors=int(live.sum()),
                    extinct_ancestors=int((~live).sum()), live_parents=nh, extinct_parents=nl,
                    terminal_credit_unique_values=distinct,
                    terminal_credit_min=float(labels.min()), terminal_credit_max=float(labels.max()),
                    terminal_credit_mean_equal_ancestors=float(labels.mean()),
                    terminal_credit_range=float(np.ptp(labels)),
                    descendant_count_max=int(counts[i].max()))
                rows.append(row)
                slots = diverse_survivors(y, credit[i], counts[i], ref['landmarks_A'], ref['origin_A'], teachers_per_batch)
                teachers.setdefault(k, []).append(dict(batch=batch, slots=slots,
                    endpoint=y[slots].astype(np.float32).tolist(), scores=credit[i, slots].tolist(),
                    descendants=counts[i, slots].tolist()))
                previous_parents = selected[i]
        alive_means.append(am); extinct_means.append(em); scales.append(sd)
    am, em = np.asarray(alive_means), np.asarray(extinct_means)
    scale = np.maximum(np.mean(scales, axis=(0, 1)), 1e-6)
    effect = (am-em)/scale
    complete = np.isfinite(effect).all(axis=(1, 2))
    if complete.sum() < 3:
        raise ValueError('Insufficient complete independent batches for survival contrast')
    # Only complete batches enter a whole-window integral. Missing strata stay
    # NaN in compact tables rather than synthetic zero-contrast observations.
    elapsed = times[-1]-times[0]
    measures = {'survival_enrichment_z': np.trapezoid(effect[complete], times, axis=1)/elapsed,
                'survival_enrichment_change_rate_z': (effect[complete, -1]-effect[complete, 0])/elapsed}
    pvalues = np.concatenate([np.nan_to_num(ttest_1samp(v, 0, axis=0).pvalue, nan=1.) for v in measures.values()])
    qvalues = bh(pvalues)
    rng = np.random.default_rng(42)
    boot = rng.integers(0, int(complete.sum()), (2000, int(complete.sum())))
    effects = []
    for m, (name, value) in enumerate(measures.items()):
        ci = np.quantile(value[boot].mean(1), [.025, .975], axis=0)
        for j, field in enumerate(fields):
            effects.append(dict(feature=field, measure=name, effect_z=float(value[:, j].mean()),
                CI_low=float(ci[0, j]), CI_high=float(ci[1, j]), p=float(pvalues[m*len(fields)+j]),
                q=float(qvalues[m*len(fields)+j]), positive_batch_fraction=float((value[:, j] > 0).mean()),
                independent_batches=int(complete.sum())))
    table = pd.DataFrame(rows)
    moments = pd.DataFrame({'batch': np.repeat(batches, len(times)), 'score_time': np.tile(times, len(batches))})
    moments = pd.concat([moments, pd.DataFrame({f'{kind}__{f}': arr[:, :, j].reshape(-1)
        for kind, arr in (('live', am), ('extinct', em)) for j, f in enumerate(fields)})], axis=1)
    teacher_ref = copy.deepcopy(ref)
    for k, frame in enumerate(teacher_ref['frames']):
        group = teachers[k]
        frame['teacher_endpoint_A'] = sum((r['endpoint'] for r in group), [])
        frame['teacher_scores'] = sum((r['scores'] for r in group), [])
        frame['teacher_batches'] = sum(([r['batch']]*len(r['slots']) for r in group), [])
        frame['teacher_slots'] = sum((r['slots'] for r in group), [])
        frame['teacher_descendant_counts'] = sum((r['descendants'] for r in group), [])
        frame['teacher_base_log_weight'] = sum(([-float(np.log(len(r['slots'])))]*len(r['slots']) for r in group), [])
        # Proposal teachers inherited from the instantaneous-score library do
        # not match these endpoint modes; remove, rather than silently reuse.
        frame.pop('teacher_proposal_A', None)
    teacher_ref.update(schema_version='affinity-endpoint-library-1.0',
        reference_variant='terminal-descendant-endpoint-library-1.0',
        teachers_per_batch_max=teachers_per_batch,
        teacher_label_source='last_recorded_preselection_joint_latent_head',
        teacher_selection='surviving ancestors ranked by mean descendant credit; equal batch then ancestor prior',
        teacher_distinctness='Fixed-frame Hungarian RMS > 0.001 Angstrom; excludes near copies, not guaranteed separated basins',
        label_semantics='Mean recorded step99 joint-latent pIC50 of observable descendants; not decoded t=1 affinity',
        control_representation='predicted_endpoint', parent_reference_sha256=digest(reference),
        terminal_metadata=terminal_metadata, allowed_reward_views=['endpoint_pointcloud'],
        limitations=ref.get('limitations', [])+[
            'Extinction is missing future affinity, never a low-score label',
            'Late future genealogy can label early geometry; coordinate support stays within the learned window',
            'Early surviving root collapse reduces teacher diversity and causes survivor bias',
            'No causal affinity proof; offline last-record head differs from final decoded rescore',
            'Survival contrast includes equal extant-parent weighting; residual root dependence remains'])
    if digest(__file__) != module_sha:
        raise ValueError('Mining source changed during execution; rerun in a new output folder')
    out.mkdir(parents=True)
    moments.to_parquet(out/'batch_node_survival_moments.parquet', compression=None, index=False)
    table.to_parquet(out/'batch_node_credit.parquet', compression=None, index=False)
    pd.DataFrame(effects).to_parquet(out/'whole_window_survival_effects.parquet', compression=None, index=False)
    path = out/'terminal_reference.json.gz'
    path.write_bytes(gzip.compress(json.dumps(teacher_ref, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    mean = effect[complete].mean(0)
    functions = {f: {'values_z': mean[:, j].tolist(), 'd_dt_z': np.gradient(mean[:, j], times).tolist(),
                    'scale': float(scale[j])} for j, f in enumerate(fields)}
    write_json(out/'survival_functions.json', dict(window=ref['window'], times=ref['times'],
        functions=functions, function_definition='Piecewise linear observed whole-window node means; no extrapolation',
        credit_strata_tested=False, reason='Sparse terminal ancestors prevent complete-window affinity strata inference'))
    manifest = dict(schema_version='terminal-credit-mining-1.0', sources=ref['sources'],
        source_code_sha256=module_sha, reference_sha256=digest(path),
        parent_reference_sha256=digest(reference), window=ref['window'], times=ref['times'],
        terminal_metadata=terminal_metadata, complete_survival_batches=int(complete.sum()),
        total_batch_nodes=len(table), nodes_with_fewer_than3_ancestors=int((table.live_ancestors < 3).sum()),
        teachers_per_batch_max=teachers_per_batch,
        nodes_with_fewer_than6_ancestors=int((table.live_ancestors < 6).sum()),
        teacher_counts_per_node=[sum(len(r['slots']) for r in teachers[k]) for k in range(len(times))],
        multiple_testing='BH jointly over 400 survival field/measure tests; not terminal affinity effects',
        storage='Two parent-debiased moment strata and one compact credit table; no expanded candidate cache',
        table_bytes={p.name: p.stat().st_size for p in out.glob('*.parquet')})
    write_json(out/'manifest.json', manifest)
    return manifest
