"""Audit best-of-budget temporal candidates without generation or new scoring.

Report raw maxima, qualified top poses, graph-deduplicated shortlists, and a
descriptive batch-preserving subsample of the frozen historical Steer panel.
The optional strain envelope is a secondary diagnostic, not a new primary rank.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json


def collect(repo, config_path, previous_path, output, top_k=10, repeats=5000, seed=42):
    repo = Path(repo).resolve()
    previous = read_json(repo / previous_path)
    frozen = {str(k).replace('\\', '/'): v for k, v in previous['source_sha256'].items()}
    sources = {}

    def pin(name):
        path = (repo / str(name).replace('\\', '/')).resolve()
        key = path.relative_to(repo).as_posix()
        sha = digest(path)
        if key in frozen and sha != frozen[key]:
            raise ValueError(f'Frozen input changed: {key}')
        sources[key] = sha
        return path

    pin(previous_path)
    pin(__file__)
    config = read_json(pin(config_path))
    state = read_json(pin(Path(config['output']) / 'run_status.json'))
    bank = read_json(pin(config['control_manifest']))
    tables, scopes = {}, {}

    def load(label, name, scope):
        path = pin(name)
        d = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        if d.duplicated(['batch', 'slot']).any() or not np.isfinite(d.pic50_on_rescore).all():
            raise ValueError('Unique, completely scored attempts required')
        if any(d[k].dtype != bool for k in ['valid_connected', 'pb_fast_pass']):
            raise ValueError('Explicit boolean validity required')
        if d.loc[d.valid_connected, 'smiles'].isna().any():
            raise ValueError('Valid graph identity missing')
        tables[label], scopes[label] = d, scope

    for label, arm in [('unguided', 'unguided'), ('M2', 'gradient')]:
        load(label, bank['result']['outcomes'][arm]['metrics'], 'frozen_400')
    load('historical_Steer', config['historical_metrics'], 'historical_1000')
    for recipe in config['recipes']:
        for scope in ['confirmation', 'screen']:
            job = state['jobs'].get(f'control_{recipe}_{scope}', {})
            if job.get('status') == 'complete':
                load(recipe, job['result']['result']['outcomes']['gradient']['metrics'], scope)
                break

    m2 = tables['M2']
    energy = m2.loc[m2.energy_status.eq('converged'), 'mmff_relief_per_heavy'].dropna()
    envelope = float(energy.quantile(.9))
    threshold = config['evaluation']['elite_threshold']
    summaries, shortlists, best_rows = [], [], []
    cols = ['batch', 'slot', 'pic50_on_rescore', 'valid_connected', 'pb_fast_pass',
            'energy_status', 'mmff_relief_per_heavy', 'smiles']

    def rank(d):
        return d.sort_values(['pic50_on_rescore', 'batch', 'slot'],
                             ascending=[False, True, True], kind='stable')

    for label, d in tables.items():
        valid = rank(d.loc[d.valid_connected])
        qualified = valid.loc[valid.pb_fast_pass & valid.energy_status.eq('converged') &
                              np.isfinite(valid.mmff_relief_per_heavy)]
        unique = qualified.drop_duplicates('smiles', keep='first')
        elite = valid.loc[valid.pic50_on_rescore >= threshold]
        low_strain = qualified.loc[qualified.mmff_relief_per_heavy <= envelope]
        groups = {'raw': rank(d), 'valid': valid, 'PB_converged': qualified,
                  'PB_converged_within_M2_strain_p90': low_strain}
        row = {'condition': label, 'scope': scopes[label], 'attempts': len(d),
               'valid': len(valid), 'unique_valid_graphs': valid.smiles.nunique(),
               'elite_valid_poses': len(elite), 'elite_unique_graphs': elite.smiles.nunique(),
               'PB_converged': len(qualified), 'PB_converged_unique_graphs': len(unique),
               'qualified_unique_top5_mean': float(unique.head(5).pic50_on_rescore.mean()),
               'qualified_unique_top10_mean': float(unique.head(10).pic50_on_rescore.mean())}
        for name, group in groups.items():
            if not len(group):
                row['best_' + name] = None
                continue
            best = group.iloc[0]
            row['best_' + name] = float(best.pic50_on_rescore)
            if name == 'valid':
                row.update(best_batch=int(best.batch), best_slot=int(best.slot),
                           best_valid_strain=float(best.mmff_relief_per_heavy),
                           best_valid_PB=bool(best.pb_fast_pass))
            best_rows.append({'condition': label, 'selection': name, **best[cols].to_dict()})
        for name, group in [('PB_converged_poses', qualified), ('PB_converged_unique_graphs', unique)]:
            for position, record in enumerate(group.head(top_k)[cols].to_dict('records'), 1):
                shortlists.append({'condition': label, 'selection': name, 'rank': position, **record})
        summaries.append(row)

    # Historical samples are dependent within batches. Sample complete batches
    # without replacement to match output count; do not simulate independent clones.
    steer = tables['historical_Steer']
    batches = np.sort(steer.batch.unique())
    sizes = steer.groupby('batch').size()
    n_batches = len(m2) // config['batch_size']
    if not (sizes == config['batch_size']).all() or len(m2) % config['batch_size']:
        raise ValueError('Equal-size complete batches required for descriptive subpanels')
    per_batch_max = steer.loc[steer.valid_connected].groupby('batch').pic50_on_rescore.max()
    if set(per_batch_max.index) != set(batches):
        raise ValueError('Every subsampled batch must have a valid molecule')
    rng = np.random.default_rng(seed)
    sub_maxima = np.array([per_batch_max.loc[rng.choice(batches, n_batches, replace=False)].max()
                          for _ in range(repeats)])
    panel = {'samples_per_panel': len(m2), 'complete_batches_per_panel': n_batches,
             'repeats': repeats, 'seed': seed,
             'best_valid_pic50_mean': float(sub_maxima.mean()),
             'best_valid_pic50_quantiles': {str(q): float(np.quantile(sub_maxima, q))
                                           for q in [.025, .5, .975]},
             'fraction_exceeding_observed_candidate_max': {
                 row['condition']: float((sub_maxima > row['best_valid']).mean())
                 for row in summaries if row['attempts'] == len(m2)},
             'meaning': 'Finite-panel output-count sensitivity; not a CI, equal-compute experiment, or new generation'}
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / 'candidate_summary.csv', index=False)
    pd.DataFrame(best_rows).to_csv(output / 'best_candidates.csv', index=False)
    pd.DataFrame(shortlists).to_csv(output / 'top_candidates.csv', index=False)
    write_json(output / 'audit.json', {
        'schema_version': 'temporal-best-candidate-audit-1.0', 'source_sha256': sources,
        'top_k': top_k, 'elite_threshold': threshold, 'secondary_strain_envelope': envelope,
        'rank': 'Affinity descending; stable batch/slot tie break; no affinity-strain scalarization',
        'qualified': 'valid_connected and dock_fast PB pass and converged finite MMFF',
        'unique_policy': 'Best qualified pose per canonical isomeric SMILES for a shortlist; not a yield estimate',
        'matched_output_count_sensitivity': panel,
        'limitations': ['Historical Steer uses more online selection compute and dependent clones',
                        'Canonical graph deduplication does not establish scaffold or 3D diversity',
                        'Predicted affinity is the same FLOWR head, not experimental or independent validation',
                        'MMFF local relaxation relief is not global strain or binding free energy',
                        'No new molecular scoring, API call, or generation was performed']})
    print(pd.DataFrame(summaries).to_string(index=False))
    print(json.dumps(panel, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', required=True)
    parser.add_argument('--comparison-report', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.top_k < 1 or args.repeats < 1:
        parser.error('Positive top-k and repeats required')
    collect(args.repo, args.config, args.comparison_report,
            args.repo / args.output, args.top_k, args.repeats, args.seed)
