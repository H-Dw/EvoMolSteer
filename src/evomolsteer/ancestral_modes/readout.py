"""Deterministic campaign readout; no generation, head inference or redesign."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json, write_json, digest
from evomolsteer.local_motif.evaluation import original_reference, score_records, paired


def quality_summary(energy_rows, pb_rows, n):
    values = [r['heavy_relaxation_relief_after_H'] for r in energy_rows
              if r.get('heavy_relaxation_relief_after_H') is not None]
    passing = [r['pic50'] for r in pb_rows if r.get('passes_all') and r.get('pic50') is not None]
    return {'attempts': n, 'energy_coverage': len(values),
            'energy_converged': sum(r.get('status') == 'converged' for r in energy_rows),
            'energy_median': float(np.median(values)) if values else None,
            'energy_p90': float(np.quantile(values, .9)) if values else None,
            'PB_evaluated': sum(r.get('status') == 'evaluated' for r in pb_rows),
            'PB_pass_all': sum(bool(r.get('passes_all')) for r in pb_rows),
            'PB_unresolved': sum(r.get('status') == 'evaluated' and
                                 any(v is None for v in r.get('tests', {}).values()) for r in pb_rows),
            'PB_best_pic50': max(passing, default=None)}


def batch_bootstrap(a_groups, b_groups, tau, seed):
    """Resample paired initial-state B50 groups, rather than individual particles."""
    differences = []
    for a, b in zip(a_groups, b_groups):
        x = np.asarray([r['pic50'] for r in a], dtype=float)
        y = np.asarray([r['pic50'] for r in b], dtype=float)
        qa = np.asarray([r['qualified_structure'] for r in a], dtype=bool)
        qb = np.asarray([r['qualified_structure'] for r in b], dtype=bool)
        if len(a) != len(b) or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('Paired score coverage changed; no silent deletion or imputation')
        du = np.where(qa, np.maximum(x - tau, 0), 0) - np.where(qb, np.maximum(y - tau, 0), 0)
        differences.append([float((x - y).mean()), float(du.mean())])
    d = np.asarray(differences)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(d), (10000, len(d)))
    ci = np.quantile(d[ids].mean(1), [.025, .975], axis=0)
    return {'paired_batches': len(d), 'per_batch_differences': d.tolist(),
            'mean_difference': float(d[:, 0].mean()), 'mean95': ci[:, 0].tolist(),
            'U_difference': float(d[:, 1].mean()), 'U95': ci[:, 1].tolist(),
            'scope': 'Paired batch bootstrap, seed fixed; not target/prior holdout or adjusted development comparison.'}


def analyze(repo, config, output):
    repo = Path(repo).resolve(); m = read_json(config); study = repo / m['campaign_study']
    state = read_json(study / 'state.json'); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if state['next_round'] <= m['max_rounds'] or not (study / 'COMPLETE.json').exists():
        raise ValueError('Full readout requires the sealed campaign; use incremental quality diagnostics earlier')
    sources = {}
    def read(path):
        path = Path(path); sources[path.relative_to(repo).as_posix()] = digest(path); return read_json(path)
    read(study / 'state.json'); read(study / 'COMPLETE.json')
    original = original_reference(repo, m); tau = original['tau']; comparisons = []
    oq = read(study / 'original_same_assay_quality/summary.json')
    opb = read(study / 'original_same_assay_quality/current_posebusters.json')
    comparisons.append({'condition': 'original_Steer', **score_records(original['records'], tau),
                        'energy_coverage': oq['energy_coverage'], 'energy_converged': oq['converged'],
                        'energy_median': oq['median_heavy_relief_after_H_per_atom'],
                        'energy_p90': oq['p90_heavy_relief_after_H_per_atom'],
                        **{k: opb[v] for k, v in [('PB_evaluated', 'n_evaluated'), ('PB_pass_all', 'n_pass_all'),
                                                 ('PB_unresolved', 'n_unresolved'), ('PB_best_pic50', 'best_passing_pic50')]}})
    all_a = []; all_b = []; group_a = []; group_b = []; pools = []; execution_rows = []
    quality_root = study / 'current_terminal_quality'
    for condition in state['rounds']:
        if condition['stage'] != 'confirmation': continue
        pa = []; pb = []; er = {'guidance': [], 'unguided': []}; pr = {'guidance': [], 'unguided': []}
        if len(condition['result_folders']) != len(condition['unguided_result_folders']):
            raise ValueError('Missing paired control binding')
        for af, bf in zip(condition['result_folders'], condition['unguided_result_folders']):
            a = repo / af; b = repo / bf; pairing = paired(a, b, tau)
            ar = read(a / 'records.json'); br = read(b / 'records.json')
            if [r['slot'] for r in ar] != [r['slot'] for r in br]: raise ValueError('Slot ordering differs')
            group_a.append(ar); group_b.append(br); pa += ar; pb += br
            for arm in er:
                tag = a.parent.name + '__' + arm
                er[arm] += read(quality_root / (tag + '.energy.json'))['rows']
                pr[arm] += read(quality_root / (tag + '.posebusters.json'))['rows']
            execution_rows.append({'pool': condition['round'], 'batch': a.parent.name,
                                   'guidance': af, 'unguided': bf, **pairing})
        all_a += pa; all_b += pb
        for arm, records in [('guidance', pa), ('unguided', pb)]:
            comparisons.append({'condition': condition['round'] + '__' + arm,
                                **score_records(records, tau), **quality_summary(er[arm], pr[arm], len(records))})
        pools.append({'round': condition['round'], **condition['confirmation']})
    for arm, records in [('guidance', all_a), ('unguided', all_b)]:
        erows = []; prows = []
        for r in execution_rows:
            tag = r['batch'] + '__' + arm
            erows += read(quality_root / (tag + '.energy.json'))['rows']
            prows += read(quality_root / (tag + '.posebusters.json'))['rows']
        comparisons.append({'condition': 'frozen_400__' + arm, **score_records(records, tau),
                            **quality_summary(erows, prows, len(records))})
    rounds = []
    for c in state['rounds']:
        p = c['program']; specs = p['mode_specs']
        rounds.append({'round': c['round'], 'phase': c['phase'], 'stage': c['stage'], 'parent': c['parent'],
                       **c['metrics'], 'rollback': c['rollback'], 'new_attempts': c['new_attempts'],
                       'clock': p['clock_policy'], 'operator': p['operator'], 'routing': p['routing'],
                       'support_sizes': ','.join(str(len(s['atom_slots'])) for s in specs),
                       'typed_weight': p['typed_weight'], 'charge_weight': p['charge_weight'],
                       'native_rms_ratio': p['dose']['native_rms_ratio'], 'dose_schedule': p['dose_schedule'],
                       'numeric_identity': c['numeric_identity'], 'inference_commit': c['commit']})
    # Root manifests count conditions, including smoke/suffix and newly required controls.
    generated = []; attempted = 0
    for manifest in sorted((repo / m['generated_root']).glob('*/COMPLETE.json')):
        c = read(manifest)
        generated.append({'folder': manifest.parent.name, 'manifest': c})
        for folder in manifest.parent.iterdir():
            if folder.is_dir() and (folder / 'records.json').exists():
                attempted += len(read(folder / 'records.json'))
    summary = {'comparisons': comparisons, 'paired_batches': execution_rows, 'pools': pools,
               'combined_paired': batch_bootstrap(group_a, group_b, tau, m['seed']),
               'confirmation_passed': state['confirmed'], 'frozen_selection': state['frozen_selection'],
               'original_target_pic50': original['best'], 'total_new_terminal_attempts': attempted,
               'generated_manifests': generated, 'sources': sources,
               'limits': ['33 conditions repeatedly reuse one development initial pool; its best score is search-biased.',
                          'Confirmation uses eight fresh guided terminal pools but known historical Steer priors and one seed.',
                          'Original Steer has ancestry dependence and a different sampling budget; maximum comparisons are descriptive.',
                          'PB unresolved is not pass; MMFF heavy relaxation after H placement is a local proxy, not binding energy.',
                          'Scored failed structures remain in all-attempt score summaries and have zero qualified tail utility.',
                          'Real API binding and executed gradients do not establish causal LLM benefit over deterministic search.']}
    write_json(output / 'summary.json', summary)
    pd.DataFrame(comparisons).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(rounds).to_csv(output / 'rounds.csv', index=False)
    pd.DataFrame(execution_rows).to_csv(output / 'paired_batches.csv', index=False)
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('--repo', default='.'); p.add_argument('--config', required=True)
    p.add_argument('--output', required=True); a = p.parse_args()
    r = analyze(a.repo, a.config, a.output)
    print({k: r[k] for k in ('confirmation_passed', 'combined_paired', 'total_new_terminal_attempts')})


if __name__ == '__main__': main()
