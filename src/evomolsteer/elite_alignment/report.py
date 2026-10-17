"""Compact cohort views, failure coverage and paired root-cluster intervals."""
import argparse
from pathlib import Path
import numpy as np

from evomolsteer.io import read_json, write_table
from evomolsteer.storage.transactions import atomic_json
from .common import result_path


def cluster_interval(attempts, values):
    roots = sorted({x['initial_id'] for x in attempts})
    grouped = [np.asarray([v for a, v in zip(attempts, values) if a['initial_id'] == root], float) for root in roots]
    sums = np.asarray([x.sum() for x in grouped]); counts = np.asarray([len(x) for x in grouped])
    rng = np.random.default_rng(42); sampled = rng.integers(0, len(roots), (10000, len(roots)))
    means = sums[sampled].sum(1) / counts[sampled].sum(1)
    return np.quantile(means, [.025, .975]).tolist()


def report(work, config, bootstrap=True):
    import pandas as pd
    work = Path(work); index = {}
    for condition in config['conditions']:
        index[condition['id']] = {a['id']: read_json(result_path(work, condition, a))
            for a in config['attempts'] if result_path(work, condition, a).exists()}
    rows, pairs = [], []
    for cohort, spec in config['cohorts'].items():
        attempts = config['attempts'][:spec['n']]
        ids = [a['id'] for a in attempts]
        for condition in config['conditions']:
            records = [index[condition['id']].get(k) for k in ids]
            complete = [r for r in records if r and r['status'] == 'complete']
            values = np.asarray([r['pic50'] for r in complete], float)
            valid = [r['pic50'] for r in complete if r.get('valid_connected')]
            qualified = [r['pic50'] for r in complete if r.get('qualified_structure')]
            root_means = [np.mean([r['pic50'] for r in complete if r['attempt']['initial_id'] == root])
                          for root in sorted({r['attempt']['initial_id'] for r in complete})]
            energy = [r['mmff_relief_per_heavy'] for r in complete if r.get('energy_status') == 'converged' and r.get('mmff_relief_per_heavy') is not None]
            rows.append({'cohort': cohort, 'condition': condition['id'], 'n_planned': len(attempts),
                'n_complete': len(complete), 'n_failed': sum(r is not None and r['status'] == 'failed' for r in records),
                'n_skipped': sum(r is not None and r['status'] == 'skipped_condition_error' for r in records),
                'n_pending': sum(r is None for r in records), 'score_coverage': len(complete)/len(attempts),
                'mean_pic50': float(values.mean()) if len(values) else None,
                'root_equal_mean_pic50': float(np.mean(root_means)) if root_means else None,
                'best_all_scored': float(values.max()) if len(values) else None,
                'best_valid_connected': max(valid) if valid else None, 'best_qualified': max(qualified) if qualified else None,
                'n_valid_connected': len(valid), 'n_qualified': len(qualified),
                'tail_valid_count': sum(v >= config['tail_threshold'] for v in valid),
                'tail_valid_rate_all_attempts': sum(v >= config['tail_threshold'] for v in valid)/len(attempts),
                'above_steer_best_valid_count': sum(v > config['steer_best'] for v in valid),
                'historical_selected_mean': float(np.mean([a['historical_pic50'] for a in attempts])),
                'historical_selected_best': max(a['historical_pic50'] for a in attempts),
                'at_least_own_historical_score': sum(r['pic50'] >= r['attempt']['historical_pic50'] for r in complete),
                'n_converged_energy': len(energy), 'strain_median': float(np.median(energy)) if energy else None,
                'strain_p90': float(np.quantile(energy, .9)) if energy else None,
                'n_pb_evaluated': sum(r.get('pb_status') == 'evaluated' for r in complete),
                'n_pb_pass': sum(r.get('pb_pass') is True for r in complete),
                'pb_status': 'not_evaluated' if not any('pb_pass' in r for r in complete) else 'available'})
            for baseline in ('unguided', 'M2'):
                paired = [(a, index[condition['id']].get(a['id']), index[baseline].get(a['id'])) for a in attempts]
                paired = [(a, r, b) for a, r, b in paired if r and b and r['status'] == b['status'] == 'complete']
                if not paired or condition['id'] == baseline: continue
                differences = np.asarray([r['pic50']-b['pic50'] for a, r, b in paired])
                interval = cluster_interval([a for a, r, b in paired], differences) if bootstrap else [None, None]
                pairs.append({'cohort': cohort, 'condition': condition['id'], 'baseline': baseline,
                    'n_pairs': len(paired), 'n_roots': len({a['initial_id'] for a, r, b in paired}),
                    'mean_delta_pic50': float(differences.mean()), 'cluster_bootstrap95_low': interval[0],
                    'cluster_bootstrap95_high': interval[1], 'interval_scope': 'Exploratory marginal interval; not simultaneous superiority evidence'})
    output = work / 'analysis'; output.mkdir(exist_ok=True)
    write_table(output / 'cohort_metrics.parquet', pd.DataFrame(rows))
    write_table(output / 'paired_comparisons.parquet', pd.DataFrame(pairs))
    atomic_json(output / 'summary.json', {'protocol_sha256': config['protocol_sha256'], 'metrics': rows, 'pairs': pairs,
        'inference_count': sum(len(x) for x in index.values()), 'interpretation': config['interpretation'],
        'PB': 'Not silently inferred from basic structural validity; separate PB evaluation is required.',
        'multiplicity': 'No confirmatory significance claim; nested cohorts, cloned roots, and multiple methods are dependent.'})
    text = '# 优质初态对齐复测\n\n'+config['interpretation']+'\n\n'
    text += '| 集合 | 方法 | 完成/计划 | 均值 | 有效Best | 高分数 | 失败/跳过 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in rows:
        fmt = lambda x: '—' if x is None else f'{x:.6f}'
        text += f"| {r['cohort']} | {r['condition']} | {r['n_complete']}/{r['n_planned']} | {fmt(r['mean_pic50'])} | {fmt(r['best_valid_connected'])} | {r['tail_valid_count']} | {r['n_failed']}/{r['n_skipped']} |\n"
    text += '\n原始Steer优质集合属于终态后筛选参照；新推理关闭SMC。均值仅使用有限评分，同时保留全部计划尝试分母及评分覆盖。根聚类区间为探索性边际区间，不据此宣称多重比较显著。\n'
    (output / 'report.zh-CN.md').write_bytes(text.encode('utf-8'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--work', required=True)
    a = p.parse_args(); report(a.work, read_json(a.config))


if __name__ == '__main__': main()
