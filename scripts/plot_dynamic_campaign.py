"""Export a deterministic scientific comparison from final compact reports."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evomolsteer.io import read_json


def plot(summary, output):
    summary, output = Path(summary), Path(output)
    d = pd.read_csv(summary / 'rounds.csv')
    v = read_json(summary / 'validation.json')
    if len(d) != 10 or v['confirmation_independent_batches'] != 4:
        raise ValueError('Completed ten-round report and four confirmation batches required')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.hashsalt': 'dynamic-cohort-campaign'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    screen = d[d['round'].le(7)]
    ax = axes[0, 0]
    ax.plot(screen['round'], screen.all_mean_pic50, 'o-', color='#2563eb', label='Candidate / R26 replay')
    incumbent = float(screen.loc[screen['round'].eq(1), 'all_mean_pic50'].iloc[0])
    baseline = read_json(summary.parent / 'round01/comparison.json')['results']['unguided']['all_mean_pic50']
    ax.axhline(incumbent, color='#475569', ls='--', label='R26 screen control')
    ax.axhline(baseline, color='#d97706', ls=':', label='Native screen control')
    ax.set(xticks=range(1, 8), xlabel='Sequential round (one screening batch)', ylabel='All-attempt mean predicted pIC50')
    ax.set_title('A. Adaptive screen; no independent CI')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    names = ['selected_vs_native_pooled', 'selected_vs_native_panel_a', 'selected_vs_incumbent_panel_a']
    labels = ['vs native, 4 batches', 'vs native, panel A', 'vs R26, panel A']
    for i, name in enumerate(names):
        effect = v[name]; mean = effect['paired_mean_pic50']; low, high = effect['batch_bootstrap_CI95']
        ax.errorbar(mean, i, xerr=np.array([[mean-low], [high-mean]]), fmt='o', capsize=4, color='#2563eb')
    ax.axvline(0, color='#475569', ls='--')
    ax.set(yticks=range(3), yticklabels=labels, xlabel='Paired predicted pIC50 difference (batch-bootstrap CI95)')
    ax.set_title(f'B. Frozen round {v["candidate_round"]}; limited batch uncertainty')
    groups = [v['native'], v['selected'], v['incumbent_panel_a']]
    labels = [f'Native\npooled, n={groups[0]["n"]}', f'Candidate\npooled, n={groups[1]["n"]}',
              f'R26\npanel A, n={groups[2]["n"]}']
    x = np.arange(3)
    ax = axes[1, 0]
    ax.bar(x-.17, [m['valid_n']/m['n'] for m in groups], .34, label='Valid connected', color='#2563eb')
    ax.bar(x+.17, [m['pb_fast_rate'] for m in groups], .34, label='PB-fast pass', color='#10b981')
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel='Fraction of all attempted molecules')
    ax.set_title('C. Final physical checks; no inference graph gate')
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1, 1]
    ax.plot(x, [m['strain_median_per_heavy'] for m in groups], 'o', color='#9333ea', label='Median')
    ax.plot(x, [m['strain_p90_per_heavy'] for m in groups], 's', color='#d97706', label='90th percentile')
    ax.set(xticks=x, xticklabels=labels, ylabel='MMFF relaxation relief per heavy atom')
    ax.set_title('D. Converged valid molecules; lower relief is better')
    ax.legend(fontsize=8)
    fig.suptitle('Dynamic spatial cohorts: screening and frozen FLOWR confirmation', fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=170)
    svg = output.with_suffix('.svg')
    fig.savefig(svg, metadata={'Date': None})
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    plot(a.summary, a.output)
