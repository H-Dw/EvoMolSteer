"""Standalone evidence figure; no inference, fitting or candidate selection."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evomolsteer.generation.path_campaign_report import qualified_summary
from evomolsteer.io import digest, read_json, write_json


def plot(reports, summary, output):
    reports, summary, output = map(Path, (reports, summary, output))
    result = read_json(summary)
    threshold = result['threshold_pic50']
    control = pd.read_csv(reports / 'round04/candidate_metrics.csv')
    native = control[(control.arm == 'unguided') & (control.batch == 30)].pic50_on_rescore.mean()
    incumbent = control[(control.arm == 'gradient') & (control.batch == 30)].pic50_on_rescore.mean()
    rows = []
    for number in range(5, 18):
        folder = reports / ('round08_replay' if number == 8 else f'round{number:02d}')
        data = pd.read_csv(folder / 'candidate_metrics.csv')
        data = data[(data.arm == 'gradient') & (data.batch == 30)]
        metric = qualified_summary(data, threshold)
        rows.append((number, metric['all_mean_pic50'] - native, metric['elite_pb_fast_yield'] * 100))
    rounds, effects, hits = map(np.array, zip(*rows))
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'evomolsteer-path-campaign'})
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3), constrained_layout=True)
    colors = ['#16746e' if r == result['winner_round'] else '#4778a3' for r in rounds]
    axes[0].bar(rounds, effects, color=colors)
    axes[0].axhline(0, color='#444', linewidth=.7)
    axes[0].axhline(incumbent - native, color='#9f492a', linestyle='--', label='Old incumbent')
    axes[0].set(title='Adaptive screens: mean affinity', xlabel='Round (same batch 30; n=50)',
                ylabel='All-output predicted pIC50 minus native')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(rounds, hits, color=colors)
    axes[1].set(title='Adaptive screens: qualified elite yield', xlabel='Round',
                ylabel='Valid + PB-fast elites / all attempts (%)')
    axes[1].text(.02, .96, f"{result['adaptive_elites']['records']} hit records; "
                 f"{result['adaptive_elites']['unique_graphs']} distinct graphs\nRepeated hits are not independent discoveries",
                 transform=axes[1].transAxes, va='top', fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks(rounds);ax.tick_params(axis='x', labelsize=8)
    evidence = result['selected_vs_native_pooled']
    batches = [r['batch'] for r in evidence['batch_details']]
    delta = [r['paired_mean_pic50'] for r in evidence['batch_details']]
    axes[2].scatter(batches, delta, color='#16746e', s=45, zorder=3)
    axes[2].axhline(0, color='#444', linewidth=.7)
    mean = evidence['paired_mean_pic50'];ci = evidence['batch_bootstrap_CI95']
    axes[2].axhline(mean, color='#16746e', label='Pooled paired mean')
    if ci:axes[2].axhspan(*ci, color='#16746e', alpha=.15, label='Batch bootstrap CI95')
    axes[2].set_xticks(batches)
    axes[2].set(title=f"Frozen round {result['winner_round']}: independent panels",
                xlabel='Batch ID (master seed 42)', ylabel='Paired predicted pIC50 gain over native')
    axes[2].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Path-guidance evidence: discovery and frozen confirmation are separate', fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        path = output / ('path_campaign.' + suffix)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path, dpi=180, metadata={'Date': None} if suffix == 'svg' else {})
        if suffix == 'svg':
            # Matplotlib path lines have cosmetic trailing spaces; preserve XML
            # line separators while making the exported artifact diff-clean.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                            encoding='utf-8', newline='\n')
    plt.close(fig)
    write_json(output / 'figure_provenance.json', {'summary_sha256': digest(summary), 'plot_code_sha256': digest(__file__),
                                                'screen_comparison': 'batch30 only; repaired round08 replay',
                                                'confirmation': 'frozen selected versus native; no validation tuning'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', required=True)
    parser.add_argument('--summary', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.reports, args.summary, args.output)
