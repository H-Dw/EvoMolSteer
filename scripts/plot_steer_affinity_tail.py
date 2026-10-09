"""Plot compact tail evidence without a persistent coordinate feature cache."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from evomolsteer.continuous.affinity_tail import comparison_metrics
from evomolsteer.io import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True)
    p.add_argument('--metrics', action='append', required=True, help='NAME=PATH#ARM')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.analysis), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    threshold = read_json(root/'summary.json')['threshold']
    panel = pd.read_csv(root/'quality_tail_panel.csv')
    budget = panel.strain_budget.iloc[0]
    named = {name: comparison_metrics(spec)[0] for name, spec in (v.split('=', 1) for v in args.metrics)}
    colors = {'native': '#777777', 'R26': '#6f81b5', 'R11': '#198f87', 'steer': '#d85e2b'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name, m in named.items():
        valid = m[m.valid_connected]
        s = np.sort(valid.pic50_on_rescore)
        axes[0, 0].step(s, (len(s)-np.arange(len(s)))/len(s), where='post', label=name, color=colors.get(name))
        elite = valid[valid.pic50_on_rescore >= threshold]
        axes[0, 1].scatter(elite.pic50_on_rescore, elite.mmff_relief_per_heavy, s=22,
                           alpha=.65, label=name, color=colors.get(name))
    axes[0, 0].set(xlim=(7.8, 8.75), ylim=(.0008, 1), yscale='log',
                   xlabel='Predicted pIC50 threshold', ylabel='Valid-sample survival fraction',
                   title='Upper tail at 1,000 final slots per arm')
    axes[0, 0].axvline(threshold, linestyle=':', color='black', linewidth=.8)
    axes[0, 0].legend()
    axes[0, 1].axhline(budget, linestyle=':', color='black', linewidth=.8)
    axes[0, 1].set(yscale='log', xlabel='Final predicted pIC50', ylabel='MMFF relief (kcal/mol) / heavy atom',
                   title='Tail affinity and strain; dotted line = native P90')
    paths = pd.read_csv(root/'elite_paths.csv')
    examples = pd.read_csv(root/'elite_terminal_features.csv').head(1)
    eligible = pd.read_csv(root/'elite_terminal_features.csv')
    for _, f in eligible[eligible.mmff_relief_per_heavy <= budget].groupby('batch'):
        examples = pd.concat([examples, f.nlargest(1, 'pic50_on_rescore')])
    examples = examples.nlargest(3, 'pic50_on_rescore')
    for e in examples.itertuples():
        path = paths[(paths.batch == e.batch) & (paths.final_slot == e.slot) & paths.actual_selection_event]
        axes[1, 0].plot(path.score_time, path.score_rank_fraction, linewidth=1.2,
                        label=f'b{e.batch:02d}/slot{e.slot:02d}; final={e.pic50_on_rescore:.3f}')
    axes[1, 0].set(xlabel='Actual selection score time', ylabel='Within-batch online score percentile',
                   ylim=(0, 1.04), title='Elite ancestors were not always the highest-score nodes')
    axes[1, 0].legend(fontsize=8)
    correlations = pd.read_csv(root/'suffix_correlations.csv')
    s = correlations.groupby('score_time').spearman.agg(['median', 'min', 'max'])
    positions = np.arange(len(s))
    axes[1, 1].bar(positions, s['median'], color='#597a88')
    axes[1, 1].set_xticks(positions, [f'{x:.2f}' for x in s.index])
    axes[1, 1].set(ylim=(0, 1), xlabel='Diagnostic score time', ylabel='Median within-batch Spearman correlation',
                   title='Online score vs. decoded final score (20 batches)')
    for i, value in enumerate(s['median']):
        axes[1, 1].text(i, value+.025, f'{value:.3f}', ha='center', fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out/'tail_and_genealogy.png', dpi=170)
    plt.close(fig)
    legends = [f'b{e.batch:02d}/slot{e.slot:02d}\npIC50={e.pic50_on_rescore:.3f}; relief/atom={e.mmff_relief_per_heavy:.3f}'
               for e in examples.itertuples()]
    Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in examples.smiles], molsPerRow=3,
                         subImgSize=(420, 300), legends=legends).save(str(out/'tail_example_graphs.png'))


if __name__ == '__main__':
    main()
