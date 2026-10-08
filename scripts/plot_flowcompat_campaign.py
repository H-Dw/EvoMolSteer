"""Plot retained paired scores and strain without fitting or selecting a model."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evomolsteer.io import digest, read_json, write_json


def plot(input_dataset, output_dataset):
    source, output = Path(input_dataset).resolve(), Path(output_dataset).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('Keep exported figures outside the retained evidence dataset')
    table_path = source / 'rounds_summary.csv'
    table = pd.read_csv(table_path)
    if table.empty or table['round'].duplicated().any():
        raise ValueError('Nonempty one-row-per-round retained summary required')
    table = table.sort_values('round')
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = np.where(table['round'].ge(25), '#2368a0',
                      np.where(table['screening_admissible'], '#398453', '#b36740'))
    for _, row in table.iterrows():
        if pd.notna(row['vs_R26']):
            axes[0].vlines(row['round'], 0, row['vs_R26'], color='#999999', linewidth=.8)
    axes[0].scatter(table['round'], table['vs_R26'], c=colors, s=35)
    axes[0].axhline(0, color='#444444', linewidth=.8)
    axes[0].set_ylabel('Paired mean pIC50 difference vs R26')
    axes[0].set_title(f'Actual FLOWR trials: {len(table)}/30 retained rounds')
    axes[1].plot(table['round'], table['strain_median'], 'o-', color='#398453', markersize=4, label='Median')
    axes[1].plot(table['round'], table['strain_p90'], 'o-', color='#b36740', markersize=4, label='P90')
    axes[1].set_ylabel('MMFF strain per heavy atom (kcal/mol)')
    axes[1].set_xlabel('Round (controls and negative-direction tests included)')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.axvline(24.5, linestyle='--', color='#666666', linewidth=1)
        axis.spines[['top', 'right']].set_visible(False)
    axes[1].set_xticks(range(1, 31))
    fig.text(.08, .015,
             'Green: screen-admissible exploratory trial; brown: other screening; blue: held-out panel. '
             'Point differences are not independent effect estimates.', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(output / 'paired_performance.png', dpi=180)
    plt.close(fig)
    table.to_csv(output / 'plotted_rounds.csv', index=False)
    write_json(output / 'manifest.json', {
        'input_dataset': str(source), 'input_files': {str(table_path): digest(table_path)},
        'script_sha256': digest(__file__),
        'semantics': 'Descriptive retained paired means; no new efficacy test or candidate selection',
        'outputs': {p.name: digest(p) for p in sorted(output.iterdir())
                    if p.is_file() and p.name != 'manifest.json'},
    })
    return {'retained_rounds': len(table)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dataset', required=True)
    parser.add_argument('--output-dataset', required=True)
    args = parser.parse_args()
    print(plot(args.input_dataset, args.output_dataset))
