"""Plot retained frozen confirmation data without fitting or changing a reward."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt'] = 'evomolsteer-outcome-confirmation-1'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dispatch_path_round import verify_retention
from evomolsteer.io import digest, read_json, write_json


def plot(reports, historical_steer, output):
    reports, historical_steer, output = map(Path, (reports, historical_steer, output))
    aggregate = read_json(reports/'aggregate/campaign_summary.json')
    maximum = aggregate['maximum_rounds']
    if aggregate['completed_rounds'] != maximum or 'frozen_confirmation' not in aggregate:
        raise ValueError('Completed frozen confirmation required')
    tables = {name: [] for name in ('candidate', 'native', 'R11_051', 'R26_051')}
    sources = []
    for number in (maximum-1, maximum):
        folder = reports/f'round{number:02d}'
        verify_retention(folder/'retention.json')
        summary = read_json(folder/'summary.json')
        candidate = next(k for k in summary['results'] if k not in ('R11_051', 'R26_051'))
        for name, cohort, arm in [('candidate', candidate, 'gradient'), ('native', candidate, 'unguided'),
                                 ('R11_051', 'R11_051', 'gradient'), ('R26_051', 'R26_051', 'gradient')]:
            path = folder/'results'/cohort/'candidate_metrics.csv'
            d = pd.read_csv(path)
            tables[name].append(d[d.arm.eq(arm)])
            sources.append({'path': str(path.resolve()), 'sha256': digest(path)})
    joined = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    joined['Steer_unpaired'] = pd.read_csv(historical_steer)
    sources.append({'path': str(historical_steer.resolve()), 'sha256': digest(historical_steer)})
    labels = {'candidate': 'Frozen final-outcome reward', 'native': 'Native',
              'R11_051': 'Historical R11', 'R26_051': 'Historical R26', 'Steer_unpaired': 'Steer (unpaired)'}
    colors = dict(zip(joined, ['#0072B2', '#666666', '#009E73', '#D55E00', '#CC79A7']))
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for name, d in joined.items():
        if d.valid_connected.dtype != bool:
            raise ValueError('Boolean decoded validity required')
        valid = d[d.valid_connected & np.isfinite(d.pic50_on_rescore)]
        score = np.sort(valid.pic50_on_rescore.to_numpy())
        energy = np.sort(valid[valid.energy_status.eq('converged')].mmff_relief_per_heavy.dropna().to_numpy())
        style = '--' if name == 'Steer_unpaired' else '-'
        axes[0, 0].step(score, np.arange(1, len(score)+1)/len(score), where='post',
                        label=f'{labels[name]} (n={len(d)}, valid={len(valid)})', color=colors[name], linestyle=style)
        axes[0, 1].step(energy, np.arange(1, len(energy)+1)/len(energy), where='post',
                        color=colors[name], linestyle=style)
    axes[0, 0].set(xlabel='Valid final predicted pIC50', ylabel='Empirical cumulative fraction', title='Final affinity distribution')
    axes[0, 0].legend(fontsize=7, loc='upper left')
    axes[0, 1].set(xlabel='Converged MMFF strain relief / heavy atom (kcal/mol)', ylabel='Empirical cumulative fraction',
                   title='Raw-pose strain distribution')
    axes[0, 1].set_xscale('symlog', linthresh=.5)
    paired = aggregate['frozen_confirmation']['paired']
    batches = sorted(joined['candidate'].batch.unique())
    for name in ('native', 'R11_051', 'R26_051'):
        axes[1, 0].plot(batches, paired[name]['batch_means'], marker='o', color=colors[name], label=f'Candidate - {labels[name]}')
    axes[1, 0].axhline(0, color='black', linewidth=.8)
    axes[1, 0].set(xlabel='Independent confirmation batch', ylabel='Paired all-slot pIC50 difference',
                   title='Batch effects; four replication units', xticks=batches)
    axes[1, 0].legend(fontsize=7)
    metrics = aggregate['frozen_confirmation']['metrics']
    metrics = {**metrics, 'Steer_unpaired': aggregate['historical_steer']}
    for index, name in enumerate(joined):
        m = metrics[name]
        values = [m['valid_n']/m['n'], m['pb_fast_rate'], m['elite_yield']]
        axes[1, 1].bar(np.arange(3)+(index-2)*.15, values, width=.14, color=colors[name], label=labels[name])
    axes[1, 1].set(xticks=np.arange(3), xticklabels=['Valid decoded', 'PB fast pass', 'Elite valid yield'],
                   ylabel='Fraction of all slots', ylim=(0, 1.05), title='Validity and tail yield (unequal Steer n)')
    for ax in axes.flat:
        ax.grid(alpha=.18)
    figure.suptitle('Frozen confirmation: 200 outputs/arm; historical Steer n=1000, unpaired', fontsize=11)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output/'confirmation.png', output/'confirmation.svg']
    for path in paths:
        figure.savefig(path, dpi=180, metadata={'Date': None} if path.suffix == '.svg' else {})
        if path.suffix == '.svg':
            # Path whitespace remains separated by newline; strip formatter-only
            # trailing spaces without changing drawing commands or coordinates.
            text = path.read_text(encoding='utf-8')
            path.write_bytes(('\n'.join(line.rstrip() for line in text.splitlines())+'\n').encode('utf-8'))
    plt.close(figure)
    write_json(output/'manifest.json', {'source_script_sha256': digest(__file__), 'matplotlib_version': matplotlib.__version__, 'input_files': sources,
        'aggregate_sha256': digest(reports/'aggregate/campaign_summary.json'),
        'outputs': [{'path': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size} for path in paths],
        'scope': 'Read-only figure; no new fitting or effectiveness test. Batch effects and selected valid distributions are distinct.'})
    return str(paths[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('reports', 'historical-steer', 'output'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    print(plot(args.reports, args.historical_steer, args.output))
