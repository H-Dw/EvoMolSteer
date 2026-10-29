"""Summarize completed read-only audit; preserves all raw PoseB outcomes."""
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).parent
EVIDENCE = BASE/'evidence'


def main():
    results = json.loads((EVIDENCE/'RESULTS.json').read_text(encoding='utf-8'))
    attempts = json.loads((EVIDENCE/'ATTEMPTS.json').read_text(encoding='utf-8'))
    chemistry = json.loads((EVIDENCE/'CHEMISTRY.json').read_text(encoding='utf-8'))
    hydrogen = json.loads((EVIDENCE/'HYDROGEN_ENERGY_SENSITIVITY_RAW.json').read_text(encoding='utf-8'))
    for row in hydrogen:
        row['usable'] = (row['convergence_code'] == 0 and row['hydrogen_relaxed_energy'] <= row['unrelaxed_copy_energy']+1e-6)
        row['sensitivity_status'] = 'usable' if row['usable'] else 'unusable_optimization'
    (EVIDENCE/'HYDROGEN_ENERGY_SENSITIVITY.json').write_text(json.dumps(hydrogen, indent=2)+'\n', encoding='utf-8', newline='\n')
    generated = [r for r in results if r['kind']=='generated']
    summary = []
    for scale in sorted({r['scale'] for r in attempts}):
        measured = [r for r in generated if r['scale']==scale]
        population = [r for r in attempts if r['scale']==scale]
        identities = [r for r in chemistry if r['scale']==scale]
        failures = Counter(k for r in measured for k in r['failed_checks'])
        row = dict(scale=scale, attempts=len(population), checked_built=len(measured),
                   build_failed=sum(not r['build_success'] for r in population),
                   connected=sum(r['connected'] for r in measured),
                   posebusters_pass=sum(r['posebusters_status']=='passed' for r in measured),
                   intramolecular_pass=sum(r['intramolecular_status']=='passed' for r in measured),
                   protein_context_pass=sum(r['protein_context_status']=='passed' for r in measured),
                   unknown=sum(bool(r['unknown_checks']) for r in measured),
                   evaluation_errors=sum(r['status']!='evaluated' for r in measured),
                   nonenergy_failure=sum(any(k!='internal_energy' for k in r['failed_checks']) for r in measured),
                   condition_coordinate_rmsd_mean_A=mean(r['condition_coordinate_rmsd_A'] for r in population),
                   limiter_events=sum(r['limiter_events'] for r in population),
                   limiter_fraction=mean(r['limiter_fraction'] for r in population),
                   core_chemistry_matches=sum(r['core_chemistry_matches'] for r in identities),
                   failures=dict(failures))
        summary.append(row)
    with (BASE/'summary.csv').open('w', encoding='utf-8', newline='') as f:
        columns = [k for k in summary[0] if k!='failures']
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k!='failures'} for row in summary)
    high = [r for r in generated if r['scale']>=256]
    (BASE/'summary.json').write_text(json.dumps(dict(
        per_scale=summary,
        all_generated_failures=dict(Counter(k for r in generated for k in r['failed_checks'])),
        high_strength=dict(built=len(high), posebusters_pass=sum(r['posebusters_status']=='passed' for r in high),
                           intramolecular_pass=sum(r['intramolecular_status']=='passed' for r in high)),
        workers=sorted({r['pid'] for r in results}),
        controls=[r for r in results if r['kind']!='generated'],
        hydrogen_sensitivity=hydrogen), indent=2)+'\n', encoding='utf-8', newline='\n')
    plot(summary)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].plot(x, [100*r['posebusters_pass']/r['checked_built'] for r in rows], 'o-', label='PoseB dock pass / built')
    axes[0].plot(x, [100*r['intramolecular_pass']/r['checked_built'] for r in rows], 's--', label='Intramolecular pass / built')
    axes[0].plot(x, [100*r['limiter_fraction'] for r in rows], 'x-', label='Clipped candidate-steps / 900')
    axes[0].set(ylabel='Percent', ylim=(-5,105), title='6pyy: fixed saved coarse-screen population (10 attempts per strength)')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[1].plot(x, [r['condition_coordinate_rmsd_mean_A'] for r in rows], 'o-', color='#3949ab')
    axes[1].set(ylabel='Condition coordinate RMSD (angstrom)', xlabel='Actual guidance strength; shared seeds and slots')
    axes[1].set_xticks(x, [f'{r["scale"]:g}' for r in rows], rotation=45, ha='right')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(BASE/'quality_vs_strength.png', dpi=190)
    fig.savefig(BASE/'quality_vs_strength.svg')
    plt.close(fig)
    svg = BASE/'quality_vs_strength.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')


if __name__=='__main__':
    main()
