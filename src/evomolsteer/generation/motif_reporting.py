"""Failure-inclusive campaign reporting from retained summaries only."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json,digest
from .prototypes import write_json


def heldout(evidence,rounds=(14,15)):
    evidence=Path(evidence);candidates=[];windows=[];sources=[]
    for number in rounds:
        base=evidence/f'round_{number:02d}/local'
        path=base/'candidate_metrics.csv';candidates.append(pd.read_csv(path))
        windows.extend(read_json(base/'window/report.json')['batch_results'])
        sources.append({'round':number,'candidate_metrics_sha256':digest(path),'window_sha256':digest(base/'window/report.json')})
    table=pd.concat(candidates,ignore_index=True);window=pd.DataFrame(windows)
    if table.duplicated(['arm','batch','slot']).any():raise ValueError('Duplicate heldout candidates')
    arms={};paired=[]
    for arm in ('unguided','gradient'):
        d=table[table.arm==arm];energy=d[d.energy_status=='converged'].mmff_relief_per_heavy.dropna()
        valid=d[d.valid_connected];unique=valid.drop_duplicates('smiles')
        arms[arm]={'attempted':len(d),'batches':sorted(map(int,d.batch.unique())),
            'all_head':float(d.pic50_on_rescore.mean()),'valid_head':float(valid.pic50_on_rescore.mean()),
            'unique_valid_head':float(unique.pic50_on_rescore.mean()),'valid':int(d.valid_connected.sum()),
            'PB_fast_pass':int(d.pb_fast_pass.sum()),'unique_graphs':len(unique),'energy_converged':len(energy),
            'MMFF_relief_per_heavy_median':float(energy.median()),'MMFF_relief_per_heavy_p90':float(energy.quantile(.9)),
            'surround_relax_RMS_A':float(d.relax_rms_surround_A.mean()),
            'boundary_shape_A':float(np.average(window.loc[window.arm==arm,'symmetric_shape_A'],weights=window.loc[window.arm==arm,'n']))}
    for batch in sorted(table.batch.unique()):
        ds={arm:table[(table.arm==arm)&(table.batch==batch)] for arm in arms}
        g,n=ds['gradient'],ds['unguided']
        if not np.array_equal(g.slot.to_numpy(),n.slot.to_numpy()) or not np.array_equal(g.seed.to_numpy(),n.seed.to_numpy()):raise ValueError('Heldout initial-slot alignment differs')
        wg=window[(window.arm=='gradient')&(window.batch==batch)].iloc[0]
        wn=window[(window.arm=='unguided')&(window.batch==batch)].iloc[0]
        eg=g[g.energy_status=='converged'].mmff_relief_per_heavy;en=n[n.energy_status=='converged'].mmff_relief_per_heavy
        paired.append({'batch':int(batch),'n_per_arm':len(g),'head_change':float(g.pic50_on_rescore.mean()-n.pic50_on_rescore.mean()),
            'MMFF_median_change':float(eg.median()-en.median()),
            'MMFF_p90_change':float(eg.quantile(.9)-en.quantile(.9)),
            'surround_RMS_change_A':float(g.relax_rms_surround_A.mean()-n.relax_rms_surround_A.mean()),
            'boundary_shape_change_A':float(wg.symmetric_shape_A-wn.symmetric_shape_A),
            'valid_count_change':int(g.valid_connected.sum()-n.valid_connected.sum()),'PB_count_change':int(g.pb_fast_pass.sum()-n.pb_fast_pass.sum())})
    g,n=arms['gradient'],arms['unguided']
    comparison={'head_change':g['all_head']-n['all_head'],
        'shape_improvement':1-g['boundary_shape_A']/n['boundary_shape_A'],
        'MMFF_median_improvement':1-g['MMFF_relief_per_heavy_median']/n['MMFF_relief_per_heavy_median'],
        'MMFF_p90_improvement':1-g['MMFF_relief_per_heavy_p90']/n['MMFF_relief_per_heavy_p90'],
        'surround_RMS_improvement':1-g['surround_relax_RMS_A']/n['surround_relax_RMS_A']}
    return {'schema_version':'motif-heldout-report-1.0','arms':arms,'paired_batch_changes':paired,'comparison_vs_matched_native':comparison,
        'sources':sources,'uncertainty_unit':'Three independent batches; candidate/clone rows are not independent experimental replicates',
        'scope':'Descriptive frozen-reward heldout check; no selection or retuning from these results',
        'limitations':['Head shares the original selection oracle and is not measured affinity.',
            'Isolated-ligand MMFF relief and surrounding relaxation displacement are proxies, not receptor binding energy or measured compatibility.',
            'Boundary shape is distance to the fixed discovery Steer pool, not a graph identity target. Native graphs remain free.',
            'Historical Steer terminal100 is a discovery benchmark, not a paired heldout control.']}


def report(evidence,output,require_complete=False):
    from .motif_campaign import summarize
    evidence,output=Path(evidence),Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[read_json(p) for p in sorted(evidence.glob('round_*.outcome.json'))]
    if require_complete and [r['round'] for r in rows]!=list(range(1,16)):raise ValueError('All fifteen retained outcomes required')
    summarize(evidence,output/'summary.md')
    if not {14,15}<=set(r['round'] for r in rows):return output
    combined=heldout(evidence);write_json(output/'heldout_summary.json',combined)
    g=combined['arms']['gradient'];n=combined['arms']['unguided'];steer=read_json(evidence.parent/'ck2_terminal_seed42_20261005/reference/terminal_report.json')['results']['single']
    lines=['','## Frozen heldout reward','',
        '|Arm / benchmark|Attempts|Predicted head|MMFF relief/heavy median|Surround relaxation RMS A|Valid / PB|Unique graphs|',
        '|---|---:|---:|---:|---:|---|---:|']
    for label,r in (('Matched native',n),('Frozen gradient',g)):
        lines.append(f"|{label}|{r['attempted']}|{r['all_head']:.6f}|{r['MMFF_relief_per_heavy_median']:.6f}|{r['surround_relax_RMS_A']:.6f}|{r['valid']} / {r['PB_fast_pass']}|{r['unique_graphs']}|")
    lines.append(f"|Historical discovery Steer|{steer['n']}|{steer['all_pic50_on_rescore_mean']:.6f}|{steer['all_mmff_relief_per_heavy_median']:.6f}|{steer['all_relax_rms_surround_A_mean']:.6f}|{steer['valid_connected']} / {steer['pb_fast_pass']}|{steer['unique_smiles']}|")
    lines+=['','Historical Steer is not a paired heldout control. All-head, unique-head and energy availability are separate in heldout_summary.json.',
        '','|Heldout batch|Head change|MMFF median change|MMFF p90 change|Surround RMS change A|Boundary shape change A|Valid / PB count change|',
        '|---:|---:|---:|---:|---:|---:|---|']
    for r in combined['paired_batch_changes']:
        lines.append(f"|{r['batch']}|{r['head_change']:.6f}|{r['MMFF_median_change']:.6f}|{r['MMFF_p90_change']:.6f}|{r['surround_RMS_change_A']:.6f}|{r['boundary_shape_change_A']:.6f}|{r['valid_count_change']} / {r['PB_count_change']}|")
    lines+=['','Positive head and negative physical changes are favorable. Three batch contrasts are descriptive; no candidate-level significance or biological affinity proof is claimed.']
    with (output/'summary.md').open('a',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    return output
