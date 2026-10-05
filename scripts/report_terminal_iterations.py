"""Plot continuous-window diagnostics and independent terminal quality axes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evomolsteer.io import read_json, digest
from evomolsteer.generation.prototypes import write_json


def report(experiment, output):
    root=Path(experiment);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    ledger=read_json(root/'round_history.json');items=[
        ('SMC',root/'reference','single'), ('Native',root/'round_01/global','unguided'),
        ('Global MMD',root/'round_01/global','gradient')]
    for row in ledger['rounds']:
        p=root/f"round_{row['round']:02d}"/'local'
        if row['status']=='complete' and (p/'terminal_report.json').is_file():items.append((f"Local R{row['round']}",p,'gradient'))
    rows=[];sources=[]
    for label,p,arm in items:
        r=read_json(p/'terminal_report.json');s=r['results'][arm]
        row={'arm':label,'n':s['n'],'valid_n':s['valid_connected'],'pb_fast_n':s['pb_fast_pass'],
             'unique_n':s['unique_smiles'],'unique_scaffolds':s['unique_scaffolds'],'unique_yield':s['unique_yield'],
             'valid_head_mean':s['valid_pic50_on_rescore_mean'],'unique_first_pose_head_mean':s['unique_valid_pic50_on_rescore_mean'],
             'mmff_relief_per_heavy_median':s['valid_mmff_relief_per_heavy_median'],
             'energy_converged_n':s['valid_mmff_relief_per_heavy_n'],
             'surround_relax_rms_A':s['valid_relax_rms_surround_A_mean'],
             'final_region_q_r2':s['valid_terminal_patch_q_over_r2_mean']}
        if (p/'execution_report.json').exists():
            e=read_json(p/'execution_report.json');d=[v['mean_deficit'] for v in e['endpoint_diagnostics_at_window_end'] if v['arm']==arm]
            row['window_endpoint_deficit']=float(np.mean(d))
            batch=[v for v in e['batch_results'] if v['arm']==arm]
            row['cumulative_injection_rms_A']=float(np.mean([v['mean_cumulative_injection_rms_A'] for v in batch]))
        rows.append(row);sources.append({'path':str(p/'terminal_report.json'),'sha256':digest(p/'terminal_report.json')})
    data=pd.DataFrame(rows);data.to_csv(out/'iteration_quality_axes.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    # This figure uses the whole selection window as one continuous domain.
    for label,p,arm in items[1:]:
        d=pd.read_csv(p/'regional_time_metrics.csv');d=d[d.arm==arm]
        window=read_json(p/'execution_report.json')['window'];d=d[d.time.between(*window)]
        curve=d.groupby('time').mean(numeric_only=True)
        axs[0,0].plot(curve.index,curve.mean_deficit,label=label)
    d=pd.read_csv(root/'reference/original_regional_window.csv')
    d=d[(d.arm=='single')&(d.mass=='selection_probability')].groupby('time').mean(numeric_only=True)
    axs[0,0].plot(d.index,d.mean_deficit,'k--',label='Original SMC probability mass')
    axs[0,0].set(xlabel='Inference score time',ylabel='Mean regional deficit',title='Continuous learning window')
    axs[0,0].legend(fontsize=8,frameon=False)
    x=np.arange(len(data));labels=data.arm
    axs[0,1].plot(x,data.valid_n/data.n,'o-',label='Connected valid');axs[0,1].plot(x,data.pb_fast_n/data.n,'s--',label='PB dock_fast')
    axs[0,1].set(ylabel='Passing / all candidates',ylim=(.8,1.01),title='All-slot quality yield');axs[0,1].legend(frameon=False,fontsize=8)
    axs[0,2].bar(x,data.unique_yield,color='#4477aa');axs[0,2].set(ylabel='Unique graph / all candidates',ylim=(0,1),title='Graph freedom proxy')
    axs[1,0].plot(x,data.valid_head_mean,'o-',label='All valid poses');axs[1,0].plot(x,data.unique_first_pose_head_mean,'s--',label='Unique first pose')
    axs[1,0].set(ylabel='FLOWR target head pIC50',title='Prediction, not measured binding');axs[1,0].legend(frameon=False,fontsize=8)
    axs[1,1].plot(x,data.mmff_relief_per_heavy_median,'o-')
    axs[1,1].set(ylabel='Median relief / heavy atom (kcal/mol)',title='Same-graph local strain relief')
    axs[1,2].plot(x,data.surround_relax_rms_A,'s-')
    axs[1,2].set(ylabel='Surround relaxation RMS (A)',title='Surrounding geometry diagnostic')
    for ax in [axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[1,2]]:ax.set_xticks(x,labels,rotation=25,ha='right')
    fig.savefig(out/'terminal_comparison.png',dpi=200);fig.savefig(out/'terminal_comparison.pdf');plt.close(fig)
    write_json(out/'report_manifest.json',{'rounds_completed':len(ledger['rounds']),'maximum_rounds':5,'sources':sources,
        'figure_scope':'One continuous learning window; no subinterval bins. Final evaluations are separate validation outcomes.',
        'limits':'Fixed seed development data; no population CI, no true binding-energy or affinity claim. Energy and geometry have different units and must not be combined into a score.'})
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(report(a.experiment,a.output).to_string(index=False))
