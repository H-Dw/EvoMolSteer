"""Build local tables/figures for the capped sequential window experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from evomolsteer.io import read_json,write_json,digest


def report(results,output):
    results=Path(results);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    paths=sorted(results.glob('round_*/report.json'));reports=[read_json(p) for p in paths]
    baseline=reports[0]['results']['unguided'];rows=[]
    for r in reports:
        row={'campaign':r['campaign'],'seed':r['seed'],**r['results']['gradient']}
        row['improvement_vs_adaptation_native_percent']=100*(1-row['symmetric_shape_A']/baseline['symmetric_shape_A']) if r['seed']==reports[0]['seed'] else np.nan
        rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(output/'round_metrics.csv',index=False)
    confirmation=None
    for r in reports:
        if r['seed']==reports[0]['seed']:continue
        native=sorted([v for v in r['batch_results'] if v['arm']=='unguided'],key=lambda x:x['batch'])
        guided=sorted([v for v in r['batch_results'] if v['arm']=='gradient'],key=lambda x:x['batch'])
        if len(native)!=len(guided) or len(native)<2:raise ValueError('Complete paired confirmation required')
        for x,y in zip(native,guided):
            if x['batch']!=y['batch']:raise ValueError('Mismatched confirmation batches')
            k=str(x['batch'])
            if r['initial_state_signatures']['unguided/'+k]!=r['initial_state_signatures']['gradient/'+k]:raise ValueError('Unpaired confirmation prior')
        confirmation={'campaign':r['campaign'],'seed':r['seed'],'n_per_arm':r['results']['gradient']['n'],
                      'batches':len(native),'not_used_for_retuning':True,'paired_metrics':{}}
        for key in ['symmetric_shape_A','generated_to_reference_A','reference_to_generated_A',
                    'atom_mismatch_geometry_assignment','bonded_union_mismatch','atom_fraction_L1','bond_fraction_L1']:
            x=np.array([v[key] for v in native]);y=np.array([v[key] for v in guided]);delta=y-x
            se=delta.std(ddof=1)/np.sqrt(len(delta));margin=student_t.ppf(.975,len(delta)-1)*se
            confirmation['paired_metrics'][key]={'native_mean':float(x.mean()),'gradient_mean':float(y.mean()),
                'relative_improvement_percent':float(100*(1-y.mean()/x.mean())),
                'mean_delta':float(delta.mean()),'batch_deltas':delta.tolist(),
                'paired_t_95CI_delta':[float(delta.mean()-margin),float(delta.mean()+margin)]}
        write_json(output/'frozen_confirmation.json',confirmation)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    adaptation=df[df.seed==reports[0]['seed']]
    axes[0].plot(range(1,len(adaptation)+1),adaptation.symmetric_shape_A,'o-',label='Guided')
    axes[0].axhline(baseline['symmetric_shape_A'],color='gray',linestyle='--',label='Matched native')
    axes[0].set(xlabel='Optimization round',ylabel='Symmetric assignment RMSD (Å)',title='Adaptation: exact learned-window endpoint')
    axes[0].legend(frameon=False)
    if confirmation:
        r=reports[-1];native=[v['symmetric_shape_A'] for v in r['batch_results'] if v['arm']=='unguided'];guided=[v['symmetric_shape_A'] for v in r['batch_results'] if v['arm']=='gradient']
        for i,(x,y) in enumerate(zip(native,guided)):axes[1].plot([0,1],[x,y],'o-',alpha=.7,label=f'Batch {i+1}')
        axes[1].set_xticks([0,1],['Native','Frozen guided']);axes[1].set(ylabel='Symmetric assignment RMSD (Å)',title='Independent seeds: paired batch results')
        axes[1].legend(frameon=False)
    else:axes[1].axis('off')
    fig.savefig(output/'window_fit_comparison.png',dpi=200);plt.close(fig)
    summary={'rounds':len(reports),'generated_candidates_including_controls':sum(v['n'] for r in reports for v in r['results'].values()),
             'reference_time':reports[0]['exact_time'],'all_reports':[{'path':str(p),'sha256':digest(p)} for p in paths],
             'confirmation':confirmation,'plot':'window_fit_comparison.png'}
    write_json(output/'experiment_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=report(a.results,a.output);print(json.dumps({'rounds':r['rounds'],'generated_candidates_including_controls':r['generated_candidates_including_controls'],'confirmation':r['confirmation']},indent=2))
