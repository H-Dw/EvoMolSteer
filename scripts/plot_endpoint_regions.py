"""Standalone scientific plot of observed whole-window regional node effects."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evomolsteer.io import read_json,write_json,digest


def plot(prior_path,table_path,output,region=0):
    prior=read_json(prior_path);table=pd.read_parquet(table_path);times=np.asarray(prior['times'])
    prefix=f'landmark_{region:02d}_'
    fields=[prefix+c for c in ('soft_mass','local_dy','local_cyy')]
    if any(f not in prior['supported_node_functions'] for f in fields):raise ValueError('Selected evidence fields unavailable')
    arrays=[]
    for _,g in table.groupby('batch',sort=True):
        g=g.sort_values('score_time')
        if not np.array_equal(g.score_time.to_numpy(),times):raise ValueError('Exact whole-window support required')
        arrays.append(np.stack([(g['high__'+f]-g['low__'+f]).to_numpy()/prior['supported_node_functions'][f]['scale'] for f in fields],1))
    values=np.asarray(arrays);rng=np.random.default_rng(42)
    ci=np.quantile(values[rng.integers(0,len(values),(2000,len(values)))].mean(1),[.025,.975],axis=0)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharex=True)
    for j,(ax,f) in enumerate(zip(axes,fields)):
        mean=values[:,:,j].mean(0)
        ax.fill_between(times,ci[0,:,j],ci[1,:,j],color='#5587a3',alpha=.22,label='Pointwise batch bootstrap 95% CI')
        ax.plot(times,mean,color='#155777',lw=1.8,label='High-minus-low score effect')
        ax.axhline(0,color='#555555',lw=.8);ax.set_title(f.replace(prefix,f'Region {region}: '));ax.set_xlabel('Recorded selection score time')
        ax.set_xlim(times[0],times[-1]);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Equal-parent contrast / fixed feature scale (z)')
    axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.48),ncol=2,frameon=False,fontsize=8)
    fig.suptitle('Coordinate evidence changes within the learned selection window',fontsize=12)
    fig.subplots_adjust(top=.80,bottom=.31,wspace=.30)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);fig.savefig(output/'regional_node_effects.png',dpi=200);fig.savefig(output/'regional_node_effects.svg');plt.close(fig)
    write_json(output/'regional_plot.json',{'window':prior['window'],'times':prior['times'],'fields':fields,
        'input_hashes':{str(prior_path):digest(prior_path),str(table_path):digest(table_path)},'source_code_sha256':digest(__file__),
        'uncertainty_unit':'Equal generation batches after equal parent averaging, pointwise intervals only',
        'limitations':['Not simultaneous confidence bands or causal affinity mechanisms',
            'Endpoint forecasts at score nodes; final update ends at the support boundary',
            'Whole-window signs need not equal first/last-node signs']})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('prior','batch-nodes','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--region',type=int,default=0)
    a=p.parse_args();plot(a.prior,a.batch_nodes,a.output,a.region)
