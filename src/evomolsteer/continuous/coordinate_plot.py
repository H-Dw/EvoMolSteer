"""Export selected continuous coordinate evidence without raw particle details."""
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial import Legendre
from ..io import read_json,write_json,digest


def plot(mining,features,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mining,out=Path(mining),Path(output)
    if out.with_suffix('.png').exists():raise FileExistsError(out)
    m=read_json(mining/'manifest.json');catalog=read_json(mining/'feature_catalog.json')['features']
    if not features or set(features)-set(catalog):raise ValueError('Explicit measured feature list required')
    d=pd.read_parquet(mining/'batch_coordinate_statistics.parquet',
        columns=['batch','time','feature','population_mean','selected_mean','selection_shift','lag_partial_gain_correlation'],
        filters=[('batch','in',m['splits']['discovery']),('feature','in',features)])
    fits=read_json(mining/'continuous_functions.json');times=np.asarray(m['times'])
    fig,axes=plt.subplots(len(features),3,figsize=(13,2.7*len(features)),squeeze=False,layout='constrained')
    for i,feature in enumerate(features):
        g=d[d.feature==feature];mean=g.groupby('time').mean(numeric_only=True).reindex(times)
        axes[i,0].plot(times,mean.population_mean,label='candidate',color='0.5')
        axes[i,0].plot(times,mean.selected_mean,label='selected',color='#0072B2')
        if feature in fits:
            f=fits[feature];curve=Legendre(f['legendre_coefficients'],domain=[f['time_start'],f['time_end']])
            axes[i,0].plot(times,curve(times),':',label='unconstrained fit',color='#D55E00')
            axes[i,1].plot(times,curve.deriv()(times),color='#D55E00',label='fitted d/dt')
        axes[i,1].plot(times,np.gradient(mean.selected_mean,times),color='#0072B2',alpha=.65,label='node d/dt')
        axes[i,2].plot(times,mean.selection_shift,color='#009E73',label='selected - candidate')
        axes[i,2].axhline(0,color='0.65',linewidth=.7)
        axes[i,0].set_title(feature,fontsize=9);axes[i,0].set_ylabel(catalog[feature]['unit'])
        axes[i,1].set_title('Temporal change, not coordinate force',fontsize=9)
        axes[i,2].set_title('Selection effect',fontsize=9)
        for ax in axes[i]:
            ax.set_xlabel('score time');ax.set_xlim(*m['window']);ax.legend(fontsize=7);ax.grid(alpha=.15)
    out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):fig.savefig(out.with_suffix(suffix),dpi=170)
    plt.close(fig)
    write_json(out.with_suffix('.json'),{'features':features,'window':m['window'],'discovery_batches':m['splits']['discovery'],
        'mining_manifest_sha256':digest(mining/'manifest.json'),
        'interpretation':'Equal-batch means; global fits may violate physical bounds. Raw exact-time nodes define inference references; temporal derivatives are not spatial forces.'})

