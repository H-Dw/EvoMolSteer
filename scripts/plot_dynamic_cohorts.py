"""Standalone scientific plot of whole-window adaptive cohorts and region contrasts."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='evomolsteer-dynamic-cohorts-v1'
import matplotlib.pyplot as plt
import pandas as pd


def plot(evidence,output):
    evidence,output=Path(evidence),Path(output);output.mkdir(parents=True,exist_ok=True)
    c=pd.read_csv(evidence/'event_cohorts.csv');g=c[c.identifiable].groupby('time').mean(numeric_only=True)
    effects=pd.read_csv(evidence/'whole_window_effects.csv');trends=pd.read_csv(evidence/'feature_trends.csv')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for key,label in [('positive_score_mean','Positive'),('threshold','Adaptive threshold'),('negative_score_mean','Lower-score control')]:
        axs[0,0].plot(g.index,g[key],label=label,lw=1.8)
    axs[0,0].set_ylabel('Recorded joint-head pIC50');axs[0,0].legend(frameon=False,fontsize=8)
    for key,label in [('positive_n','Positive'),('negative_n','Lower-score'),('ambiguous_n','Ambiguous')]:
        axs[0,1].plot(g.index,g[key],label=label)
    axs[0,1].set_ylabel('Mean candidate count per batch');axs[0,1].legend(frameon=False,fontsize=8)
    region_fields=['landmark_00_softmin','landmark_03_softmin','landmark_04_softmin','landmark_12_occupancy3']
    for feature in region_fields:
        d=trends[trends.feature==feature];line,=axs[1,0].plot(d.time,d.contrast_z,label=feature.replace('landmark_','L'))
        axs[1,0].plot(d.time,d.fitted_contrast_z,ls='--',alpha=.5,color=line.get_color())
    axs[1,0].axhline(0,c='grey',lw=.7);axs[1,0].set_ylabel('Positive - lower-score contrast (z)')
    axs[1,0].legend(frameon=False,fontsize=7)
    axs[1,1].plot(g.index,g.root_n,label='Mean ancestral families')
    axs[1,1].plot(g.index,g.positive_effective_n,label='Positive weight ESS')
    axs[1,1].plot(g.index,g.negative_effective_n,label='Lower-score weight ESS')
    axs[1,1].set_ylabel('Coverage diagnostics, not independent n');axs[1,1].legend(frameon=False,fontsize=8)
    for ax in axs.flat:ax.set_xlabel('Pre-selection score time');ax.spines[['right','top']].set_visible(False)
    fig.suptitle('Dynamic cohorts over the complete learned window (14 independent batches)')
    fig.savefig(output/'dynamic_cohorts.png',dpi=180)
    svg=output/'dynamic_cohorts.svg';fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plot(a.evidence,a.output)
