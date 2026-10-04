"""Standalone publication-style descriptive plots; no refitting or new inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--comparison');p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.analysis);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    d=pd.read_csv(root/'discovery/trends/effects.csv')
    d=d[(d.arm=='single')&(d.representation=='predicted_endpoint')&(d.contrast=='expected_selection_shift')]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for feature,label,color in [('ck2:A:GLU114::distance_softmin','GLU114','#737373'),('ck2:A:HIS115::distance_softmin','HIS115','#e69f00'),
                                ('ck2:A:VAL116::distance_softmin','VAL116','#0072b2'),('ck2:A:ASN117::distance_softmin','ASN117','#009e73')]:
        g=d[d.feature==feature].sort_values('stage');t=g.stage*.1+.05
        axes[0].plot(t,g.effect,'o-',label=label,color=color,lw=1.6)
        axes[0].fill_between(t,g.ci_low,g.ci_high,color=color,alpha=.09)
    g=d[d.feature=='ligand::radius_gyration'].sort_values('stage');t=g.stage*.1+.05
    axes[1].errorbar(t,g.effect,yerr=np.array([g.effect-g.ci_low,g.ci_high-g.effect]),fmt='o-',color='#cc79a7',capsize=3)
    for ax in axes:
        ax.axhline(0,c='#555555',lw=.8);ax.set_xlabel('Score-time stage midpoint');ax.set_ylabel('Expected selection shift (Å)')
        ax.set_xticks([.05,.15,.25,.35,.45]);ax.grid(alpha=.15)
    axes[0].set_title('Hinge-proximal endpoint geometry');axes[0].legend(frameon=False)
    axes[1].set_title('Endpoint radius of gyration')
    fig.suptitle('CK2 single-target discovery: 14 independent batches\nBands/bars: batch bootstrap 95% intervals; selection association only',fontsize=12)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'discovery_regions.{ext}',dpi=180)
    plt.close(fig)
    if a.comparison:
        comp=Path(a.comparison);pairs=pd.read_csv(comp/'paired_outcomes.csv');geom=pd.read_csv(comp/'paired_geometry.csv')
        labels={'single':'Single SMC','gradient_region':'Regional gradient','gradient_region_compact':'Regional + compact'}
        colors={'single':'#e69f00','gradient_region':'#0072b2','gradient_region_compact':'#009e73'}
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        q=pairs[pairs.metric=='ck2_rescore_all'].copy()
        for i,row in enumerate(q.itertuples()):
            axes[0].errorbar(row.mean_difference,i,xerr=[[row.mean_difference-row.t_ci_low],[row.t_ci_high-row.mean_difference]],
                            fmt='o',capsize=4,color=colors[row.arm])
        axes[0].set_yticks(range(len(q)),[labels[s] for s in q.arm]);axes[0].axvline(0,c='#777777',lw=.8)
        axes[0].set_xlabel('Δ final CK2 pIC50 rescore');axes[0].set_title('Paired-batch mean and 95% t CI')
        for ax,term,title in zip(axes[1:],['region','compact'],['VAL116 soft-min distance','Radius of gyration']):
            for arm,g in geom[(geom.term_id==term)&(geom.step<=50)].groupby('arm'):
                ax.plot(g.score_time,g.mean_difference,label=labels[arm],color=colors[arm],lw=1.4)
            ax.axhline(0,c='#777777',lw=.8);ax.set_xlabel('Score time');ax.set_ylabel('Δ endpoint observable (Å)');ax.set_title(title)
        axes[2].legend(frameon=False,fontsize=8)
        fig.suptitle('Frozen reward comparison vs unguided: 4 independent batches × 50 candidates\nModel affinity is not experimental binding; geometry evaluated separately',fontsize=12)
        for ext in ['png','svg','pdf']:fig.savefig(out/f'guidance_comparison.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':main()
