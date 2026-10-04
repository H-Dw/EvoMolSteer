"""Plot whole-window feature contrasts and lineage retention, without bins."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--features',nargs='+',default=['ck2:A:VAL116::distance_softmin','ligand::radius_gyration',
        'ligand::atom_fraction_N','ligand::bond_label_4_fraction'])
    a=p.parse_args();root=Path(a.analysis);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    dest=root/'discovery/continuous'
    observed=pd.read_parquet(dest/'trends/point_curves.parquet')
    fitted=pd.read_parquet(dest/'trends/fitted_curves.parquet')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(len(a.features),2,figsize=(10,2.4*len(a.features)),layout='constrained',squeeze=False)
    for row,feature in enumerate(a.features):
        for col,contrast in enumerate(['expected_selection_shift','window_ancestor_shift']):
            ax=axes[row,col]
            for frame,kind in [(observed,'observed'),(fitted,'fit')]:
                g=frame[(frame.feature==feature)&(frame.contrast==contrast)&(frame.representation=='predicted_endpoint')].sort_values('time')
                if not len(g):continue
                if kind=='observed':ax.plot(g.time,g['mean'],'.',color='#777777',ms=3,label='Event means')
                else:
                    ax.plot(g.time,g.fitted,color='#0072b2',label='Global fit')
                    ax.fill_between(g.time,g.simultaneous_low,g.simultaneous_high,color='#0072b2',alpha=.16,label='95% whole-curve band')
            ax.axhline(0,color='#555555',lw=.7);ax.set_xlabel('Score time');ax.set_title(feature+'\n'+contrast)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Complete selection window: discovery batches only\nRetained ancestry is retrospective; fitted derivatives are not spatial gradients',fontsize=11)
    save(fig,out/'continuous_feature_curves');plt.close(fig)
    lineage=pd.read_csv(dest/'lineage/diagnostics.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax,column,label in [(axes[0],'retained_ancestor_candidates','Candidates with window-end copies'),
                            (axes[1],'current_root_count','Distinct initial roots in current population')]:
        for _,g in lineage.groupby('batch'):ax.plot(g.time,g[column],color='#009e73',alpha=.2,lw=.8)
        g=lineage.groupby('time')[column].mean();ax.plot(g.index,g.values,color='#0072b2',lw=2)
        ax.set_xlabel('Score time');ax.set_ylabel(label)
    fig.suptitle('Branch retention and root collapse; thin curves = independent batches')
    save(fig,out/'continuous_lineage_retention');plt.close(fig)


def save(fig,path):
    for ext in ['png','pdf','svg']:
        target=path.with_suffix('.'+ext);fig.savefig(target,dpi=180)
        if ext=='svg':target.write_bytes(('\n'.join(s.rstrip() for s in target.read_text(encoding='utf-8').splitlines())+'\n').encode())


if __name__=='__main__':main()
