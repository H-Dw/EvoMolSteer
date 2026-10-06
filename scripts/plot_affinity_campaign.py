"""Publication-style diagnostics from compact reports; no model inference."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from evomolsteer.io import read_json


def plot(evidence,output,round_number=None):
    evidence=Path(evidence);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[read_json(p) for p in sorted(evidence.glob('round_*.outcome.json'))]
    fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
    for split,marker in [('discovery','o'),('validation','s'),('heldout','D')]:
        selected=[r for r in rows if r['split']==split]
        if selected:ax.scatter([r['round'] for r in selected],[r['all_head_change_vs_native'] for r in selected],marker=marker,label=split)
    ax.axhline(0,color='0.5',lw=1);ax.set(xlabel='Round (adaptive selection precedes frozen checks)',ylabel='Paired mean predicted pIC50 change')
    ax.legend(frameon=False);fig.savefig(output/'affinity_rounds.png',dpi=180);plt.close(fig)
    if round_number is None:return
    folder=evidence/f'round_{round_number:02d}'/'local';head=pd.read_parquet(folder/'paired_head_window.parquet')
    dose=pd.read_csv(folder/'coordinate_dose_time.csv');report=read_json(folder/'head_window_response.json')
    end=next(r for r in rows if r['round']==round_number)['all_head_change_vs_native']
    fig,axes=plt.subplots(2,1,figsize=(8,5.5),sharex=True,layout='constrained')
    for batch,part in head.groupby('batch'):
        axes[0].plot(part.score_time,part.mean_delta,label=f'batch {batch}: mean change')
        axes[0].plot(part.score_time,part.mean_absolute_delta,ls=':',alpha=.6,label=f'batch {batch}: mean absolute change')
    for batch,part in dose[(dose.arm=='gradient')&dose.active].groupby('batch'):
        axes[1].plot(part.score_time,part.injection_rms_A,label=f'batch {batch}')
    axes[0].axhline(0,color='0.5',lw=1);axes[0].set_ylabel('Recorded head change (pIC50)');axes[0].legend(frameon=False,ncol=2,fontsize=8)
    axes[1].set(xlim=report['window'],xlabel='Score time (last label precedes last injection)',ylabel='Delivered coordinate RMS (A / step)')
    fig.suptitle(f'Round {round_number}: final mean rescore change {end:+.4f}; batch lines are descriptive')
    fig.savefig(output/f'round_{round_number:02d}_window_response.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);p.add_argument('--round',type=int)
    a=p.parse_args();plot(a.evidence,a.output,a.round)
