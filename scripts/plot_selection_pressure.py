"""Standalone compact lineage-pressure figure from retained summaries only."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder,output):
    folder,output=Path(folder),Path(output);events=pd.read_parquet(folder/'events.parquet');neutral=pd.read_parquet(folder/'neutral_copying.parquet')
    mean=events.groupby('time').mean(numeric_only=True)
    fig,ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for batch,rows in events.groupby('batch'):
        ax[0].plot(rows.time,rows.surviving_roots,color='#2563eb',alpha=.18,lw=.8)
    ax[0].plot(mean.index,mean.surviving_roots,color='#2563eb',label='Observed batch mean')
    ax[0].plot(neutral.score_time,neutral.neutral_mean_roots,color='#dc2626',label='Exact neutral-copy reference')
    ax[0].fill_between(neutral.score_time,neutral.neutral_panel_low95,neutral.neutral_panel_high95,color='#dc2626',alpha=.12,label='Neutral panel 95% range')
    ax[0].set(xlabel='Selection score time',ylabel='Surviving source families',title='Accumulated ancestry loss')
    ax[0].legend(fontsize=7)
    ax[1].plot(mean.index,mean.expected_ess_fraction,label='Probability ESS / population',color='#059669')
    ax[1].plot(mean.index,mean.realized_ess_fraction,label='Actual-copy ESS / population',color='#7c3aed')
    ax[1].set(xlabel='Selection score time',ylabel='Fraction',ylim=(0,1.02),title='Single-event pressure and copying')
    ax[1].legend(fontsize=7)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();plot(a.input,a.output)
