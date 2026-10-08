"""Export compact screening and independent-confirmation figures."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evomolsteer.io import read_json


def plot(directory):
    root=Path(directory);d=pd.read_csv(root/'rounds.csv')
    final=read_json(root/'final_comparison.json');fig,axes=plt.subplots(2,2,figsize=(11,7.5))
    ax=axes[0,0]
    groups=[('Earlier implementation',[2,3,4,5,7],'#999999'),
            ('Implementation control',[6],'#d08716'),
            ('Corrected screening',list(range(8,17)),'#266ba9'),
            ('Frozen confirmation',[18,20],'#178363')]
    for label,ids,color in groups:
        rows=d[d['round'].isin(ids)]
        ax.scatter(rows['round'],rows.vs_R26,label=label,color=color,s=40,zorder=3)
    ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Round',ylabel='Paired mean pIC50 change vs R26')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[0,1]
    for label,ids,color in groups:
        rows=d[d['round'].isin(ids)]
        ax.scatter(rows['round'],rows.strain_p90,color=color,s=40)
    ax.axhline(d[d['round'].eq(1)].strain_p90.iloc[0],color='#777',linestyle='--',label='R26 screening')
    ax.axhline(final['historical_Steer']['all']['strain_p90_per_heavy'],color='#b24d45',linestyle=':',label='Historical Steer')
    ax.set(xlabel='Round',ylabel='MMFF relief P90 per heavy atom (kcal/mol)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    labels=['Native','R26','Frozen candidate'];keys=['native','R26','candidate']
    colors=['#999999','#266ba9','#178363'];values=final['confirmation_results'];x=np.arange(3)
    ax=axes[1,0]
    ax.bar(x,[values[k]['all_mean_pic50'] for k in keys],color=colors,width=.6)
    low=min(values[k]['all_mean_pic50'] for k in keys)-.08
    high=max(values[k]['all_mean_pic50'] for k in keys)+.08
    ax.set(xticks=x,xticklabels=labels,ylim=(low,high),ylabel='All-attempt mean pIC50',title='Independent confirmation: n=200 per arm')
    for i,k in enumerate(keys):
        value=values[k]['all_mean_pic50'];ax.text(i,value+.005,f'{value:.4f}',ha='center',fontsize=9)
    ax=axes[1,1]
    ax.bar(x-.18,[values[k]['strain_median_per_heavy'] for k in keys],width=.36,color=colors,label='Median')
    ax.bar(x+.18,[values[k]['strain_p90_per_heavy'] for k in keys],width=.36,color=colors,alpha=.4,label='P90')
    ax.set(xticks=x,xticklabels=labels,ylabel='MMFF relief per heavy atom (kcal/mol)',title='Secondary structural stability')
    ax.legend(fontsize=8)
    fig.suptitle('R26 selection-path exploration: screening separated from confirmation',fontsize=12)
    fig.text(.5,.015,'Historical Steer is unpaired and uses different compute; earlier implementations are excluded from direct adoption.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.95]);target=root/'rounds_analysis.png';fig.savefig(target,dpi=180);plt.close(fig)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);print(plot(p.parse_args().reports))
