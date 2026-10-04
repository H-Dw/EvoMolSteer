"""Scientific reference/trajectory/control plots; vector and raster exports."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from evomolsteer.io import read_json

p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--evaluation');p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
packet=read_json(a.reference);t=np.array(packet['times']);centers=np.array(packet['centers_A']);knots=packet['knots']
grid=np.linspace(0,1,401);fitted=PchipInterpolator(t[knots],centers[knots],axis=0)(np.minimum(grid,.5))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def save(fig,name):
    for extension in ['png','pdf','svg']:
        path=out/(name+'.'+extension)
        fig.savefig(path,dpi=180,bbox_inches='tight')
        if extension=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharex=True)
for k,ax in enumerate(axes):
    for batch in range(centers.shape[1]):ax.plot(grid,fitted[:,batch,k],lw=.8,alpha=.55)
    ax.plot(t,centers[:,:,k].mean(1),color='black',lw=2,label='Mean of 14 batch representatives')
    ax.scatter(t[knots],centers[knots,:,k].mean(1),s=12,color='black',zorder=5,label='Adaptive interpolation knots')
    ax.axvspan(.5,1,color='gray',alpha=.12,label='Frozen-reference continuation')
    ax.set(xlabel='Generation time',ylabel='Typed soft-min distance (Å)',title=packet['features'][k].split('::')[0])
axes[1].legend(fontsize=8,loc='upper right');fig.suptitle('Multistage retained-ancestor references: discovery data only')
fig.tight_layout();save(fig,'multistage_references')
if not a.evaluation:raise SystemExit(0)
ev=Path(a.evaluation);curves=pd.read_parquet(ev/'feature_curve_batches.parquet');control=pd.read_parquet(ev/'control_particle_steps.parquet')
arms=['unguided','single','gradient_legacy','static_full','multistage_window','multistage_full']
colors=dict(zip(arms,['#777777','#111111','#b5838d','#bc8f19','#418bb8','#db5724']))
fig,axes=plt.subplots(2,2,figsize=(11,7.2),sharex=True)
features=packet['features']+['radius_gyration_A','ck2:A:ASN117::centroid_z']
for ax,feature in zip(axes.flat,features):
    for arm in arms:
        g=curves[curves.arm.eq(arm)].groupby('time')[feature].agg(['mean','sem'])
        if g.empty:continue
        ax.plot(g.index,g['mean'],label=arm,color=colors[arm],lw=1.5)
        ax.fill_between(g.index,g['mean']-g['sem'],g['mean']+g['sem'],color=colors[arm],alpha=.10)
    ax.axvline(.5,color='gray',ls=':',lw=1)
    ax.set(title=feature.replace('ck2:A:','').replace('::',' / '),ylabel='Å',xlabel='Generation time')
axes[0,0].legend(fontsize=7);fig.suptitle('Matched new batches: mean ± batch SEM (descriptive)')
fig.tight_layout();save(fig,'multistage_trajectory_comparison')
fig,axes=plt.subplots(1,2,figsize=(11,3.8))
for arm in arms:
    g=control[control.arm.eq(arm)]
    if not len(g):continue
    means=g.groupby('time').injection_max_atom_A.mean()
    axes[0].plot(means.index,means.values,label=arm,color=colors[arm])
    g=g[g.injection_rms_A.notna()]
    if len(g):
        mean=g.groupby('time').cumulative_injection_rms_A.mean()
        axes[1].plot(mean.index,mean.values,label=arm,color=colors[arm])
for ax in axes:ax.axvline(.5,color='gray',ls=':');ax.set_xlabel('Generation time')
axes[0].set_ylabel('Mean maximum atom correction per step (Å)');axes[0].legend(fontsize=7)
axes[1].set_ylabel('Mean sum of RMS corrections (Å)')
fig.tight_layout();save(fig,'multistage_actual_control')
distance_path=ev/'distribution_distance_batches.parquet'
if distance_path.exists():
    distance=pd.read_parquet(distance_path)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharex=True)
    for ax,metric,label in zip(axes,
        ['energy_squared_to_smc','nearest_smc_pose_chamfer_A'],
        ['Whitened two-feature energy statistic','Nearest SMC heavy-atom Chamfer (Å)']):
        for arm in arms:
            g=distance[distance.arm.eq(arm)].groupby('time')[metric].agg(['mean','sem'])
            if g.empty:continue
            ax.plot(g.index,g['mean'],label=arm,color=colors[arm],lw=1.5)
            ax.fill_between(g.index,g['mean']-g['sem'],g['mean']+g['sem'],color=colors[arm],alpha=.10)
        ax.axvline(.5,color='gray',ls=':');ax.set(xlabel='Generation time',ylabel=label)
    axes[0].legend(fontsize=7)
    fig.suptitle('Distance to same-batch SMC samples (lower is closer; mean ± batch SEM)')
    fig.tight_layout();save(fig,'multistage_smc_imitation')
