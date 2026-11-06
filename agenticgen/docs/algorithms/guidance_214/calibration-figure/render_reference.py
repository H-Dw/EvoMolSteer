"""Render measured input coordinates as a reference for the image model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--outdir',required=True);args=ap.parse_args()
out=Path(args.outdir);d=json.loads((out/'example_data.json').read_text(encoding='utf-8'))
xyz=np.asarray(d['coordinates_angstrom']);selected=d['condition_atom_indices_zero_based']
centered=xyz-xyz.mean(0);lookup={a['id']:i for i,a in enumerate(d['atoms'])}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none'})
fig=plt.figure(figsize=(14,4.6),facecolor='white',layout='constrained')
ax=fig.add_subplot(131);ax.imshow(plt.imread(out/'atom_graph.png'));ax.axis('off')
ax.set_title('Atomic graph',fontweight='bold')
ax=fig.add_subplot(132,projection='3d')
for bond in d['bonds']:
    points=centered[[lookup[i] for i in bond['atom_ids']]]
    ax.plot(*points.T,color='#59616b',lw=3,solid_capstyle='round',zorder=1)
colors=['#d65c52' if a['element']=='O' else '#626c76' for a in d['atoms']]
ax.scatter(*centered.T,s=115,c=colors,edgecolors='white',linewidths=.5,depthshade=False)
ax.scatter(*centered[selected].T,s=235,facecolors='none',edgecolors='#228d87',linewidths=1.7,depthshade=False)
ax.view_init(elev=22,azim=-50);ax.set_box_aspect((1,1,1));ax.axis('off')
lim=np.max(np.ptp(centered,axis=0))*.58
ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim));ax.set_title('Reference 3D coordinates',fontweight='bold')
ax=fig.add_subplot(133,projection='3d')
ax.scatter(*centered[selected].T,s=100,c='#228d87',edgecolors='white',depthshade=False)
for k,i in enumerate(selected):ax.text(*(centered[i]+.12),str(k+1),fontsize=10)
ax.view_init(elev=22,azim=-50);ax.set_box_aspect((1,1,1));ax.axis('off')
ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim));ax.set_title('9 condition atoms',fontweight='bold')
fig.suptitle('6PYY / H7S: one molecular input, consistent atom identity',fontsize=15,fontweight='bold')
fig.savefig(out/'molecular_reference.png',dpi=220,facecolor='white')
plt.close(fig)
print('Rendered molecular_reference.png from the actual atom graph and coordinates.')
