"""Plot saved evidence only; no inference or data exclusion."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/'evidence.json').read_text())
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
curves=[('chemical__raw999__window','Chemical + gradient, raw999','#c5393b'),
        ('chemical__raw1__window','Chemical + gradient, raw1','#2866ab'),
        ('gradient__raw999__window','Gradient only, raw999','#25846b')]
for arm,label,color in curves:
    rows=data['traces']['6pyy_H7S_A_402/'+arm]
    rows=[r for r in rows if r['step']>=10]
    axes[0].plot([r['step'] for r in rows],[r['rmsd_A'] for r in rows],label=label,color=color,lw=1.8)
    active=[r for r in rows if r['scale']!=0]
    axes[1].plot([r['step'] for r in active],[r['max_increment_A'] for r in active],color=color,lw=1.8)
for ax in axes:
    ax.set_yscale('log');ax.grid(True,which='major',alpha=.22)
    ax.axvspan(26,50,alpha=.08,color='#c5393b');ax.axvline(29,color='#c5393b',alpha=.4,ls=':')
    ax.axvline(51,color='#777777',alpha=.4,ls='--')
axes[0].set_ylabel('Condition-atom RMSD (Angstrom)');axes[0].legend(loc='upper right',fontsize=9)
axes[0].set_title('6pyy: paired seed 42, slot 2 — divergence occurs during integration')
axes[1].set_ylabel('Maximum atom guidance increment (Angstrom)')
axes[1].set_xlabel('Integration step (zero-based)')
axes[1].annotate('step 29: 30 A',xy=(29,29.98),xytext=(15,1500),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=9)
axes[1].annotate('step 50: 692,590 A',xy=(50,692589.75),xytext=(57,2e5),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=9)
fig.savefig(root/'outlier_trajectory.png',dpi=180)
fig.savefig(root/'outlier_trajectory.svg')
