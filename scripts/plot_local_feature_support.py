"""Plot the declared reference support and its saved endpoint trajectory."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evomolsteer.io import read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('program','teacher','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();program=read_json(a.program);teacher=read_json(a.teacher)
    spec=program['mode_specs'][0];mode=teacher['modes'][spec['template_id']];slots=spec['atom_slots']
    xyz=np.asarray(mode['terminal_coords_A']);support=xyz[slots]
    fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d');bx=fig.add_subplot(122,projection='3d')
    ax.scatter(*xyz.T,c='#b9c4cd',s=25,label='Other atoms: context only')
    ax.scatter(*support.T,c='#c33754',s=70,label=f'Declared support: {len(slots)}/{len(xyz)} atoms')
    for i,neighbors in mode['full_adjacency'].items():
        for j in neighbors:
            if int(i)<int(j):
                edge=xyz[[int(i),int(j)]];ax.plot(*edge.T,color='#9aa6ae',lw=1)
    for i,point in zip(slots,support):ax.text(*(point+np.array([.06,0,.08])),str(i),fontsize=9,color='#92263b')
    for site in spec['pocket_sites']:
        q=np.asarray(site['coords_A']);ax.scatter(*q,c='#4a7790',s=32,marker='x')
        ax.text(*q,site.get('id','pocket background'),fontsize=7,color='#3c6378')
    ax.set_title('Terminal teacher: support declaration')
    times=np.asarray(mode['path']['times']);values=np.asarray(mode['path']['endpoint_A'])
    keep=(times>=program['window'][0]-1e-6)&(times<program['window'][1]-1e-6)
    for color,i in zip(plt.cm.viridis(np.linspace(.1,.9,len(slots))),slots):
        q=values[keep,i];bx.plot(*q.T,color=color,lw=1.5,label='Teacher slot '+str(i))
        bx.scatter(*q[0],color=color,marker='x',s=35);bx.scatter(*q[-1],color=color,marker='o',s=22)
    bx.set_title('Saved teacher endpoint predictions\nx: first guided time; dot: last guided time')
    for axis in (ax,bx):
        axis.set_xlabel('x (Å)');axis.set_ylabel('y (Å)');axis.set_zlabel('z (Å)');axis.view_init(25,-65)
        axis.legend(fontsize=7,loc='upper left')
    context=np.concatenate([xyz,np.asarray([s['coords_A'] for s in spec['pocket_sites']]).reshape(-1,3)])
    ax.set_box_aspect(np.maximum(np.ptp(context,axis=0),.1))
    bx.set_box_aspect(np.maximum(np.ptp(values[keep][:,slots].reshape(-1,3),axis=0),.1))
    fig.suptitle('Reference points are local; student correspondence is recomputed during inference',fontsize=11)
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);plt.close(fig)


if __name__=='__main__':main()
