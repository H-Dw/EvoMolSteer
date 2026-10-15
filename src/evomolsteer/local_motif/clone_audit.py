"""Check whether coordinate+element clone keys omit distinct bond/charge states."""
import argparse
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.trajectory import TrajectoryPackage


def partitions(xyz,atom,bonds,charges):
    n=len(xyz);base=np.concatenate([xyz.reshape(n,-1),atom.reshape(n,-1)],axis=1)
    full=np.concatenate([base,bonds.reshape(n,-1),charges.reshape(n,-1)],axis=1)
    _,ib=np.unique(base,axis=0,return_inverse=True);_,iff=np.unique(full,axis=0,return_inverse=True)
    return int(len(np.unique(ib))),int(len(np.unique(iff)))


def build(repo,config):
    repo=Path(repo);m=read_json(repo/config);study=repo/m['campaign_study'];templates=read_json(study/'templates.json')
    root=repo/m['compact_dataset']/'results'/m['campaign']/m['arm'];rows=[];sources={}
    for batch in sorted({t['batch'] for t in templates.values()}):
        path=root/f'batch_{batch:03d}'/'trajectory.h5';sources[str(path.relative_to(repo))]=digest(path)
        with TrajectoryPackage(path) as tr:
            times=tr.read('score_time')[:,0]
            for step,t in enumerate(times):
                if not m['learning_window'][0]-2e-6<=t<=m['learning_window'][1]+2e-6:continue
                a,b=partitions(tr.read('current_coords',step),tr.read('current_atomics',step),tr.read('current_bonds',step),tr.read('current_charges',step))
                rows.append({'batch':batch,'step':step,'time':float(t),'coordinate_element_groups':a,'full_structure_groups':b,'difference':b-a})
    result={'checked_time_slices':len(rows),'different_slices':sum(r['difference']!=0 for r in rows),'rows':rows,'sources':sources,
        'scope':'Empirical equality of clone partitions on these archived snapshots only; not a general guarantee that bonds/charges may be omitted.'}
    write_json(study/'clone_partition_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);a=p.parse_args()
    r=build(a.repo,a.config);print({k:r[k] for k in ['checked_time_slices','different_slices']})
