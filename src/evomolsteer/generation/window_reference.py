"""Local extraction of exact-time actual-state SMC representatives."""
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from ..io import read_json, digest, write_json
from ..trajectory_source import open_trajectory


def load_reference(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def reference_node_time(reference, score_time, state_time):
    """Keep forecast labels on their pre-native clock, not terminal-state time."""
    representation=reference.get('control_representation','current')
    if reference.get('schema_version')=='affinity-endpoint-library-1.0':
        if representation not in ('proposal','predicted_endpoint'):
            raise ValueError('Endpoint libraries require an explicit pre-native forecast clock')
        return score_time
    if representation=='proposal':return score_time
    if representation=='current':return state_time
    raise ValueError('Unknown reference time representation')


def descriptor(x, atoms, bonds, vocabulary_size):
    center=x.mean(1);centered=x-center[:,None]
    covariance=np.einsum('bni,bnj->bij',centered,centered)/x.shape[1]
    eig=np.linalg.eigvalsh(covariance)
    fraction=np.stack([(atoms==i).mean(1) for i in range(vocabulary_size)],1)
    edge=~np.eye(x.shape[1],dtype=bool)
    bond=np.stack([(bonds[:,edge]==i).mean(1) for i in range(1,5)],1)
    return np.concatenate([center,eig,fraction,bond],1)


def medoids(values, count=2):
    scale=np.maximum(values.std(0),.1)
    distance=cdist(values/scale,values/scale,'sqeuclidean')
    chosen=[int(distance.sum(1).argmin())]
    while len(chosen)<min(count,len(values)):
        rem=distance[:,chosen].min(1);rem[chosen]=-1
        if rem.max()<1e-14:break
        chosen.append(int(rem.argmax()))
    for _ in range(5):
        labels=distance[:,chosen].argmin(1)
        updated=[]
        for k,old in enumerate(chosen):
            ids=np.flatnonzero(labels==k)
            updated.append(int(ids[distance[np.ix_(ids,ids)].sum(1).argmin()]) if len(ids) else old)
        if updated==chosen:break
        chosen=updated
    labels=distance[:,chosen].argmin(1)
    counts=np.bincount(labels,minlength=len(chosen))
    return np.asarray(chosen),counts


def build_reference(dataset, campaign, batches, output, window_start=None, window_end=None, representatives=2):
    dataset=Path(dataset);root=dataset/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    cfg=read_json(root/'config.json');vocab=cfg['atom_vocabulary'];scale=cfg['coord_scale']
    frames=[];source=[];time_grid=None
    for batch in batches:
        path=root/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        frame=np.asarray(read_json(root/f'frame_batch_{batch:03d}.json')['target_com'])
        with open_trajectory(path) as z:
            times=np.round(z['score_time'][:,0].astype(float),6)
            events=np.flatnonzero(z['resampled'])
            lo=times[events].min() if window_start is None else float(window_start)
            hi=times[events].max() if window_end is None else float(window_end)
            if lo<times[events].min()-1e-6 or hi>times[events].max()+1e-6 or hi<=lo:
                raise ValueError('Requested learning window exceeds observed selection times')
            steps=np.flatnonzero((times>=lo-1e-6)&(times<=hi+1e-6))
            grid=times[steps]
            if not np.isclose(grid[0],lo) or not np.isclose(grid[-1],hi):raise ValueError('Window endpoints must have actual states')
            if time_grid is None:time_grid=grid
            elif not np.array_equal(time_grid,grid):raise ValueError('Unmatched reference time grids')
            # Exact X_s, before the selection scored at s; that selection acts on X_(s+dt).
            x=z['current_coords'][steps].astype(float)*scale+frame[None,:,None,:]
            atoms=z['current_atomics'][steps];bonds=z['current_bonds'][steps]
            mask=z['mask'][steps].astype(bool)
            if not mask.all():raise ValueError('This compact reference requires fixed active atom count')
            ep=z['predicted_atomics_probs_f16'][steps].astype(float)
            ep/=ep.sum(-1,keepdims=True)
            eb=z['predicted_bonds'][steps]
            for j,step in enumerate(steps):
                ids,counts=medoids(descriptor(x[j],atoms[j],bonds[j],len(vocab)),representatives)
                for slot,count in zip(ids,counts):
                    if count==0:continue
                    frames.append({'time':float(times[step]),'batch':int(batch),'slot':int(slot),
                        'mass':float(count/len(atoms[j])/len(batches)),
                        'coords':x[j,slot].tolist(),'atomics':atoms[j,slot].tolist(),'bonds':bonds[j,slot].tolist(),
                        'endpoint_type_fraction':ep[j,slot].mean(0).tolist(),
                        'endpoint_bond_fraction':[(eb[j,slot][~np.eye(len(atoms[j,slot]),dtype=bool)]==i).mean().item() for i in range(5)]})
        source.append({'path':path.relative_to(dataset).as_posix(),'sha256':digest(path)})
    data={'schema_version':'window-state-reference-1.0','representation':'actual_current_world_A',
        'window':[float(time_grid[0]),float(time_grid[-1])],'times':time_grid.tolist(),'batches':list(batches),
        'atom_vocabulary':vocab,'charge_vocabulary':cfg['charge_vocabulary'],
        'representatives_per_batch':representatives,'frames':frames,'sources':source,
        'required_input_sha256':{p.name:digest(p) for p in sorted((dataset/'inputs').iterdir()) if p.suffix in ('.pdb','.sdf')},
        'reference_policy':'Equal batch mass; within-batch deterministic medoids weighted by assignment counts; no coordinate interpolation between molecules',
        'target_semantics':'Actual current state at requested time; selection at that score time occurs after integration and is excluded from this state',
        'energy_status':'No physical force-field energies; geometry penalties must be named proxies'}
    out.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n').encode()
    out.write_bytes(gzip.compress(payload,compresslevel=6,mtime=0))
    write_json(out.with_suffix('.manifest.json'),{'sha256':digest(out),'bytes':out.stat().st_size,
        'window':data['window'],'times':len(time_grid),'representatives':len(frames),'batches':list(batches),'source':str(dataset)})
    return data
