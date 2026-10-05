"""Continuous same-time selected regional references; no terminal outcome leakage."""
import gzip
import json
from pathlib import Path
import numpy as np
from .prototypes import FEATURES,measure_patch,regularize_covariance
from ..io import read_json,digest,write_json
from ..trajectory_source import open_trajectory


def weighted_quantile(values,weights,q):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order];w=np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w)/w.sum(),q,side='left')])


def build_local_reference(dataset,campaign,analysis,output,mass=.5):
    root=Path(dataset);analysis=Path(analysis);output=Path(output)
    if output.exists():raise FileExistsError(output)
    cfg=read_json(analysis/'config.json');catalog=read_json(analysis/'feature_catalog.json')
    source=root/'results'/campaign;scfg=read_json(source/'config.json');scale=scfg['coord_scale']
    catalog={k:catalog[k] for k in ['atom_vocabulary','regions','features']}
    catalog['features']={f:catalog['features'][f] for f in FEATURES}
    regions={v['region'] for v in catalog['features'].values()}
    catalog['regions']={k:v for k,v in catalog['regions'].items() if k in regions}
    batches=cfg['discovery_batches'];samples={};sources=[];grid=None
    for arm in ['single','unguided']:
        for batch in batches:
            p=source/arm/f'batch_{batch:03d}'/'trajectory.h5'
            com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])
            with open_trajectory(p) as z:
                times=np.round(z['score_time'][:,0].astype(float),6)
                selection=np.flatnonzero(z['resampled']) if arm=='single' else None
                if arm=='single':
                    window=[float(times[selection].min()),float(times[selection].max())]
                ids=np.flatnonzero((times>=window[0]-1e-7)&(times<=window[1]+1e-7))
                if grid is None:grid=times[ids]
                if not np.array_equal(grid,times[ids]):raise ValueError('Time grid mismatch')
                xyz=z['predicted_coords'][ids].astype(float)*scale+com[None,:,None,:]
                a=z['predicted_atomics'][ids];mask=z['mask'][ids]
                values=measure_patch(xyz.reshape(-1,xyz.shape[2],3),a.reshape(-1,a.shape[2]),mask.reshape(-1,mask.shape[2]),catalog).reshape(len(ids),a.shape[1],-1)
                probability=z['selection_probability'][ids].astype(float)
                samples[arm,batch]=(values,probability)
            sources.append({'path':p.relative_to(root).as_posix(),'sha256':digest(p)})
    frames=[]
    for i,t in enumerate(grid):
        all_values=[];all_weights=[];cov=[];availability=[]
        for b in batches:
            v,p=samples['single',b];valid=np.isfinite(v[i]).all(1)
            w=p[i]*valid
            if w.sum()<=0:raise ValueError('Missing selected region')
            all_values.append(v[i,valid]);all_weights.append(w[valid]/w.sum()/len(batches))
            u=samples['unguided',b][0][i];u=u[np.isfinite(u).all(1)]
            if len(u)<3:raise ValueError('Insufficient native regional covariance')
            cov.append(np.cov(u,rowvar=False));availability.append(float(valid.mean()))
        v=np.concatenate(all_values);w=np.concatenate(all_weights)
        center=(v*w[:,None]).sum(0);covariance=regularize_covariance(np.mean(cov,0))
        delta=v-center;q=np.einsum('bi,ij,bj->b',delta,np.linalg.inv(covariance),delta)
        radius2=max(weighted_quantile(q,w,mass),1e-6)
        frames.append({'time':float(t),'center_A':center.tolist(),'covariance_A2':covariance.tolist(),
                       'radius_squared':radius2,'selected_mass_inside':float(w[q<=radius2].sum()),
                       'availability_by_batch':availability})
    from .launcher import INPUT_FILES
    ref={'schema_version':'regional-interval-reference-1.0','window':window,'times':grid.tolist(),
         'representation':'predicted_endpoint_world_A','catalog':catalog,'frames':frames,'features':FEATURES,
         'batches':batches,'sources':sources,'central_probability_mass':mass,
         'weighting':'equal discovery batch mass; conditional within-batch selection_probability',
         'covariance_source':'same-time unweighted native within-batch covariance; .1 diagonal shrinkage, .1 A eigen-sd floor',
         'limitations':['Selection association, not causal effect','Single ellipse may bridge modes','50 percent central mass is an uncalibrated acceptable-set assumption','No terminal labels used'],
         'required_input_sha256':{name:digest(root/'inputs'/name) for name in INPUT_FILES}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(gzip.compress((json.dumps(ref,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    write_json(output.with_suffix('.manifest.json'),{'sha256':digest(output),'bytes':output.stat().st_size,'times':len(grid),'window':window,'source':str(root)})
    return ref
