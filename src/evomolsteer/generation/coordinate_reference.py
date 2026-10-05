"""Discovery-only, time-varying regional current-state mixture references."""
import gzip
import json
from pathlib import Path
import numpy as np
from ..continuous.coordinate_features import regional_moments
from ..storage.trajectory import TrajectoryPackage
from ..io import read_json,digest,write_json
from .prototypes import regularize_covariance


def build(dataset,campaign,mining,output,regions,channel='all',std_floor_A=.15):
    root=Path(dataset);source=root/'results'/campaign;mining=Path(mining);output=Path(output)
    if output.exists():raise FileExistsError(output)
    m=read_json(mining/'manifest.json');cat=read_json(mining/'feature_catalog.json')
    if not regions or len(regions)!=len(set(regions)) or any(r not in cat['regions'] for r in regions):raise ValueError('Explicit distinct measured regions required')
    if channel not in ('all','NOS') or std_floor_A<=0:raise ValueError('Invalid coordinate reference')
    scale=read_json(source/'config.json')['coord_scale'];batches=m['splits']['discovery'];frames={};pooled={};sources=[]
    labels=None if channel=='all' else [cat['atom_vocabulary'][e] for e in ('N','O','S')]
    for batch in batches:
        p=source/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with TrajectoryPackage(p) as z:
            times=np.round(z.read('score_time')[:,0].astype(float),6)
            for i in np.flatnonzero(z.read('resampled')):
                t=float(times[i]);x=z.read('current_coords',int(i)).astype(float)*scale+com
                atoms=z.read('predicted_atomics',int(i));mask=z.read('mask',int(i))
                v=np.concatenate([regional_moments(x,atoms,mask,cat['regions'][r]['points_A'],labels,cat['spatial_width_A']) for r in regions],axis=1)
                valid=np.isfinite(v).all(1);v=v[valid];w=z.read('selection_probability',int(i)).astype(float)[valid]
                if len(v)<3 or w.sum()<=0:raise ValueError('Missing reference measurements')
                w/=w.sum();mu=w@v;d=v-mu;cov=np.einsum('bi,b,bj->ij',d,w,d)
                pooled.setdefault(t,[]).append((v,w/len(batches)))
                frames.setdefault(t,[]).append({'center_A':mu.tolist(),'covariance_A2':regularize_covariance(cov,.1,std_floor_A).tolist(),
                    'unweighted_spread_variance_A2':np.var(v[:,3::4],axis=0,ddof=1).tolist(),
                    'source_batch':batch,'availability':float(valid.mean())})
        sources.append({'path':p.relative_to(root).as_posix(),'sha256':digest(p)})
    times=sorted(frames)
    if times!=m['times']:raise ValueError('Reference/learning time mismatch')
    from .local_reference import weighted_quantile
    extra={}
    for t in times:
        values=np.concatenate([v for v,w in pooled[t]]);weights=np.concatenate([w for v,w in pooled[t]])
        upper=[weighted_quantile(values[:,k],weights,.75) for k in range(3,values.shape[1],4)]
        sigma=np.sqrt(np.maximum(np.mean([v['unweighted_spread_variance_A2'] for v in frames[t]],axis=0),std_floor_A**2))
        extra[t]={'upper_spread_A':upper,'spread_scale_A':sigma.tolist(),'quantile':.75}
    from .launcher import INPUT_FILES
    ref={'schema_version':'current-coordinate-mixture-1.0','window':m['window'],'times':times,
         'regions':{r:cat['regions'][r] for r in regions},'channel':channel,'atom_vocabulary':cat['atom_vocabulary'],
         'spatial_width_A':cat['spatial_width_A'],'frames':[{'time':t,'modes':frames[t],**extra[t]} for t in times],
         'features':[f'{r}::{channel}::{k}' for r in regions for k in ('centroid_x','centroid_y','centroid_z','spread')],
         'batches':batches,'sources':sources,'std_floor_A':std_floor_A,
         'representation':'actual current state in aligned world angstrom; endpoint labels as conditional masks',
         'target_definition':'one same-time affinity-selected mean/covariance per independent discovery batch; equal mode weights',
         'upper_spread_definition':'equal-batch conditional selected 75th quantile; scale from same-event unweighted within-batch candidate variance with declared SD floor',
         'limitations':['Observational selection imitation, not causal regional affinity improvement',
             'Modes are batch representatives, not independent recovered molecular lineages',
             'Regional centroid/spread do not uniquely determine atomic geometry',
             'Overlapping regional views are a joint covariance objective, not independent mechanisms'],
         'required_input_sha256':{name:digest(root/'inputs'/name) for name in INPUT_FILES}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(gzip.compress((json.dumps(ref,separators=(',',':'),allow_nan=False)+'\n').encode(),mtime=0))
    write_json(output.with_suffix('.manifest.json'),{'sha256':digest(output),'bytes':output.stat().st_size,'window':ref['window'],
        'regions':list(regions),'channel':channel,'mining_manifest_sha256':digest(mining/'manifest.json')})
    return ref
