"""Discovery-only, time-varying regional current-state mixture references."""
import gzip
import json
from pathlib import Path
import numpy as np
from ..continuous.coordinate_features import regional_moments
from ..storage.trajectory import TrajectoryPackage
from ..io import read_json,digest,write_json
from .prototypes import regularize_covariance


def mode_record(v,w,batch,availability,std_floor_A):
    """Small joint moments plus auditable selection/background contrast diagnostics."""
    mu=w@v;d=v-mu;cov=np.einsum('bi,b,bj->ij',d,w,d)
    bg=v.mean(0);residual=v-bg;background=residual.T@residual/len(v)
    regularized_background=regularize_covariance(background,.1,std_floor_A)
    shift=mu-bg;positive=w>0
    return {'center_A':mu.tolist(),'covariance_A2':regularize_covariance(cov,.1,std_floor_A).tolist(),
        'raw_selected_eigenvalues_A2':np.linalg.eigvalsh(cov).tolist(),
        'background_center_A':bg.tolist(),'background_covariance_A2':regularized_background.tolist(),
        'selected_probability_ESS':float(1/(w@w)),
        'selected_KL_to_uniform':float(np.sum(w[positive]*np.log(w[positive]*len(v)))),
        'dimension_normalized_mean_contrast':float(np.sqrt(max(0,shift@np.linalg.solve(regularized_background,shift)/len(mu)))),
        'unweighted_spread_variance_A2':np.var(v[:,3::4],axis=0,ddof=1).tolist(),
        'source_batch':batch,'availability':float(availability),'n_available':len(v)}


def build(dataset,campaign,mining,output,regions,channel='all',std_floor_A=.15,global_control=False):
    root=Path(dataset);source=root/'results'/campaign;mining=Path(mining);output=Path(output)
    if output.exists():raise FileExistsError(output)
    m=read_json(mining/'manifest.json');cat=read_json(mining/'feature_catalog.json')
    if not regions or len(regions)!=len(set(regions)) or any(r not in cat['regions'] for r in regions):raise ValueError('Explicit distinct measured regions required')
    if channel not in ('all','NOS') or std_floor_A<=0:raise ValueError('Invalid coordinate reference')
    if global_control and len(regions)!=1:raise ValueError('One frame origin is sufficient for global control')
    width=None if global_control else cat['spatial_width_A']
    anchor_view=m.get('spatial_anchor','current');control_view=m.get('control_representation','current')
    if global_control and anchor_view!='current':raise ValueError('Global control ignores regional anchor; use current-anchor input')
    scale=read_json(source/'config.json')['coord_scale'];batches=m['splits']['discovery'];frames={};pooled={};sources=[]
    labels=None if channel=='all' else [cat['atom_vocabulary'][e] for e in ('N','O','S')]
    for batch in batches:
        p=source/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with TrajectoryPackage(p) as z:
            times=np.round(z.read('score_time')[:,0].astype(float),6)
            for i in np.flatnonzero(z.read('resampled')):
                t=float(times[i]);x=z.read(f'{control_view}_coords',int(i)).astype(float)*scale+com
                anchor=z.read('predicted_coords',int(i)).astype(float)*scale+com if anchor_view=='endpoint' else None
                atoms=z.read('predicted_atomics',int(i));mask=z.read('mask',int(i))
                v=np.concatenate([regional_moments(x,atoms,mask,cat['regions'][r]['points_A'],labels,width,anchor) for r in regions],axis=1)
                valid=np.isfinite(v).all(1);v=v[valid];w=z.read('selection_probability',int(i)).astype(float)[valid]
                if len(v)<3 or w.sum()<=0:raise ValueError('Missing reference measurements')
                w/=w.sum()
                pooled.setdefault(t,[]).append((v,w/len(batches)))
                frames.setdefault(t,[]).append(mode_record(v,w,batch,valid.mean(),std_floor_A))
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
         'regions':{'ligand':cat['regions'][regions[0]]} if global_control else {r:cat['regions'][r] for r in regions},'channel':channel,'atom_vocabulary':cat['atom_vocabulary'],
         'spatial_weighting':'uniform_global_control' if global_control else 'gaussian_regional',
         'spatial_anchor':anchor_view,'control_representation':control_view,
         'time_alignment':'score time t -> proposal state time t+dt, anchor from endpoint at t' if control_view=='proposal' else 'state time s -> current reference s',
         'spatial_width_A':cat['spatial_width_A'],'frames':[{'time':t,'modes':frames[t],**extra[t]} for t in times],
         'features':[f'{r}::{channel}::{("proposal_" if control_view=="proposal" else "")+k}' for r in (['ligand'] if global_control else regions) for k in ('centroid_x','centroid_y','centroid_z','spread')],
         'batches':batches,'sources':sources,'std_floor_A':std_floor_A,
         'representation':f'actual {control_view} state in aligned world angstrom; {anchor_view} Gaussian anchor and endpoint labels as conditional masks',
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
