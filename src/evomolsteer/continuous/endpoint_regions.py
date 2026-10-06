"""Permutation-invariant local endpoint moments on the whole selection window.

Stream original trajectories; discard candidate features after parent-debiased
batch/node summaries. Score strata are observational, never causal labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from .coordinate_mining import bh
from .endpoint_kinematics import parent_mean
from ..io import read_json, write_json, digest
from ..storage.trajectory import TrajectoryPackage
from ..generation.window_reference import load_reference


COMPONENTS=('soft_mass','local_dx','local_dy','local_dz','local_cxx','local_cyy','local_czz','local_cxy','local_cxz','local_cyz')


def regional_moments(coords, landmarks, radius_A=5.):
    """Gaussian mass, mean offset and six covariance entries in receptor frame."""
    x=np.asarray(coords,float);p=np.asarray(landmarks,float)
    if x.ndim!=3 or x.shape[-1]!=3 or not np.isfinite(x).all() or radius_A<=0:raise ValueError('Finite B,N,3 coordinates and positive radius required')
    d=x[:,:,None,:]-p[None,None,:,:]
    w=np.exp(-(d*d).sum(-1)/(2*radius_A**2));mass=w.sum(1)
    mean=(w[:,:,:,None]*d).sum(1)/mass[:,:,None].clip(1e-30)
    centered=d-mean[:,None];cov=np.einsum('bnr,bnri,bnrj->brij',w,centered,centered)/mass[:,:,None,None].clip(1e-30)
    return np.concatenate([mass[:,:,None]/x.shape[1],mean,
        np.stack([cov[:,:,0,0],cov[:,:,1,1],cov[:,:,2,2],cov[:,:,0,1],cov[:,:,0,2],cov[:,:,1,2]],-1)],-1).reshape(len(x),-1)


def mine(reference,dataset,campaign,output,radius_A=5.):
    ref=load_reference(reference);root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    cfg=read_json(source/'config.json');times=np.asarray(ref['times'],float);nregion=len(ref['landmarks_A'])
    features=[f'landmark_{r:02d}_{c}' for r in range(nregion) for c in COMPONENTS]
    high=[];low=[];edge_high=[];edge_low=[];scales=[];batches=[];counts=[]
    for item in ref['sources']:
        path=root/item['path']
        if digest(path)!=item['sha256']:raise ValueError('Immutable original source mismatch')
        batch=int(path.parent.name.split('_')[-1]);batches.append(batch)
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        h=[];l=[];eh=[];el=[];s=[];previous=None;previous_parents=None
        with TrajectoryPackage(path) as tr:
            grid=np.round(tr.read('score_time')[:,0].astype(float),6);states=tr.read('state_time')
            ids=np.flatnonzero(tr.read('resampled')&(grid>=ref['window'][0]-1e-6)&(states<=ref['window'][1]+1e-6))
            if not np.array_equal(grid[ids],times) or np.any(np.diff(ids)!=1):raise ValueError('Exact contiguous selection support required')
            for k,i in enumerate(ids):
                y=tr.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                phi=regional_moments(y,ref['landmarks_A'],radius_A)
                score=tr.read('pic50_on',int(i)).reshape(-1);q0,q1=np.quantile(score,[.25,.75])
                parents=np.arange(len(phi)) if k==0 else previous_parents
                if parents.shape!=(len(phi),) or np.any((parents<0)|(parents>=len(phi))):raise ValueError('Invalid selected-parent edge')
                hh,nh=parent_mean(phi,parents,score>=q1);ll,nl=parent_mean(phi,parents,score<=q0)
                h.append(hh);l.append(ll);s.append(phi.std(0));counts.append(dict(batch=batch,score_time=float(times[k]),high_parents=nh,low_parents=nl))
                if k:
                    delta=phi-previous[parents]
                    eh.append(parent_mean(delta,parents,score>=q1)[0]);el.append(parent_mean(delta,parents,score<=q0)[0])
                else:eh.append(np.zeros(len(features)));el.append(np.zeros(len(features)))
                previous=phi;previous_parents=tr.read('selected_indices',int(i)).astype(int)
        high.append(h);low.append(l);edge_high.append(eh);edge_low.append(el);scales.append(s)
    high=np.asarray(high);low=np.asarray(low);eh=np.asarray(edge_high);el=np.asarray(edge_low)
    scale=np.maximum(np.mean(scales,axis=(0,1)),1e-6);effect=(high-low)/scale
    elapsed=times[-1]-times[0]
    measures={'enrichment_z':np.trapezoid(effect,times,axis=1)/elapsed,
              'enrichment_change_rate_z':(effect[:,-1]-effect[:,0])/elapsed,
              'parent_edge_contrast_rate_z':((eh-el)[:,1:]/scale).sum(1)/elapsed}
    # One correction over all three predeclared families, not per region.
    pvals=np.concatenate([np.nan_to_num(ttest_1samp(v,0,axis=0).pvalue,nan=1.) for v in measures.values()]);qvals=bh(pvals)
    rng=np.random.default_rng(42);indices=rng.integers(0,len(high),(2000,len(high)));rows=[]
    for mi,(kind,values) in enumerate(measures.items()):
        ci=np.quantile(values[indices].mean(1),[.025,.975],axis=0)
        for j,f in enumerate(features):
            rows.append(dict(feature=f,region=j//len(COMPONENTS),component=COMPONENTS[j%len(COMPONENTS)],measure=kind,
                effect_z=float(values[:,j].mean()),CI_low=float(ci[0,j]),CI_high=float(ci[1,j]),
                positive_batch_fraction=float((values[:,j]>0).mean()),p=float(pvals[mi*len(features)+j]),q=float(qvals[mi*len(features)+j])))
    table=pd.DataFrame(rows);supported=table[(table.measure=='enrichment_z')&(table.q<.05)&(table.effect_z.abs()>=.1)]
    supported=supported[np.maximum(supported.positive_batch_fraction,1-supported.positive_batch_fraction)>=12/14]
    region_weights=np.zeros(nregion)
    for r,g in supported.groupby('region'):region_weights[int(r)]=float(g.effect_z.abs().max())
    if region_weights.max():region_weights/=region_weights.max()
    node_functions={}
    for f in supported.feature:
        j=features.index(f);mean=effect[:,:,j].mean(0)
        node_functions[f]={'values_z':mean.tolist(),'d_dt_z':np.gradient(mean,times).tolist(),'scale':float(scale[j]),
            'mean_high':high[:,:,j].mean(0).tolist(),'mean_low':low[:,:,j].mean(0).tolist()}
    # 700 batch/node rows: no candidate-level expanded cache or coordinate copy.
    compact=pd.DataFrame({'batch':np.repeat(batches,len(times)),'score_time':np.tile(times,len(batches))})
    arrays={'high':high,'low':low,'edge_high':eh,'edge_low':el}
    compact=pd.concat([compact,pd.DataFrame({f'{name}__{f}':v[:,:,j].reshape(-1) for name,v in arrays.items() for j,f in enumerate(features)})],axis=1)
    out.mkdir(parents=True);compact.to_parquet(out/'batch_node_moments.parquet',compression=None,index=False)
    table.to_parquet(out/'whole_window_effects.parquet',compression=None,index=False)
    pd.DataFrame(counts).to_parquet(out/'parent_support.parquet',compression=None,index=False)
    prior={'schema_version':'endpoint-local-moment-prior-1.0','window':ref['window'],'times':ref['times'],
        'reference_sha256':digest(reference),'source_code_sha256':digest(__file__),'sources':ref['sources'],
        'radius_A':radius_A,'features':features,'feature_scales':scale.tolist(),'region_weights':region_weights.tolist(),
        'supported_node_functions':node_functions,'supported_region_indices':np.flatnonzero(region_weights).tolist(),
        'independent_batches':len(batches),'multiple_testing':'BH jointly over all 600 predeclared field/measure tests',
        'statistics_unit':'Equal batches after equal averaging of children within extant parent and score stratum',
        'first_edge_observed':False,'limitations':['Selection-conditioned observational associations, not causal affinity features',
            'Endpoint forecast moments, not physical atom velocity; no atom-type fields',
            'Gaussian regions overlap; a supported field does not identify a unique residue mechanism',
            'Compact moment functions do not recover full atom coordinates; raw original trajectories remain protected']}
    write_json(out/'region_prior.json',prior)
    write_json(out/'manifest.json',{'prior_sha256':digest(out/'region_prior.json'),'feature_count':len(features),
        'supported_fields':len(supported),'supported_regions':prior['supported_region_indices'],
        'table_bytes':{p.name:p.stat().st_size for p in out.glob('*.parquet')}})
    return prior
