"""Stream endpoint regional contrasts across the complete actual control window.

Only compact event/batch moments, cohort codes, whole-window uncertainty and
global polynomial fits are saved. Coordinates stay in the source trajectory.
"""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from numpy.polynomial.legendre import legval,legder,legvander
from ..io import read_json,write_json,digest
from ..trajectory_source import open_trajectory,trajectory_paths
from .affinity_geometry import numpy_geometry
from .coordinate_mining import bh
from .functional import quadrature
from .dynamic_cohorts import dynamic_partition,control_weights


def observed_integral(curves,times):
    """Quadrature mean on observed grid mass; missing evidence never becomes zero."""
    w=quadrature(np.asarray(times));valid=np.isfinite(curves)
    return np.nansum(curves*w[None,:,None],axis=1)/(valid*w[None,:,None]).sum(1)


def masked_curve_fit(times,curves,max_degree=3):
    times=np.asarray(times);curves=np.asarray(curves);weights=quadrature(times)
    u=2*(times-times[0])/(times[-1]-times[0])-1;basis=legvander(u,max_degree)
    masks=np.isfinite(curves);models=[];errors=[]
    for degree in range(max_degree+1):
        coefs=[]
        for curve,mask in zip(curves,masks):
            if mask.sum()<degree+2:raise ValueError('Insufficient observed curve support')
            coefs.append(np.linalg.lstsq(basis[mask,:degree+1]*np.sqrt(weights[mask,None]),
                curve[mask]*np.sqrt(weights[mask]),rcond=None)[0])
        coefs=np.asarray(coefs);models.append(coefs)
        loo=(coefs.sum(0)-coefs)/(len(coefs)-1);residual=curves-loo@basis[:,:degree+1].T
        errors.append(np.nansum(residual**2*weights,axis=1)/(masks*weights).sum(1))
    errors=np.asarray(errors);loss=errors.mean(1);best=loss.argmin()
    chosen=int(np.flatnonzero(loss<=loss[best]+errors[best].std(ddof=1)/np.sqrt(len(curves))+1e-14)[0])
    return {'degree':chosen,'legendre_coefficients':models[chosen].mean(0).tolist(),
        'time_start':float(times[0]),'time_end':float(times[-1]),'cv_mse_by_degree':loss.tolist(),
        'n_independent_batches':len(curves),'observed_grid_fraction_by_batch':masks.mean(1).tolist(),
        'derivative_semantics':'temporal derivative of observed cohort contrasts, not a coordinate force'}


def mine(dataset,campaign,incumbent_reference,output,batches=range(14),window=None,
         hard_mix=0.,margin_fraction=.10,gap_weight=False,maximum_regions=4):
    root,out=Path(dataset).resolve(),Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    base=json.loads(gzip.decompress(Path(incumbent_reference).read_bytes()))
    support=list(base['window'] if window is None else window)
    if support!=base['window']:raise ValueError('Incumbent and learned support must agree')
    source=root/'results'/campaign;cfg=read_json(source/'config.json')
    points,origin=np.asarray(base['landmarks_A']),np.asarray(base['origin_A'])
    fields=base['features'];batch_ids=sorted(map(int,batches));moments=[];events=[];codes_all=[];sources=[];grid=None
    paths={int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name=='single'}
    if len(batch_ids)<3:raise ValueError('At least three independent batches required')
    for batch in batch_ids:
        path=paths[batch];com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with open_trajectory(path) as z:
            t=np.round(np.asarray(z['score_time'])[:,0].astype(float),6);s=np.asarray(z['state_time'],float)
            event_ids=np.flatnonzero(np.asarray(z['resampled'],bool)&(t>=support[0]-2e-6)&(s<=support[1]+2e-6))
            if not len(event_ids) or np.any(np.diff(event_ids)!=1):raise ValueError('Contiguous actual selection support required')
            if grid is None:grid=t[event_ids]
            if not np.array_equal(grid,t[event_ids]) or not np.array_equal(grid,np.asarray(base['times'])):
                raise ValueError('Teacher and evidence clocks differ')
            actual_update_end=float(s[event_ids[-1]])
            if not np.isclose(actual_update_end,support[1],atol=2e-6):raise ValueError('Incomplete dynamic control window')
            x=np.asarray(z['predicted_coords'],float)[event_ids]*cfg['coord_scale']+com
            score=np.asarray(z['pic50_on'],float)[event_ids].reshape(len(event_ids),-1)
            roots=np.asarray(z['root_slot'])[event_ids];offspring=np.asarray(z['offspring_count'])[event_ids]
            if not np.asarray(z['mask'])[event_ids].all():raise ValueError('Fixed active point-cloud slots required')
            features=[numpy_geometry(xx,points,origin) for xx in x]
            rows=[];bcodes=[]
            for j,(xx,v,sc,r,oc) in enumerate(zip(x,features,score,roots,offspring)):
                c,w,part=dynamic_partition(sc,r,margin_fraction)
                if not part['identifiable']:
                    rows.append(np.full((5,len(fields)),np.nan));bcodes.append(c)
                    events.append({'batch':batch,'time':float(grid[j]),**part,'positive_n':0,'negative_n':0,
                        'ambiguous_n':len(c),'root_n':int(np.unique(r).size)})
                    continue
                wp,wn,control=control_weights(v,sc,c,w,hard_mix,gap_weight=gap_weight)
                high=wp@v[c==1];low=wn@v[c==-1]
                mu=w@v;var=w@((v-mu)**2)
                pos_var=wp@((v[c==1]-high)**2);neg_var=wn@((v[c==-1]-low)**2)
                rows.append((high,low,var,pos_var,neg_var))
                lo,hi=np.quantile(sc,[.25,.75]);quart=v[sc>=hi].mean(0)-v[sc<=lo].mean(0)
                events.append({'batch':batch,'time':float(grid[j]),**part,**control,
                    'positive_n':int((c==1).sum()),'negative_n':int((c==-1).sum()),'ambiguous_n':int((c==0).sum()),
                    'root_n':int(np.unique(r).size),'positive_root_n':int(np.unique(r[c==1]).size),
                    'negative_root_n':int(np.unique(r[c==-1]).size),
                    'positive_score_mean':float(wp@sc[c==1]),'negative_score_mean':float(wn@sc[c==-1]),
                    'positive_offspring_mean':float(wp@oc[c==1]),'negative_offspring_mean':float(wn@oc[c==-1]),
                    'quartile_contrast_l2':float(np.linalg.norm(quart))})
                bcodes.append(c)
            moments.append(rows);codes_all.append(bcodes)
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path),'batch':batch})
    # B,T,{positive,negative,population variance,positive variance,negative variance},F
    a=np.asarray(moments);delta_raw=a[:,:,0]-a[:,:,1]
    raw_scale=np.sqrt(np.nanmean(a[:,:,2],axis=0))
    # Floors relative to the observed metric's scale across pocket landmarks,
    # avoiding spurious amplification at almost unoccupied distant landmarks.
    scale=raw_scale.copy()
    for offset in range(3):
        ids=np.arange(14+offset,len(fields),3)
        floor=np.median(raw_scale[:,ids],axis=1)[:,None]*.10
        scale[:,ids]=np.maximum(scale[:,ids],floor)
    scale=scale.clip(1e-6);delta=delta_raw/scale[None]
    integral=observed_integral(delta,grid)
    rng=np.random.default_rng(42);boot=integral[rng.integers(0,len(batch_ids),(2000,len(batch_ids)))].mean(1)
    rows=[];functions={};trends=[]
    for k,name in enumerate(fields):
        ci=np.quantile(boot[:,k],[.025,.975]);fit=masked_curve_fit(grid,delta[:,:,k],max_degree=3)
        functions[name]=fit;coef=fit['legendre_coefficients'];u=2*(grid-grid[0])/(grid[-1]-grid[0])-1
        fitted=legval(u,coef);rate=legval(u,legder(coef))*2/(grid[-1]-grid[0])
        rows.append({'feature':name,'feature_index':k,'integrated_contrast_z':float(integral[:,k].mean()),
            'CI_low':float(ci[0]),'CI_high':float(ci[1]),'p':float(ttest_1samp(integral[:,k],0).pvalue),
            'sign_agreement':float((np.sign(integral[:,k])==np.sign(integral[:,k].mean())).mean()),
            'end_minus_start_z':float(np.nanmean(delta[:,-1,k])-np.nanmean(delta[:,0,k])),'fit_degree':fit['degree'],
            'fit_derivative_start':float(rate[0]),'fit_derivative_end':float(rate[-1])})
        for j,t in enumerate(grid):trends.append({'time':float(t),'feature':name,'contrast_z':float(np.nanmean(delta[:,j,k])),
            'positive_mean':float(np.nanmean(a[:,j,0,k])),'negative_mean':float(np.nanmean(a[:,j,1,k])),
            'observed_batches':int(np.isfinite(delta[:,j,k]).sum()),
            'endpoint_feature_scale':float(scale[j,k]),'fitted_contrast_z':float(fitted[j]),'fitted_rate_z_per_time':float(rate[j])})
    for row,q in zip(rows,bh(np.nan_to_num([r['p'] for r in rows],nan=1.))):row['q']=float(q)
    effects=pd.DataFrame(rows);eligible=effects[(effects.feature_index>=14)&(effects.q<=.05)&(effects.sign_agreement>=.75)]
    region_ranks=[]
    for region,g in eligible.groupby((eligible.feature_index-14)//3):
        region_ranks.append({'region_index':int(region),'score':float(g.integrated_contrast_z.abs().max()),
            'minimum_q':float(g.q.min()),'feature_indices':g.feature_index.astype(int).tolist()})
    region_ranks.sort(key=lambda r:(-r['score'],r['region_index']))
    chosen=region_ranks[:maximum_regions]
    if not chosen:raise ValueError('No multiplicity-adjusted region supports a reward proposal')
    selected=sorted({k for r in chosen for k in r['feature_indices']})
    frames=[]
    for j,t in enumerate(grid):
        frame=base['frames'][j].copy()
        valid=np.isfinite(a[:,j,0,0])
        if valid.sum()<3:raise ValueError('Fewer than three observed independent cohort batches at a control node')
        frame['dynamic_region']={'positive_centers':a[valid,j,0].tolist(),'negative_centers':a[valid,j,1].tolist(),
            'observed_batches':np.asarray(batch_ids)[valid].tolist(),
            'feature_scale':scale[j].tolist(),'positive_variance':(np.nanmean(a[:,j,3],axis=0)/scale[j]**2+.25).tolist(),
            'negative_variance':(np.nanmean(a[:,j,4],axis=0)/scale[j]**2+.25).tolist()}
        frames.append(frame)
    ref={**base,'frames':frames,'reference_variant':'dynamic-cohort-endpoint-library-1.0',
        'allowed_reward_views':['endpoint_pointcloud','endpoint_dynamic_region'],
        'dynamic_cohort':{'policy':'root-balanced weighted Otsu with ambiguous guard band','margin_fraction':margin_fraction,
            'hard_mix':hard_mix,'gap_weight':gap_weight,'maximum_regions':maximum_regions,'selected_feature_indices':selected,
            'regions':chosen,'label_semantics':'Relative recorded online joint-head quality; no terminal negatives inferred',
            'coordinate_representation':'predicted endpoint world Angstrom at the pre-selection score clock',
            'negative_semantics':'Observed separated lower online scores; unobserved future outcome is not failure'},
        'dynamic_sources':sources,'incumbent_reference_sha256':digest(incumbent_reference)}
    out.mkdir(parents=True);p=out/'reference.json.gz'
    p.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    effects.to_parquet(out/'whole_window_effects.parquet',index=False,compression=None)
    pd.DataFrame(events).to_parquet(out/'event_cohorts.parquet',index=False,compression=None)
    pd.DataFrame(trends).to_parquet(out/'feature_trends.parquet',index=False,compression=None)
    np.savez_compressed(out/'cohort_codes.npz',codes=np.asarray(codes_all,np.int8),batches=np.array(batch_ids),times=grid)
    write_json(out/'effect_functions.json',functions)
    manifest={'schema_version':'dynamic-coordinate-enrichment-1.0','window':support,'score_grid':grid.tolist(),
        'actual_update_end':actual_update_end,'batches':batch_ids,'independent_units':'generation batches',
        'policy':ref['dynamic_cohort'],'features':len(fields),'sources':sources,
        'unidentifiable_event_count':int(sum(not r['identifiable'] for r in events)),
        'observed_grid_fraction_by_batch':np.isfinite(a[:,:,:,0]).all(2).mean(1).tolist(),
        'reference_sha256':digest(p),'incumbent_reference_sha256':digest(incumbent_reference),
        'storage':'One-byte cohort codes and aggregate float64 moments; no expanded particle features or coordinate copies',
        'selection_boundary':'Events with proposal state outside support excluded; boundary state retained in original source',
        'limitations':['Online head association is observational, not causal affinity gain',
            'Family balance reduces clone count bias but is not a new independent replicate',
            'Gaussian moments approximate multimodal regional fields; reward trials must validate this approximation'],
        'files':[{ 'path':f.name,'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(out.iterdir())]}
    write_json(out/'manifest.json',manifest);return manifest
