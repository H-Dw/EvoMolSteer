"""Multi-depth conditional coordinate mutations with compact spatial evidence.

Missing ancestry/directions remain missing. No terminal utility, fitted affinity
head, chemistry gate or new model is introduced. This independent tool does not
modify the frozen lag-two tool or the existing Analyst/Designer contract.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legder
from scipy.stats import ttest_1samp

from ..io import digest,read_json,write_json,clean
from ..trajectory_source import open_trajectory,trajectory_paths
from .affinity_geometry import names,numpy_geometry
from .branch_mutation import alias_diagnostics,chemistry_nuisance
from .coordinate_mining import bh
from .dynamic_regions import masked_curve_fit,observed_integral


def parse_depths(value):
    values=[int(v) for v in value.split(',')] if isinstance(value,str) else list(value)
    if not values or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<2 for v in values):
        raise ValueError('Distinct ancestry depths >=2 required')
    if len(set(values))!=len(values):raise ValueError('Duplicate ancestry depths')
    return sorted(map(int,values))


def ancestor_indices(selected,event,depth):
    s=np.asarray(selected)
    if s.ndim!=2 or not np.issubdtype(s.dtype,np.integer) or depth<1 or event<depth or event>=len(s):
        raise ValueError('Observed integer ancestry and in-window depth required')
    if np.any((s<0)|(s>=s.shape[1])):raise ValueError('Ancestry index out of range')
    ancestor=s[event-1].copy()
    for index in range(event-2,event-depth-1,-1):ancestor=s[index,ancestor]
    return ancestor


def signed_regions(endpoint,ancestor_endpoint,points,width=4.):
    """Density-weighted internal XYZ change in a fixed receptor coordinate frame.

    Membership is evaluated on the ancestor endpoint. Whole-cloud translation is
    removed explicitly; companion geometry still records all global pose changes.
    Unnormalized membership avoids inventing a large region far from all atoms.
    """
    y,a,p=np.asarray(endpoint,float),np.asarray(ancestor_endpoint,float),np.asarray(points,float)
    if y.shape!=a.shape or y.ndim!=3 or p.ndim!=2 or p.shape[1]!=3 or width<=0:raise ValueError('Aligned finite coordinate regions required')
    delta=y-a;translation=delta.mean(1);internal=delta-translation[:,None]
    membership=np.exp(-.5*((a[:,:,None]-p[None,None])**2).sum(-1)/width**2)
    regional=np.einsum('bnr,bnd->brd',membership,internal)/a.shape[1]
    return regional,translation,membership.mean(1)


def conditional_statistics(x,y,z,parents,ancestors):
    """Copy-invariant equal-ancestor moments, arbitrary diagnostic nuisance size."""
    x,y,z,p,g=np.asarray(x,float),np.asarray(y,float),np.asarray(z,float),np.asarray(parents),np.asarray(ancestors)
    if x.ndim!=2 or y.shape!=(len(x),) or z.ndim!=2 or len(z)!=len(x) or p.shape!=y.shape or g.shape!=y.shape:
        raise ValueError('Aligned observations required')
    if not all(np.isfinite(v).all() for v in (x,y,z)):raise ValueError('Finite measured observations required')
    average=lambda v:v[0].copy() if np.array_equal(v,np.broadcast_to(v[0],v.shape)) else v.mean(0)
    branches=[]
    for parent in np.unique(p):
        ids=p==parent
        if np.unique(g[ids]).size!=1:raise ValueError('One ancestor per immediate-parent branch required')
        branches.append((g[ids][0],average(x[ids]),float(average(y[ids])),average(z[ids])))
    bx=[];by=[];bz=[];weights=[];count=0
    for ancestor in sorted({v[0] for v in branches}):
        group=[v for v in branches if v[0]==ancestor]
        if len(group)<2:continue
        xx,yy,zz=np.array([v[1] for v in group]),np.array([v[2] for v in group]),np.array([v[3] for v in group])
        dx=xx-xx.mean(0);dx[:,np.ptp(xx,axis=0)==0]=0.
        dz=zz-zz.mean(0);dz[:,np.ptp(zz,axis=0)==0]=0.
        dy=yy-yy.mean() if np.ptp(yy)>0 else np.zeros_like(yy)
        bx.extend(dx);by.extend(dy);bz.extend(dz);weights.extend([1/len(group)]*len(group));count+=1
    missing=np.full(x.shape[1],np.nan)
    if not count:return {m:missing.copy() for m in ('raw_covariance','adjusted_covariance','raw_correlation','adjusted_correlation')},{'branching_ancestors':0,'distinct_branches':0,'nuisance_rank':0,'conditional_score_variance':np.nan}
    bx,by,bz,w=np.asarray(bx),np.asarray(by),np.asarray(bz),np.asarray(weights)/count
    def moments(xx,yy):
        covariance=np.einsum('n,nf,n->f',w,xx,yy);variance_x=np.einsum('n,nf,nf->f',w,xx,xx);variance_y=float(w@(yy*yy))
        denominator=np.sqrt(variance_x*variance_y)
        return covariance,np.divide(covariance,denominator,out=missing.copy(),where=denominator>1e-14),variance_y
    cov,corr,vy=moments(bx,by)
    weighted=bz*np.sqrt(w[:,None]);inverse=np.linalg.pinv(weighted,rcond=1e-12)
    rx=bx-bz@(inverse@(bx*np.sqrt(w[:,None])));ry=by-bz@(inverse@(by*np.sqrt(w)))
    adjusted,adjusted_corr,_=moments(rx,ry)
    return {'raw_covariance':cov,'adjusted_covariance':adjusted,'raw_correlation':corr,'adjusted_correlation':adjusted_corr}, {
        'branching_ancestors':count,'distinct_branches':len(bx),'nuisance_rank':int(np.linalg.matrix_rank(weighted,tol=1e-12)),
        'conditional_score_variance':vy,'within_ancestor_contrast_degrees':len(bx)-count}


def cross_batch_directions(vectors,tolerance=1e-12):
    """Observed nonzero batch directions only; missing never receives cosine zero."""
    v=np.asarray(vectors,float)
    if v.ndim!=3 or v.shape[-1]!=3:raise ValueError('Batch/time/XYZ direction arrays required')
    norms=np.linalg.norm(v,axis=-1);valid=np.isfinite(v).all(-1)&(norms>tolerance)
    cosines=[];times_with_pairs=0
    for time in range(v.shape[1]):
        ids=np.flatnonzero(valid[:,time])
        if len(ids)<2:continue
        times_with_pairs+=1;unit=v[ids,time]/norms[ids,time,None]
        for i in range(len(unit)):
            for j in range(i):cosines.append(float(unit[i]@unit[j]))
    return {'direction_observed_fraction':float(valid.mean()),'batch_time_direction_count':int(valid.sum()),
            'observed_pair_count':len(cosines),'times_with_observed_pairs':times_with_pairs,
            'observed_pair_mean_cosine':float(np.mean(cosines)) if cosines else np.nan,
            'observed_pair_positive_fraction':float(np.mean(np.asarray(cosines)>0)) if cosines else np.nan,
            'interpretation':'Descriptive dependent batch pairs, never an inferential sample count. Missing/zero directions excluded.'}


def write_compact_json(path,value):
    Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,separators=(',',':'),sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def mine(dataset,campaign,reference,output,depths=(2,3,5,8,13),seed=42,region_width=4.,nuisance='chemistry_pose'):
    root,refpath,out=Path(dataset).resolve(),Path(reference).resolve(),Path(output).resolve();depths=parse_depths(depths)
    if out.exists():raise FileExistsError(out)
    if nuisance not in ('chemistry','chemistry_pose'):raise ValueError('Registered diagnostic nuisance required')
    ref=json.loads(gzip.decompress(refpath.read_bytes()));grid=np.asarray(ref['times'],float)
    if ref['schema_version']!='affinity-endpoint-library-1.0' or max(depths)>=len(grid)-3:raise ValueError('Full-window reference and observable depth required')
    batches=ref['discovery_batches'];source=root/'results'/campaign;cfg=read_json(source/'config.json')
    points,origin=np.asarray(ref['landmarks_A'],float),np.asarray(ref['origin_A'],float)
    geometric=names(points)+['joint_innovation_RMS_A','joint_translation_RMS_A','joint_internal_RMS_A']
    regional=[f'region_{r:02d}_internal_displacement_{axis}' for r in range(len(points)) for axis in 'xyz'];features=geometric+regional
    paths={int(p.parent.name.split('_')[1]):p for p in trajectory_paths(source) if p.parent.parent.name=='single'}
    metrics=('raw_covariance','adjusted_covariance','raw_correlation','adjusted_correlation')
    curves={d:{m:[] for m in metrics} for d in depths};diagnostics=[];sources=[];edge_error=0.
    for batch in batches:
        path=paths[int(batch)];com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        with open_trajectory(path) as tr:
            clock=np.round(np.asarray(tr['score_time'],float)[:,0],6);state=np.asarray(tr['state_time'],float)
            ids=np.flatnonzero(np.asarray(tr['resampled'],bool)&(clock>=ref['window'][0]-2e-6)&(state<=ref['window'][1]+2e-6))
            if not np.array_equal(clock[ids],grid) or np.any(np.diff(ids)!=1):raise ValueError('Exact contiguous observed support required')
            if not np.isclose(state[ids[-1]],ref['window'][1],atol=2e-6) or not np.asarray(tr['mask'])[ids].all():raise ValueError('Complete support and fixed active slots required')
            y=np.asarray(tr['predicted_coords'],float)[ids]*cfg['coord_scale']+com
            current=np.asarray(tr['current_coords'],float)[ids]*cfg['coord_scale']+com;proposal=np.asarray(tr['proposal_coords'],float)[ids]*cfg['coord_scale']+com
            score=np.asarray(tr['pic50_on'],float)[ids];selected=np.asarray(tr['selected_indices'])[ids]
            atom,bond=np.asarray(tr['predicted_atomics'])[ids],np.asarray(tr['predicted_bonds'])[ids]
            geometry=np.array([numpy_geometry(v,points,origin) for v in y])
            for k in range(1,len(grid)):
                error=float(np.abs(current[k]-proposal[k-1,selected[k-1]]).max());edge_error=max(error,edge_error)
                if error>2e-6:raise ValueError('Actual parent transport mismatch')
            for depth in depths:
                history={m:[] for m in metrics}
                for k in range(depth,len(grid)):
                    ancestors=ancestor_indices(selected,k,depth);parent=selected[k-1];older=y[k-depth,ancestors]
                    delta=y[k]-older;region,translation,membership=signed_regions(y[k],older,points,region_width)
                    internal=delta-translation[:,None]
                    amplitude=np.column_stack([np.sqrt((delta*delta).sum(-1).mean(-1)),np.linalg.norm(translation,axis=-1),np.sqrt((internal*internal).sum(-1).mean(-1))])
                    feature=np.column_stack([geometry[k]-geometry[k-depth,ancestors],amplitude,region.reshape(len(y[k]),-1)])
                    gain=score[k]-score[k-depth,ancestors];chem=chemistry_nuisance(atom[k],bond[k],atom[k-depth],bond[k-depth],ancestors)
                    control=chem if nuisance=='chemistry' else np.column_stack([chem,translation])
                    values,support=conditional_statistics(feature,gain,control,parent,ancestors)
                    for m in metrics:history[m].append(values[m])
                    alias=alias_diagnostics(y[k],score[k],parent)
                    diagnostics.append({'batch':int(batch),'depth':depth,'time':float(grid[k]),'state_time':float(state[ids[k]]),**support,
                      'regional_membership_mean':float(membership.mean()),'atom_change_fraction':float(chem[:,0].mean()),'bond_change_fraction':float(chem[:,1].mean()),
                      'immediate_coordinate_range_max_A':alias['immediate_coordinate_range_max_A'],'immediate_score_range_max':alias['immediate_score_range_max']})
                for m in metrics:curves[depth][m].append(history[m])
        frame_path=source/f'frame_batch_{batch:03d}.json'
        sources.append({'batch':int(batch),'path':path.relative_to(root).as_posix(),'sha256':digest(path),
          'frame_path':frame_path.relative_to(root).as_posix(),'frame_sha256':digest(frame_path)})
        print(f'multi-depth mutation batch {batch}: depths '+','.join(map(str,depths)),flush=True)
    out.mkdir(parents=True);rows=[];batch_rows=[];fits={};integrals={};rng=np.random.default_rng(seed)
    for depth in depths:
        times=grid[depth:];fits[str(depth)]={'times':times.tolist(),'first_missing_times':grid[:depth].tolist(),'functions':{}}
        integrals[depth]={}
        for metric in metrics:
            values=np.asarray(curves[depth][metric]);integrated=observed_integral(values,times);integrals[depth][metric]=integrated
            sample=integrated[rng.integers(0,len(batches),(2000,len(batches)))];boot=np.nanmean(sample,axis=1)
            for j,feature in enumerate(features):
                finite=np.isfinite(integrated[:,j]);n=int(finite.sum());vector=integrated[finite,j]
                if n>2 and np.ptp(vector)>1e-15:p=float(ttest_1samp(vector,0).pvalue)
                elif n>2 and np.all(vector==0):p=1.
                else:p=np.nan
                effect=float(np.nanmean(vector)) if n else np.nan;ci=np.nanquantile(boot[:,j],[.025,.975]) if np.isfinite(boot[:,j]).any() else [np.nan,np.nan]
                rows.append({'id':f'multi_depth/{depth}/{metric}/{feature}','depth':depth,'metric':metric,'feature':feature,'whole_window_mean':effect,
                  'CI_low':float(ci[0]),'CI_high':float(ci[1]),'p':p,'same_sign_batch_fraction':float(np.mean(np.sign(vector)==np.sign(effect))) if n else np.nan,
                  'independent_batches_available':n,'observed_event_fraction':float(np.isfinite(values[:,:,j]).mean())})
                for batch,value in zip(batches,integrated[:,j]):batch_rows.append({'batch':int(batch),'depth':depth,'metric':metric,'feature':feature,'whole_window_value':float(value)})
                complete=np.isfinite(values[:,:,j]).all(0)
                if complete.sum()<5:continue
                fit=masked_curve_fit(times[complete],values[:,complete,j],max_degree=3)
                fits[str(depth)]['functions'][metric+'/'+feature]={'degree':fit['degree'],'coefficients':fit['legendre_coefficients'],
                   'derivative_coefficients':(legder(np.array(fit['legendre_coefficients']))*2/(fit['time_end']-fit['time_start'])).tolist(),
                   'cv_mse':fit['cv_mse_by_degree'],'observed_time_start':fit['time_start'],'observed_time_end':fit['time_end']}
    q=bh(np.array([row['p'] for row in rows]))
    for row,value in zip(rows,q):row['q']=float(value)
    table=pd.DataFrame(rows);table.to_parquet(out/'whole_window_effects.parquet',compression=None,index=False)
    pd.DataFrame(batch_rows).to_parquet(out/'batch_window_statistics.parquet',compression=None,index=False)
    pd.DataFrame(diagnostics).to_parquet(out/'ancestry_diagnostics.parquet',compression=None,index=False)
    write_compact_json(out/'fitted_trends.json',{'schema_version':'multi-depth-global-functions-1.0','statistics_unit':'Independent batches, leave-one-batch-out one-SE degree selection',
      'derivative_semantics':'Time derivative of an observed population association, never a spatial gradient.','depths':fits})
    directions=[]
    for depth in depths:
        for region in range(len(points)):
            start=len(geometric)+3*region
            for metric in ('raw_covariance','adjusted_covariance'):
                vector=np.asarray(curves[depth][metric])[:,:,start:start+3];diag=cross_batch_directions(vector)
                integrated=integrals[depth][metric][:,start:start+3];valid=np.isfinite(integrated).all(1)
                mean=np.mean(integrated[valid],axis=0) if valid.any() else np.full(3,np.nan)
                matches=table[(table.depth==depth)&(table.metric==metric)&(table.feature.isin(regional[3*region:3*region+3]))]
                directions.append({'depth':depth,'region':region,'metric':metric,'xyz_covariance_mean':mean.tolist(),
                  'absolute_covariance_norm':float(np.linalg.norm(mean)),'component_q':matches.q.tolist(),**diag})
    write_compact_json(out/'regional_direction_coverage.json',{'schema_version':'multi-depth-region-directions-1.0','region_landmarks_A':points.tolist(),
      'units':'Å times recorded pIC50, ancestor-density-weighted internal movement','rows':directions})
    selected_evidence=table[table.metric=='adjusted_correlation'].sort_values(['q','depth','feature']).head(24).to_dict('records')
    write_json(out/'evidence.json',{'schema_version':'multi-depth-mutation-evidence-1.0','window':ref['window'],'depths':depths,'independent_batches':len(batches),
      'evidence_items':selected_evidence,'q_family_tests':int(table.p.notna().sum()),
      'q_below_005_by_depth':{str(d):{m:int(((table.depth==d)&(table.metric==m)&(table.q<.05)).sum()) for m in metrics} for d in depths},
      'nuisance':nuisance,'region_width_A':region_width,'limitations':['Larger depths supply more observed branches but also more prior selection and latent drift.',
       'Extinct-branch future remains censored. Only in-window recorded endpoint forecasts/head labels are used.',
       'Chemistry and pose adjustments are diagnostics, not graph restrictions or trained predictors.',
       'Regional XYZ is density-weighted internal movement in a fixed receptor frame, not an identified chemical mechanism.',
       'Batch-pair directional summaries are dependent descriptive observations. Missing directions never have cosine zero.',
       'All depths and features share one BH family; trends never fill missing first-depth nodes.']})
    write_json(out/'feature_catalog.json',{'features':features,'geometric_feature_count':len(geometric),'regional_signed_feature_count':len(regional),
      'landmarks_A':points.tolist(),'origin_A':origin.tolist(),'region_width_A':region_width,'regional_representation':'Ancestor Gaussian membership times endpoint physical-slot internal XYZ innovation, divided by active atom count.',
      'nuisance':['Atom-label change fraction','Unordered-bond-label change fraction']+(['Whole-cloud XYZ translation'] if nuisance=='chemistry_pose' else []),
      'units':'Distances/centroids/internal XYZ Å; shape Å²; densities dimensionless; covariance feature-unit times recorded pIC50.'})
    import evomolsteer.continuous.branch_mutation as branch
    code={Path(__file__).name:digest(__file__),'branch_mutation.py':digest(branch.__file__)}
    manifest={'schema_version':'multi-depth-mutation-mining-1.0','window':ref['window'],'score_times':grid.tolist(),'depths':depths,
      'sources':sources,'reference_sha256':digest(refpath),'discovery_batches':batches,'independent_batches':len(batches),
      'feature_count':len(features),'seed':seed,'region_width_A':region_width,'nuisance':nuisance,
      'maximum_parent_transport_error_A':edge_error,'code_sha256':code,
      'storage':'Compact batch-window moments, diagnostics, global functions and region summaries; no feature cache/full reference duplication.',
      'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'manifest.json',manifest)
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if size>2*1024*1024:raise ValueError(f'Compact multi-depth summary exceeded 2MiB: {size}')
    return manifest
