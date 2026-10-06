"""Streaming whole-window mining linking regional coordinate selection to affinity.

All descendants of one event parent count once in lag evidence. Independent
units for inference are generation batches, never time nodes or child copies.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from numpy.polynomial.legendre import legvander
from .coordinate_features import regional_observables
from .functional import quadrature
from ..io import read_json, write_json, write_table, digest
from ..storage.trajectory import TrajectoryPackage


def correlation(x, y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y[:,None]);n=valid.sum(0)
    mx=np.where(valid,x,0).sum(0)/np.maximum(n,1)
    my=np.where(valid,y[:,None],0).sum(0)/np.maximum(n,1)
    a=np.where(valid,x-mx,0);b=np.where(valid,y[:,None]-my,0)
    den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),den,out=np.full(x.shape[1],np.nan),where=(n>=8)&(den>1e-12))


def partial_correlation(x,y,nuisance):
    design=np.column_stack([np.ones(len(y)),nuisance])
    eligibility=np.isfinite(x)&(np.isfinite(y)&np.isfinite(design).all(1))[:,None]
    patterns,groups=np.unique(eligibility.T,axis=0,return_inverse=True)
    result=np.full(x.shape[1],np.nan)
    # Missing NOS must not drop valid all-atom candidates from unrelated tests.
    for group,finite in enumerate(patterns):
        columns=np.flatnonzero(groups==group)
        if finite.sum()<design.shape[1]+8:continue
        d=design[finite];xx=x[np.ix_(finite,columns)];yy=y[finite]
        result[columns]=correlation(xx-d@np.linalg.lstsq(d,xx,rcond=None)[0],
                                   yy-d@np.linalg.lstsq(d,yy,rcond=None)[0])
    return result


def mean(x,w):
    finite=np.isfinite(x);ww=finite*np.asarray(w)[:,None];den=ww.sum(0)
    return np.divide(np.where(finite,x,0).__mul__(ww).sum(0),den,
                     out=np.full(x.shape[1],np.nan),where=den>0)


def window_descendants(indices):
    copies=np.ones(indices.shape[1],int);rows=[]
    for selected in indices[::-1]:
        copies=np.bincount(selected,weights=copies,minlength=len(copies));rows.append(copies)
    return np.asarray(rows[::-1])


def lag_evidence(x,score,next_score,selected):
    count=np.bincount(selected,minlength=len(score));alive=count>0
    delta=np.bincount(selected,weights=next_score,minlength=len(score))/np.maximum(count,1)-score
    # Each retained parent is one observation; control the starting oracle score.
    raw=correlation(x[alive],delta[alive])
    adjusted=partial_correlation(x[alive],delta[alive],score[alive,None])
    return raw,adjusted,int(alive.sum())


def bh(p):
    p=np.asarray(p,float);result=np.full_like(p,np.nan);valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid],kind='stable')];n=len(order)
    if n:result[order]=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1].clip(0,1)
    return result


def curve_fit(times,curves,max_degree=5):
    """Leave-one-batch-out, one-SE global Legendre fit with analytic derivatives."""
    times=np.asarray(times);curves=np.asarray(curves);n=len(curves)
    w=quadrature(times);u=2*(times-times[0])/(times[-1]-times[0])-1
    basis=legvander(u,min(max_degree,len(times)-2));models=[];errors=[]
    for d in range(basis.shape[1]):
        b=basis[:,:d+1];coef=curves@(np.linalg.pinv(b*np.sqrt(w[:,None]))*np.sqrt(w)).T
        loo=(coef.sum(0)-coef)/(n-1);errors.append(((curves-loo@b.T)**2*w).sum(1));models.append(coef)
    errors=np.asarray(errors);loss=errors.mean(1);best=loss.argmin()
    limit=loss[best]+errors[best].std(ddof=1)/np.sqrt(n)
    chosen=np.flatnonzero(loss<=limit+1e-14)[0]
    return {'degree':int(chosen),'legendre_coefficients':models[chosen].mean(0).tolist(),
            'time_start':float(times[0]),'time_end':float(times[-1]),
            'cv_mse_by_degree':loss.tolist(),'n_independent_batches':n,
            'derivative_semantics':'temporal derivative of an ensemble curve, not a coordinate force'}


METRICS=('population_mean','selected_mean','selection_shift','retained_shift',
         'low_enrichment','high_enrichment','affinity_correlation','partial_affinity_correlation',
         'lag_gain_correlation','lag_partial_gain_correlation','native_difference')


def mine(dataset,campaign,analysis,output,batches=None,width=4.,spatial_anchor='current',regions=None,control_representation='current',feature_family='geometry'):
    root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    cfg=read_json(Path(analysis)/'config.json');catalog=read_json(Path(analysis)/'feature_catalog.json')
    if regions is not None:
        if not regions or set(regions)-set(catalog['regions']):raise ValueError('Unknown/empty region subset')
        catalog['regions']={r:catalog['regions'][r] for r in regions}
    if control_representation not in ('current','proposal'):raise ValueError('Unknown control representation')
    include_proposal=spatial_anchor=='endpoint' or control_representation=='proposal'
    scfg=read_json(source/'config.json');scale=scfg['coord_scale']
    splits={'discovery':cfg['discovery_batches'],'validation':cfg['validation_batches'],'heldout':cfg['heldout_batches']}
    selected_batches=sorted(set(batches if batches is not None else sum(splits.values(),[])))
    rows=[];audit=[];sources=[];grid=None
    for batch in selected_batches:
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None,:]
        path=source/'single'/f'batch_{batch:03d}'/'trajectory.h5'
        nativepath=source/'unguided'/f'batch_{batch:03d}'/'trajectory.h5'
        with TrajectoryPackage(path) as z, TrajectoryPackage(nativepath) as native:
            alltimes=z.read('score_time')[:,0].astype(float)
            if native.read('resampled').any():raise ValueError('Background has particle resampling')
            if not np.array_equal(alltimes,native.read('score_time')[:,0]):raise ValueError('Background time grid differs')
            ids=np.flatnonzero(z.read('resampled'))
            times=np.round(alltimes[ids],6)
            if len(times)<3 or np.any(np.diff(ids)!=1):raise ValueError('Contiguous actual selection window required')
            if grid is None:grid=times
            if not np.array_equal(times,grid):raise ValueError('Unequal actual selection grids')
            selected=z.read('selected_indices')[ids];copies=window_descendants(selected)
            if not np.all(copies.sum(1)==selected.shape[1]):raise ValueError('Ancestral mass not conserved')
            scores=z.read('pic50_on');prob=z.read('selection_probability')
            for k,i in enumerate(ids):
                t=float(times[k]);dt=float(z.read('step_size',int(i)).reshape(-1)[0])
                if t>=1 or dt<=0:raise ValueError('Invalid transport time')
                arrays=[z.read(f'{s}_coords',int(i)).astype(float)*scale+com for s in ('current','predicted','proposal')]
                atoms=z.read('predicted_atomics',int(i));mask=z.read('mask',int(i))
                x,names,metadata=regional_observables(*arrays,atoms,mask,catalog,t,dt,width,spatial_anchor,include_proposal,feature_family)
                n=len(x);p=prob[i].astype(float);p/=p.sum();mu=mean(x,np.ones(n));sel=mean(x,p)
                base=[native.read(f'{s}_coords',int(i)).astype(float)*scale+com for s in ('current','predicted','proposal')]
                nx,nn,_=regional_observables(*base,native.read('predicted_atomics',int(i)),native.read('mask',int(i)),catalog,t,dt,width,spatial_anchor,include_proposal,feature_family)
                if nn!=names:raise ValueError('Feature schema mismatch')
                qlow=np.nanquantile(x,.25,axis=0);qhigh=np.nanquantile(x,.75,axis=0)
                lo=np.where(np.isfinite(x),x<qlow,np.nan);hi=np.where(np.isfinite(x),x>qhigh,np.nan)
                center=arrays[0].mean(1);radius=np.sqrt(((arrays[0]-center[:,None])**2).sum(-1).mean(1))
                nos=np.isin(atoms,[catalog['atom_vocabulary'][a] for a in ('N','O','S')]).sum(1)
                # Composition and global pose controls are diagnostics, not causal identification.
                nuisance=np.column_stack([center,radius,nos])
                if feature_family=='shape':
                    # Adjust regional direction evidence for common whole-ligand
                    # second moments in the same measured representation. This
                    # is a confounding diagnostic, not causal identification.
                    global_columns=[];uniform=mask.astype(float)/mask.sum(1)[:,None]
                    for whole in (arrays[0],arrays[2]):
                        global_center=(uniform[...,None]*whole).sum(1);centered=whole-global_center[:,None]
                        tensor=np.einsum('bn,bni,bnj->bij',uniform,centered,centered)
                        global_columns.extend(tensor[:,i,j] for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)))
                    nuisance=np.column_stack([nuisance,*global_columns])
                lag,lagpartial,parents=np.full(len(names),np.nan),np.full(len(names),np.nan),0
                if k+1<len(ids):lag,lagpartial,parents=lag_evidence(x,scores[i],scores[ids[k+1]],selected[k])
                values={'population_mean':mu,'selected_mean':sel,'selection_shift':sel-mu,
                    'retained_shift':mean(x,copies[k])-mu,
                    'low_enrichment':mean(lo,p)-mean(lo,np.ones(n)),
                    'high_enrichment':mean(hi,p)-mean(hi,np.ones(n)),
                    'affinity_correlation':correlation(x,scores[i]),
                    'partial_affinity_correlation':partial_correlation(x,scores[i],nuisance),
                    'lag_gain_correlation':lag,'lag_partial_gain_correlation':lagpartial,
                    'native_difference':mu-mean(nx,np.ones(n))}
                rows.append(pd.DataFrame({'batch':batch,'time':t,'feature':names,'available_fraction':np.isfinite(x).mean(0),**values}))
                roots=z.read('root_slot',int(i));unique=np.unique(roots).size
                audit.append({'batch':batch,'time':t,'candidates':n,'unique_roots':unique,
                    'unique_retained_parents_lag':parents,'probability_ESS':float(1/(p*p).sum()),
                    'mean_affinity':float(scores[i].mean()),'selected_affinity':float(scores[i]@p)})
        sources.extend([{'path':p.relative_to(root).as_posix(),'sha256':digest(p)} for p in (path,nativepath)])
        print(f'coordinate batch {batch}: {len(times)} actual selection nodes',flush=True)
    out.mkdir(parents=True);table=pd.concat(rows,ignore_index=True)
    # Lossless float64 sufficient summaries; no per-node or per-edge feature expansion.
    table.to_parquet(out/'batch_coordinate_statistics.parquet',index=False,compression='zstd')
    write_table(out/'lineage_diagnostics.csv',audit)
    summarize(table,grid,splits,out)
    write_json(out/'feature_catalog.json',{'features':metadata,'regions':catalog['regions'],
        'atom_vocabulary':catalog['atom_vocabulary'],'spatial_width_A':width,'frame':'aligned PDB world angstrom','spatial_anchor':spatial_anchor})
    write_json(out/'manifest.json',{'schema_version':'coordinate-affinity-mining-1.0','window':[float(grid[0]),float(grid[-1])],
        'times':grid.tolist(),'splits':splits,'sources':sources,'dataset':str(root.resolve()),'campaign':campaign,
        'spatial_anchor':spatial_anchor,'control_representation':control_representation,
        'region_subset':list(catalog['regions']),'include_proposal_observables':include_proposal,
        'feature_family':feature_family,'core_diagnostic_radius_A':5.,
        'shape_partial_nuisance':'same-event partial correlation: global centroid/radius/NOS count plus masked whole-ligand current AND proposal second moments. Lag partial correlation still adjusts starting score only.' if feature_family=='shape' else None,
        'analysis_code_sha256':{p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('coordinate_features.py'))},
        'n_features':len(metadata),'n_statistic_rows':len(table),'storage':'lossless float64 zstd Parquet; no node feature cache',
        'interpretation':['Selection association is partly tautological: selection uses this affinity head.',
            'Lag associations condition on survival and starting score; observational, not causal.',
            'Batch is the independent unit; roots collapse within a batch.',
            'Endpoint transport is an endpoint residual proxy; native displacement includes SDE/corrector.',
            'NOS mask uses endpoint labels on current slots, not physical early chemistry.',
            'No regional physical energies are invented for noisy coordinates or unresolved graphs.'],
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(out.iterdir()) if p.is_file()}})
    return out


def summarize(table,grid,splits,out):
    out=Path(out);grid=np.asarray(grid,float);summary=[];fits={};effect_functions={}
    for split,bs in splits.items():
        d=table[table.batch.isin(bs)];batch_ids=sorted(d.batch.unique())
        if len(batch_ids)<2:continue
        for feature,g in d.groupby('feature',sort=True):
            for metric in METRICS:
                pivot=g.pivot(index='batch',columns='time',values=metric).reindex(index=batch_ids,columns=grid)
                # Last node has no future *within-window* observation; never extrapolate.
                eligible=grid[:-1] if metric.startswith('lag_') else grid
                curves=pivot.loc[:,eligible].to_numpy();w=quadrature(eligible)
                finite=np.isfinite(curves);coverage=finite@w
                # Late genealogical collapse makes lag effects unidentifiable.
                # Keep the available evidence with explicit coverage, never zero-fill effects.
                valid=coverage>=.25 if metric.startswith('lag_') else finite.all(1)
                curves=curves[valid];coverage=coverage[valid];finite=finite[valid]
                if len(curves)<2:continue
                a=(np.where(finite,curves,0)@w)/coverage;m=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(len(a)))
                p=stats.ttest_1samp(a,0).pvalue if a.std()>1e-14 else (1. if abs(m)<1e-14 else 0.)
                ci=stats.t.ppf(.975,len(a)-1)*se
                summary.append({'split':split,'feature':feature,'metric':metric,'window_mean':m,
                    'ci_low':m-ci,'ci_high':m+ci,'p':p,'n_batches':len(a),
                    'mean_time_coverage':float(coverage.mean()),
                    'start':float(curves[:,0].mean()),'end':float(curves[:,-1].mean()),
                    'end_minus_start':float((curves[:,-1]-curves[:,0]).mean()),
                    'mean_change_rate':float((curves[:,-1]-curves[:,0]).mean()/(eligible[-1]-eligible[0]))})
                if split=='discovery' and metric=='selected_mean':fits[feature]=curve_fit(eligible,curves)
                if split=='discovery' and metric in ('selection_shift','retained_shift','partial_affinity_correlation') and np.isfinite(curves).all():
                    effect_functions.setdefault(metric,{})[feature]=curve_fit(eligible,curves)
    summ=pd.DataFrame(summary)
    for _,g in summ.groupby(['split','metric']):summ.loc[g.index,'q']=bh(g.p.to_numpy())
    write_table(out/'whole_window_evidence.csv',summ)
    write_json(out/'continuous_functions.json',fits)
    write_json(out/'effect_functions.json',effect_functions)
    return summ
