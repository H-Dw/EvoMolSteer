"""Retrospective motif enrichment at the ACTUAL selection-window boundary.

Streaming candidate arrays are never persisted. Independent units are batches,
not clones, roots or time nodes. No terminal t=1 outcomes are read.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from ..io import read_json,digest
from ..generation.prototypes import write_json
from ..storage.trajectory import TrajectoryPackage
from .spatial_motifs import numpy_motifs
from .coordinate_mining import boundary_descendants,mean,bh
from .functional import quadrature,global_fit

METRICS=('retained_shift','root_balanced_retained_shift','low_enrichment','high_enrichment')


def mine(dataset,campaign,mining,output,channels=('all','NOS_C')):
    root,mining,out=map(Path,(dataset,mining,output))
    if out.exists():raise FileExistsError(out)
    manifest=read_json(mining/'manifest.json');cat=read_json(mining/'feature_catalog.json')
    if manifest['feature_family']!='motif' or manifest['control_representation']!='proposal' or manifest['spatial_anchor']!='endpoint':raise ValueError('Matched proposal motif evidence required')
    source=root/'results'/campaign;cfg=read_json(source/'config.json');rows=[];audit=[];sources=[];grid=None
    for batch in manifest['splits']['discovery']:
        path=source/'single'/f'batch_{batch:03d}/trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as z:
            ids,copies=boundary_descendants(z.read('resampled'),z.read('state_time'),z.read('selected_indices'),manifest['window'])
            times=np.round(z.read('score_time')[ids,0].astype(float),6)
            if grid is None:grid=times
            if not np.array_equal(grid,times):raise ValueError('Unequal learned grids')
            for k,i in enumerate(ids):
                x=z.read('proposal_coords',int(i)).astype(float)*cfg['coord_scale']+com
                a=z.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                v,names,_=numpy_motifs(x,z.read('predicted_atomics',int(i)),z.read('mask',int(i)),cat,a,cat['spatial_width_A'],channels)
                n=len(v);p=copies[k].astype(float);_,group,count=np.unique(z.read('root_slot',int(i)),return_inverse=True,return_counts=True)
                rootmass=np.bincount(group,weights=p);balanced=p/rootmass[group].clip(1e-30)
                prior=(1/count[group])*(rootmass[group]>0)
                lo=np.where(np.isfinite(v),v<np.nanquantile(v,.25,axis=0),np.nan)
                hi=np.where(np.isfinite(v),v>np.nanquantile(v,.75,axis=0),np.nan)
                values={'retained_shift':mean(v,p)-mean(v,np.ones(n)),
                        'root_balanced_retained_shift':mean(v,balanced)-mean(v,prior),
                        'low_enrichment':mean(lo,p)-mean(lo,np.ones(n)),
                        'high_enrichment':mean(hi,p)-mean(hi,np.ones(n))}
                rows.append(pd.DataFrame({'batch':batch,'time':float(times[k]),'state_time':float(z.read('state_time',int(i))),
                    'feature':[name.replace('::motif_','::proposal_motif_') for name in names],**values}))
                audit.append({'batch':batch,'time':float(times[k]),'candidates':n,'surviving_candidates':int((p>0).sum()),
                    'retained_weight_ESS':float(p.sum()**2/(p@p)),'extant_roots':len(count),'surviving_roots':int((rootmass>0).sum())})
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    out.mkdir(parents=True);table=pd.concat(rows,ignore_index=True);table.to_parquet(out/'batch_statistics.parquet',index=False,compression='zstd')
    pd.DataFrame(audit).to_csv(out/'lineage.csv',index=False,float_format='%.17g')
    records=[];functions={};w=quadrature(grid)
    for feature,g in table.groupby('feature',sort=True):
        for metric in METRICS:
            curves=g.pivot(index='batch',columns='time',values=metric).reindex(columns=grid).to_numpy()
            valid=np.isfinite(curves).all(1);curves=curves[valid]
            if len(curves)<2:continue
            values=curves@w;effect=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)));ci=stats.t.ppf(.975,len(values)-1)*se
            p=float(stats.ttest_1samp(values,0).pvalue) if values.std()>1e-14 else (1. if abs(effect)<1e-14 else 0.)
            fit=global_fit(grid,curves,{'seed':42,'permutations':4096,'max_degree':5})
            records.append({'feature':feature,'metric':metric,'window_mean':effect,'ci_low':effect-ci,'ci_high':effect+ci,'p':p,
                'global_curve_p':fit['global_curve_p'],
                'n_batches':len(curves),'start':float(curves[:,0].mean()),'end':float(curves[:,-1].mean()),
                'mean_change_rate':float((curves[:,-1]-curves[:,0]).mean()/(grid[-1]-grid[0]))})
            fit.pop('batch_coefficients')
            functions.setdefault(metric,{})[feature]=fit
    evidence=pd.DataFrame(records)
    for _,g in evidence.groupby('metric'):
        evidence.loc[g.index,'q']=bh(g.p.to_numpy())
        evidence.loc[g.index,'global_curve_q']=bh(g.global_curve_p.to_numpy())
    evidence.to_csv(out/'whole_window_evidence.csv',index=False,float_format='%.17g');write_json(out/'effect_functions.json',functions)
    write_json(out/'manifest.json',{'schema_version':'boundary-motif-enrichment-1.0','window':manifest['window'],
        'observed_score_times':grid.tolist(),'observed_proposal_boundary':float(table.state_time.max()),'channels':list(channels),
        'target':'Pre-selection candidate mass ancestral to the last actual selection state <= learned end; no outside proposal or terminal outcome',
        'independent_unit':'Discovery batch; descendant counts are weights, not replicates',
        'root_diagnostic':'Equal surviving root weighting; conditions on survival, not causal identification',
        'enrichment':'Weighted quartile-tail probability minus same-event unweighted probability; ties retain strict inequality',
        'function_policy':'Whole-window LOBO one-SE Legendre effect curves, temporal derivatives and batch-bootstrap intervals. Separate whole-curve sign-flip test avoids cancellation of opposite signed effects. Signed effects are not physical feature targets.',
        'inference_settings':{'seed':42,'permutations':4096,'max_degree':5,'FDR_family':'metric across all tested features'},
        'source_manifest_sha256':digest(mining/'manifest.json'),'sources':sources,
        'code_sha256':{p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('coordinate_mining.py'),Path(__file__).with_name('spatial_motifs.py'))},
        'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}})
    return out
