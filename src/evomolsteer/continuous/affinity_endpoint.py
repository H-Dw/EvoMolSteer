"""Offline endpoint geometry associated with recorded joint-model head labels.

This is a separate representation audit, not a fitted affinity predictor. The
original actual-proposal library and its statistical results remain immutable.
"""
import gzip,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,ttest_1samp
from .affinity_geometry import landmarks,names,numpy_geometry
from .coordinate_mining import bh,curve_fit
from ..io import read_json,write_json,digest
from ..storage.trajectory import TrajectoryPackage


def summarize(times,high,low,variance,correlation,features,seed=42):
    """Equal independent batches; arrays B,T,F, never clone-level inference."""
    grid=np.asarray(times,float);hi=np.asarray(high);lo=np.asarray(low);var=np.asarray(variance)
    if hi.shape!=lo.shape or hi.shape!=var.shape or hi.ndim!=3 or hi.shape[1]!=len(grid):
        raise ValueError('Batch/time/feature arrays must align')
    scale=np.sqrt(np.trapezoid(var.mean(0),grid,axis=0)/(grid[-1]-grid[0])).clip(1e-5)
    delta=(hi-lo)/scale
    integrated=np.trapezoid(delta,grid,axis=1)/(grid[-1]-grid[0])
    corr=np.trapezoid(correlation,grid,axis=1)/(grid[-1]-grid[0])
    rng=np.random.default_rng(seed);boot=integrated[rng.integers(0,len(hi),(2000,len(hi)))].mean(1)
    rows=[];functions={}
    for j,name in enumerate(features):
        ci=np.quantile(boot[:,j],[.025,.975]);p=float(ttest_1samp(integrated[:,j],0).pvalue)
        rows.append({'feature':name,'high_low_integrated_z':float(integrated[:,j].mean()),
            'CI_low':float(ci[0]),'CI_high':float(ci[1]),'score_correlation_integrated':float(corr[:,j].mean()),
            'positive_batch_fraction':float((integrated[:,j]>0).mean()),
            'time_last_minus_first_delta_z':float((delta[:,-1,j]-delta[:,0,j]).mean()),'p':p})
        functions[name]=curve_fit(grid,delta[:,:,j],max_degree=3)
    for r,q in zip(rows,bh(np.nan_to_num([r['p'] for r in rows],nan=1.))):r['q']=float(q)
    return scale,rows,functions


def mine_endpoint(dataset,campaign,output,batches=range(14),window=(0.,.5),coordinate_library=None):
    root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    batches=list(batches);cfg=read_json(source/'config.json')
    from ..trajectory_source import pocket_input_path
    points,origin=landmarks(pocket_input_path(root,'target_protein'),pocket_input_path(root,'target_ligand'))
    features=names(points);nodes={};sources=[];representations=[]
    for b in batches:
        path=source/'single'/f'batch_{b:03d}'/'trajectory.h5'
        com=np.asarray(read_json(source/f'frame_batch_{b:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as tr:
            times=np.round(tr.read('score_time')[:,0].astype(float),6);states=tr.read('state_time')
            ids=np.flatnonzero(tr.read('resampled')&(times>=window[0]-1e-6)&(states<=window[1]+1e-6))
            for i in ids:
                t=float(times[i]);y=tr.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                x=tr.read('proposal_coords',int(i)).astype(float)*cfg['coord_scale']+com
                score=tr.read('pic50_on',int(i)).astype(float).reshape(-1)
                if not tr.read('mask',int(i)).all():raise ValueError('Fixed active slots required')
                v=numpy_geometry(y,points,origin);lo,hi=np.quantile(score,[.25,.75]);low=score<=lo;high=score>=hi
                if min(low.sum(),high.sum())<3:raise ValueError('Insufficient within-event strata')
                nodes.setdefault(t,[]).append({'batch':b,'high':v[high].mean(0),'low':v[low].mean(0),'variance':v.var(0),
                    'correlation':[float(spearmanr(v[:,j],score).statistic) if v[:,j].std()>1e-12 else 0. for j in range(v.shape[1])],
                    'score_gap':float(score[high].mean()-score[low].mean())})
                rg=lambda cloud:np.sqrt(((cloud-cloud.mean(1)[:,None])**2).sum(2).mean(1))
                representations.append({'batch':b,'score_time':t,'proposal_state_time':float(states[i]),
                    'proposal_Rg_A_mean':float(rg(x).mean()),'endpoint_Rg_A_mean':float(rg(y).mean()),
                    'slot_endpoint_proposal_RMS_A_mean':float(np.sqrt(((x-y)**2).sum(2).mean(1)).mean())})
        sources.append({'path':path.relative_to(root).as_posix(),'sha256':digest(path)})
    grid=np.array(sorted(nodes));groups=[nodes[t] for t in grid]
    if len(grid)<2 or not np.isclose(grid[0],window[0]) or not np.isclose(grid[-1]+np.median(np.diff(grid)),window[1]):
        raise ValueError('Full actual selection support required')
    array=lambda key:np.array([[r[key] for r in group] for group in groups]).transpose(1,0,2)
    scale,rows,functions=summarize(grid,array('high'),array('low'),array('variance'),array('correlation'),features)
    # Reuse immutable high-score endpoints and exact source identities; no need
    # to copy current proposals into this endpoint-only library.
    frames=[{'time':float(t),'high_scaled':(np.array([r['high'] for r in group])/scale).tolist(),
             'low_scaled':(np.array([r['low'] for r in group])/scale).tolist(),
             'variance_scaled':(np.array([r['variance'] for r in group])/scale**2+.05).tolist(),
             'score_gap_mean':float(np.mean([r['score_gap'] for r in group]))}
            for t,group in zip(grid,groups)]
    teacher_source=None
    if coordinate_library is not None:
        from ..generation.window_reference import load_reference
        original=load_reference(coordinate_library)
        if original['window']!=list(window) or original['times']!=grid.tolist() or original['sources']!=sources:
            raise ValueError('Endpoint teachers must use identical selection nodes and source data')
        for frame,recorded in zip(frames,original['frames']):
            frame.update(teacher_endpoint_A=recorded['teacher_endpoint_A'],teacher_scores=recorded['teacher_scores'],
                         teacher_batches=recorded['teacher_batches'])
        teacher_source={'sha256':digest(coordinate_library),'role':'Recorded elite endpoint point clouds, no fitted model'}
    from ..trajectory_source import pocket_input_hashes
    ref={'schema_version':'affinity-endpoint-library-1.0','window':list(window),'times':grid.tolist(),'frames':frames,
        'landmarks_A':points.tolist(),'origin_A':origin.tolist(),'features':features,'feature_scale':scale.tolist(),
        'control_representation':'proposal','spatial_anchor':'endpoint','representation':'native model endpoint world coordinates',
        'required_input_sha256':pocket_input_hashes(root),'sources':sources,'discovery_batches':batches,
        'teacher_library':teacher_source,
        'label_semantics':'Recorded joint forward head label, not an independent endpoint-coordinate affinity rescore',
        'time_alignment':'Original score t with actual proposal state t+dt inside learned support; reward forecast at t',
        'limitations':['Observational conditional high/low contrast','No endpoint-only affinity model is fitted',
            'Endpoint reward requires real FLOWR coordinate pullback, not identity gradient']}
    out.mkdir(parents=True);path=out/'endpoint_reference.json.gz'
    path.write_bytes(gzip.compress(json.dumps(ref,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    pd.DataFrame(rows).to_parquet(out/'whole_window_effects.parquet',compression=None,index=False)
    pd.DataFrame(representations).to_parquet(out/'representation_audit.parquet',compression=None,index=False)
    write_json(out/'effect_functions.json',functions)
    write_json(out/'manifest.json',{'schema_version':'endpoint-coordinate-mining-1.0','features':len(features),'nodes':len(grid),
        'window':list(window),'sources':sources,'reference_sha256':digest(path),'reference_bytes':path.stat().st_size,
        'q_below_005':sum(r['q']<.05 for r in rows),'minimum_q':min(r['q'] for r in rows),
        'scale_floor_count':int((scale<=1e-5).sum()),'statistics_unit':'Equal batches, no clone-level significance',
        'storage':'Compact whole-window effects and representation audit; no per-particle feature cache'})
    return ref
