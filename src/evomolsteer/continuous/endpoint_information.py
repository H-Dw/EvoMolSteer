"""Coordinate information audit, not an affinity predictor or reward evaluator.

Compare geometric nearest teachers under compressed fields and full point clouds.
Leave-one-batch teacher exclusion prevents a query selecting its own source.
Only batch/node summaries are saved; query distances are discarded immediately.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from .affinity_geometry import numpy_geometry
from ..generation.window_reference import load_reference
from ..storage.trajectory import TrajectoryPackage
from ..io import read_json,write_json,digest


def proper_rigid_residual(query,matched):
    """Best proper rotation for a fixed correspondence; no reflection or rematch."""
    a=query-query.mean(0);b=matched-matched.mean(0)
    u,_,vt=np.linalg.svd(a.T@b);correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    return float(np.sqrt(((a@rotation-b)**2).sum(1).mean()))


def full_distances(query,teachers):
    values=[];orders=[]
    for target in teachers:
        cost=((query[:,None]-target[None])**2).sum(-1);rows,cols=linear_sum_assignment(cost)
        values.append(np.sqrt(cost[rows,cols].mean()));orders.append(cols)
    return np.asarray(values),orders


def audit(reference,prior,dataset,campaign,output):
    ref=load_reference(reference);prior_data=read_json(prior);root=Path(dataset);source=root/'results'/campaign;out=Path(output)
    if out.exists():raise FileExistsError(out)
    if prior_data['reference_sha256']!=digest(reference):raise ValueError('Exact prior/reference binding required')
    cfg=read_json(source/'config.json');times=np.asarray(ref['times']);scale=np.asarray(ref['feature_scale']);points=np.asarray(ref['landmarks_A'])
    selected=np.asarray(prior_data['feature_weights'],float);blocks=[0,1,2,1];cuts=[0,3,9,14,len(selected)]
    weights=np.zeros_like(selected)
    for k,(a,b) in enumerate(zip(cuts[:-1],cuts[1:])):
        if selected[a:b].sum():weights[a:b]=blocks[k]*selected[a:b]/selected[a:b].sum()
    records=[]
    for item in ref['sources']:
        path=root/item['path']
        if digest(path)!=item['sha256']:raise ValueError('Immutable original source mismatch')
        batch=int(path.parent.name.split('_')[-1]);com=np.asarray(read_json(source/f'frame_batch_{batch:03d}.json')['target_com'])[:,None]
        with TrajectoryPackage(path) as tr:
            grid=np.round(tr.read('score_time')[:,0].astype(float),6);states=tr.read('state_time')
            ids=np.flatnonzero(tr.read('resampled')&(grid>=ref['window'][0]-1e-6)&(states<=ref['window'][1]+1e-6))
            if not np.array_equal(grid[ids],times):raise ValueError('Exact whole-window selection support required')
            for j,i in enumerate(ids):
                x=tr.read('predicted_coords',int(i)).astype(float)*cfg['coord_scale']+com
                observed=tr.read('pic50_on',int(i)).reshape(-1);frame=ref['frames'][j]
                teachers=np.asarray(frame['teacher_endpoint_A']);scores=np.asarray(frame['teacher_scores']);batches=np.asarray(frame['teacher_batches'])
                keep=batches!=batch;teachers=teachers[keep];scores=scores[keep]
                if len(teachers)<4:raise ValueError('Leave-one-batch teacher support insufficient')
                z=numpy_geometry(x,points,ref['origin_A'])/scale
                targets=numpy_geometry(teachers,points,ref['origin_A'])/scale
                variance=np.asarray(frame['variance_scaled']).mean(0)
                effect=np.array([prior_data['node_effect_functions'][f]['values_z'][j] if w else 0.
                                 for f,w in zip(ref['features'],weights)])
                salience=weights*np.clip(np.abs(effect),.1,2.)
                q=((z[:,None]-targets[None])**2/variance*salience).sum(2)
                excess=[];overlap=[];rigid=[];chosen_score=[];full_score=[];fixed=[]
                for n,cloud in enumerate(x):
                    distances,orders=full_distances(cloud,teachers)
                    fp=int(np.argmin(q[n]));pc=int(np.argmin(distances))
                    excess.append(distances[fp]-distances[pc]);fixed.append(distances[fp])
                    rigid.append(proper_rigid_residual(cloud,teachers[fp,orders[fp]]))
                    overlap.append(len(set(np.argsort(q[n],kind='stable')[:4])&set(np.argsort(distances,kind='stable')[:4]))/4)
                    chosen_score.append(scores[fp]);full_score.append(scores[pc])
                correlation=lambda labels:float(spearmanr(observed,labels).statistic) if np.std(labels)>1e-12 and np.std(observed)>1e-12 else 0.
                records.append({'batch':batch,'score_time':float(times[j]),'n_queries':len(x),'n_teachers':len(teachers),
                    'compressed_choice_excess_fixed_RMS_A':float(np.mean(excess)),
                    'compressed_choice_fixed_RMS_A':float(np.mean(fixed)),
                    'compressed_choice_proper_rigid_RMS_A':float(np.mean(rigid)),
                    'top4_teacher_overlap':float(np.mean(overlap)),
                    'compressed_chosen_teacher_score_correlation':correlation(chosen_score),
                    'full_chosen_teacher_score_correlation':correlation(full_score)})
    table=pd.DataFrame(records);metrics=[c for c in table if c not in ('batch','score_time','n_queries','n_teachers')]
    summaries=[];rng=np.random.default_rng(42)
    for field in metrics:
        values=np.array([np.trapezoid(g[field],g.score_time)/(times[-1]-times[0]) for _,g in table.groupby('batch')])
        boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
        summaries.append({'metric':field,'whole_window_mean':float(values.mean()),'batch_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist()})
    out.mkdir(parents=True);table.to_parquet(out/'batch_node_information.parquet',compression=None,index=False)
    summary={'window':ref['window'],'times':ref['times'],'independent_batches':len(ref['sources']),'metrics':summaries,
        'reference_sha256':digest(reference),'prior_sha256':digest(prior),'source_code_sha256':digest(__file__),'sources':ref['sources'],
        'comparison':'Geometric nearest teacher under sparse salient fields versus Hungarian fixed-frame RMS; leave-one-batch exclusion',
        'limitations':['Not the full score-weighted reward or its FLOWR Jacobian',
            'Nonnegative excess distance is by construction, not a causal affinity effect',
            'Proper rigid alignment retains fixed Hungarian correspondence; not a globally solved rotation/permutation distance',
            'Node summaries describe correlated within-batch candidates; uncertainty uses independent batches',
            'Teacher score correlations are observational and are not a fitted or independently validated affinity predictor']}
    write_json(out/'information_summary.json',summary)
    return summary
