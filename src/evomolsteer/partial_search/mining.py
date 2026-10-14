"""Candidate-centred geometry contrasts without fitting an affinity predictor."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.storage.trajectory import TrajectoryPackage
from evomolsteer.continuous.elite_path_credit import terminal_ancestors


def match(cloud,target):
    cost=np.square(cloud[:,None]-target[None]).sum(-1)
    rows,cols=linear_sum_assignment(cost)
    assignment=rows[np.argsort(cols)]
    return float(np.sqrt(cost[assignment,np.arange(len(target))].mean())),assignment


def residual_ranks(values,global_values,batch):
    x=rankdata(values);g=rankdata(global_values)
    for b in np.unique(batch):
        take=batch==b;x[take]-=x[take].mean();g[take]-=g[take].mean()
    if np.dot(g,g)>1e-12:x-=g*np.dot(g,x)/np.dot(g,g)
    return x


def correlation(x,y):
    x=x-x.mean();y=y-y.mean();d=np.sqrt(np.dot(x,x)*np.dot(y,y))
    return float(np.dot(x,y)/d) if d>1e-12 else None


def cluster_interval(x,y,groups,repeats=400):
    keys=np.unique(groups);stats=[]
    for key in keys:
        take=groups==key;a=x[take];b=y[take]
        stats.append([len(a),a.sum(),b.sum(),np.dot(a,a),np.dot(b,b),np.dot(a,b)])
    stats=np.asarray(stats,float);rng=np.random.default_rng(42)
    sums=stats[rng.integers(0,len(keys),(repeats,len(keys)))].sum(1)
    n,sx,sy,sxx,syy,sxy=sums.T
    denominator=np.sqrt(np.maximum(sxx-sx*sx/n,0)*np.maximum(syy-sy*sy/n,0))
    valid=denominator>1e-12;values=(sxy-sx*sy/n)[valid]/denominator[valid]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]


def regional_sets(x,sizes):
    # All teacher atom slots can be covered; no receptor-landmark shortlist.
    distance=np.square(x[:,None]-x[None]).sum(-1);seen={}
    for center in range(len(x)):
        order=np.argsort(distance[center],kind='stable')
        for size in sizes:
            if not 1<=size<len(x):continue
            slots=tuple(sorted(order[:size].tolist()))
            seen.setdefault(slots,{'center_slots':[],'size':size})['center_slots'].append(center)
    return [{'region_id':f'P{i:03d}','atom_slots':list(slots),**info}
        for i,(slots,info) in enumerate(sorted(seen.items(),key=lambda item:(len(item[0]),item[0])))]


def build(repo,cfg):
    repo=Path(repo);base=repo/cfg['base_study'];out=repo/cfg['campaign_study'];out.mkdir(parents=True,exist_ok=True)
    table=pd.read_parquet(base/'ranked_finals.parquet');arrays=np.load(base/'terminal_fields.npz',allow_pickle=False)
    index=int(np.flatnonzero(table['candidate_id'].eq(cfg['candidate_id']).to_numpy())[0]);x=arrays['coords_A'].astype(float)
    teacher=x[index];rest=np.arange(len(table))!=index;other=x[rest];labels=table.loc[rest,'pic50_on_rescore'].to_numpy(float)
    batch=table.loc[rest,'batch'].to_numpy(int);valid=table.loc[rest,'valid_connected'].to_numpy(bool)
    root_ids={};source=repo/cfg['compact_dataset']/'results'/cfg['campaign']/cfg['arm']
    for b in table.batch.unique():
        with TrajectoryPackage(source/f'batch_{b:03d}'/'trajectory.h5') as tr:roots=terminal_ancestors(tr.read('selected_indices').astype(int))[0]
        for slot,root in enumerate(roots):root_ids[(int(b),slot)]=int(b)*100000+int(root)
    groups=np.array([root_ids[(int(r.batch),int(r.slot))] for r in table.loc[rest].itertuples()])
    global_error=[];point_error=[]
    for cloud in other:
        error,assignment=match(cloud,teacher);global_error.append(error);point_error.append(np.linalg.norm(cloud[assignment]-teacher,axis=-1))
    global_error=np.asarray(global_error);point_error=np.asarray(point_error)
    rank_y=residual_ranks(labels,-global_error,batch)
    protein=repo/cfg['dataset']/'inputs'/cfg['protein'];pocket=[]
    for line in protein.read_text().splitlines():
        if line[:6].strip()=='ATOM' and line[76:78].strip()!='H':pocket.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    distances=np.linalg.norm(teacher[:,None]-np.asarray(pocket)[None],axis=-1)
    point_rows=[]
    for slot in range(len(teacher)):
        rank_x=residual_ranks(-point_error[:,slot],-global_error,batch)
        point_rows.append({'slot':slot,'evidence_id':f'point/{slot}','nearest_protein_A':float(distances[slot].min()),
            'protein_atoms_within_6A':int((distances[slot]<6).sum()),'rest_match_RMS_A':float(np.sqrt(np.square(point_error[:,slot]).mean())),
            'incremental_rank_correlation':correlation(rank_x,rank_y),'cluster95':cluster_interval(rank_x,rank_y,groups)})
    regions=regional_sets(teacher,cfg['region_sizes'])
    previous=read_json(base/'candidates'/cfg['candidate_id']/'Designer.json')['atom_slots']
    if not any(r['atom_slots']==previous for r in regions):regions.append({'region_id':f'P{len(regions):03d}',
        'atom_slots':previous,'center_slots':[],'size':len(previous),'source':'frozen_previous_partial_selection'})
    errors=[];region_rows=[]
    for region in regions:
        ids=region['atom_slots'];target=teacher[ids];error=np.asarray([match(cloud,target)[0] for cloud in other]);errors.append(error)
        rx=residual_ranks(-error,-global_error,batch);near=np.argsort(error,kind='stable')[:cfg['hard_negative_count']]
        own_batch=batch==int(table.iloc[index]['batch']);independent=~own_batch
        high=labels>=np.quantile(labels,.8);eigen=np.linalg.eigvalsh(np.cov(target.T,bias=True))
        region_rows.append({**region,'evidence_id':region['region_id']+'/geometry','teacher_Rg_A':float(np.sqrt(np.square(target-target.mean(0)).sum(1).mean())),
            'teacher_cov_eigenvalues_A2':eigen.tolist(),'rest_match_RMS_median_A':float(np.median(error)),
            'candidate_match_RMS_A':0.,'one_vs_rest_difference_A':-float(error.mean()),
            'final_affinity_rank_correlation':correlation(rankdata(-error),rankdata(labels)),
            'incremental_rank_correlation':correlation(rx,rank_y),'incremental_cluster95':cluster_interval(rx,rank_y,groups),
            'valid_rest_incremental_correlation':correlation(rx[valid],rank_y[valid]),
            'exclude_teacher_batch_incremental_correlation':correlation(rx[independent],rank_y[independent]),
            'high_rest_error_median_A':float(np.median(error[high])),'lower_rest_error_median_A':float(np.median(error[~high])),
            'nearest_rest_pic50_mean':float(labels[near].mean()),'nearest_rest_pic50_min':float(labels[near].min()),
            'nearest_protein_A_mean':float(distances[ids].min(1).mean()),'protein_atoms_within_6A_mean':float((distances[ids]<6).sum(1).mean()),
            'free_teacher_slots':len(teacher)-len(ids)})
    np.savez_compressed(out/'geometry_cache.npz',region_errors_A=np.stack(errors),point_errors_A=point_error,
        global_errors_A=global_error,labels=labels,batch=batch,root_group=groups)
    packet={'candidate':table.iloc[index].to_dict(),'representation':'terminal_world_A','terminal_time':1.,
        'terminal_coords_A':teacher.tolist(),'comparison':{'one_candidate_vs_rest':len(other),'valid_rest':int(valid.sum()),
            'ancestral_root_clusters':len(np.unique(groups)),'region_count':len(region_rows),'all_teacher_slots_covered':True},
        'regions':region_rows,'point_evidence':point_rows,
        'definitions':{'match':'Unaligned receptor-frame Hungarian spatial matching; untyped coordinates',
            'incremental_rank_correlation':'Rank residuals after within-batch centering and removing overall-cloud similarity; association, not causal effect',
            'cluster95':'Bootstrap correlation of fixed rank residuals with whole initial-lineage roots as resampling clusters',
            'nearest_rest':'Closest historical finals in this region are actual counterexamples, not native-rollout utility labels'},
        'limitations':['Candidate itself excluded from all associations to avoid trivial self-fit bias',
            'Final head predictions are labels, not experimental affinities','No new affinity predictor is fitted',
            'No energy or typed-contact interpretation from coordinate proximity','Historical SMC dependence and unmeasured chemistry remain'],
        'source_sha256':{'final_fields':digest(base/'terminal_fields.npz'),'final_metrics':digest(base/'ranked_finals.parquet')}}
    write_json(out/'evidence.json',packet);return packet
