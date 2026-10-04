"""PCA fit exclusively on time-matched discovery background inside selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .scope import load
from .statistics import EVENT_KEYS,event_frame,selection_moments,save_metrics,lineage_intervals
from .io import write_json,write_table
from .output_policy import policy
from .diagnostics import ArrayAudit


def run(analysis,split='discovery'):
    analysis = Path(analysis)
    df,cfg,cat,scope = load(analysis,split)
    reference = df if split=='discovery' else load(analysis,'discovery')[0]
    models,loadings,scores,explained = {},[],[],[]
    for rep in cfg['analysis_representations']:
        ref = reference[(reference.arm==scope['background_arm'])&(reference.representation==rep)]
        features = list(cat['features'])
        columns = [f for f in features if ref[f].notna().mean()>=cfg['minimum_feature_fraction'] and ref[f].std()>1e-10]
        if not columns:
            raise ValueError('No finite variable features for PCA')
        median = ref[columns].median().to_numpy()
        x = ref[columns].to_numpy()
        x = np.where(np.isfinite(x),x,median)
        mean,scale = x.mean(0),x.std(0)
        scale = np.where(scale>1e-12,scale,1)
        xx = (x-mean)/scale
        with threadpool_limits(limits=1):
            eigen,vectors = np.linalg.eigh(xx.T@xx/max(len(xx)-1,1))
        order = np.argsort(-eigen,kind='stable')
        eigen,vectors = np.maximum(eigen[order],0),vectors[:,order]
        k = min(cfg['pca_components'],len(columns))
        basis = vectors[:,:k]
        for j in range(k):
            if basis[np.argmax(np.abs(basis[:,j])),j]<0:
                basis[:,j] *= -1
            explained.append({'representation':rep,'component':f'PC{j+1}','explained_variance_ratio':eigen[j]/eigen.sum()})
            loadings.extend({'representation':rep,'component':f'PC{j+1}','feature':f,'loading':basis[i,j]} for i,f in enumerate(columns))
        models[rep] = {'features':columns,'median':median,'mean':mean,'scale':scale,'components':basis,
            'fit_split':'discovery','fit_arm':scope['background_arm'],'n_fit':len(ref),
            'fit_score_times':scope['score_times'],'fit_window':[scope['window_start'],scope['window_end']],
            'sign_rule':'largest absolute loading positive'}
        g = df[df.representation==rep]
        v = g[columns].to_numpy()
        projection = (np.where(np.isfinite(v),v,median)-mean)/scale@basis
        meta = ['node_id','parent_node_id','representation','arm','batch','step','stage','score_time','geometry_time',
                'dt','resampled','probability','offspring_count']
        scores.append(pd.concat([g[meta].reset_index(drop=True),pd.DataFrame(projection,columns=[f'PC{i+1}' for i in range(k)])],axis=1))
    scores = pd.concat(scores,ignore_index=True)
    pcs = [c for c in scores if c.startswith('PC')]
    rows = []
    for _,g in scores[scores.resampled].groupby(EVENT_KEYS,sort=True):
        m = selection_moments(g[pcs].to_numpy(),g.probability,g.offspring_count)
        rows.append(event_frame(g,pcs,population_centroid=m['population_mean'],
            expected_selection_shift=m['expected_selection_shift'],realized_selection_shift=m['realized_selection_shift']))
    dest = analysis/split/'pca'
    save_metrics(dest,pd.concat(rows,ignore_index=True),['expected_selection_shift','realized_selection_shift'],cfg)
    retain = policy(cfg).write_node_details
    if retain:
        write_table(dest/'scores.parquet',scores)
    write_json(dest/'basis.json',models)
    write_table(dest/'loadings.csv',loadings)
    write_table(dest/'explained_variance.csv',explained)
    velocities = []
    audit = ArrayAudit(pcs)
    for rep,g in scores[scores.resampled].groupby('representation',sort=True):
        idx = g.set_index('node_id')
        child = g[g.parent_node_id.isin(idx.index)]
        parent = idx.loc[child.parent_node_id]
        dt = lineage_intervals(parent.geometry_time,child.geometry_time)
        delta = (child[pcs].to_numpy()-parent[pcs].to_numpy())/dt[:,None]
        audit.update(rep,delta)
        if retain:
            meta = child[['node_id','parent_node_id','arm','batch','stage','step']].reset_index(drop=True)
            meta['representation'] = rep;meta['geometry_delta_time'] = dt
            velocities.append(pd.concat([meta,pd.DataFrame(delta,columns=pcs)],axis=1))
    if retain:
        write_table(dest/'parent_child_velocity.parquet',pd.concat(velocities,ignore_index=True))
    audit.save(dest/'parent_child_diagnostics.json')
    return explained
