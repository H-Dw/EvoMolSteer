"""Expected preference, realized replication, and changes within selection paths."""
from pathlib import Path
import numpy as np
import pandas as pd
from .scope import load
from .statistics import EVENT_KEYS,event_frame,weighted_mean,selection_moments,save_metrics,lineage_intervals
from .io import write_table
from .output_policy import policy
from .diagnostics import ArrayAudit


def run(analysis,split='discovery'):
    analysis = Path(analysis)
    df,cfg,cat,_ = load(analysis,split)
    features = list(cat['features']);rows = []
    for _,g in df[df.resampled].groupby(EVENT_KEYS,sort=True):
        x = g[features].to_numpy(float)
        m = selection_moments(x,g.probability,g.offspring_count,cfg['minimum_feature_fraction'])
        for name,weight in [('on_score_preference_shift','weight_on'),('off_score_preference_shift','weight_off')]:
            w = g[weight].to_numpy(float,copy=True);w /= w.sum()
            mean,mass = weighted_mean(x,w)
            m[name] = np.where((m['valid_fraction']>=cfg['minimum_feature_fraction'])&
                               (mass>=cfg['minimum_feature_fraction']),mean-m['population_mean'],np.nan)
        m['expected_shift_per_dt'] = m['expected_selection_shift']/float(g.dt.iloc[0])
        m['realized_shift_per_dt'] = m['realized_selection_shift']/float(g.dt.iloc[0])
        rows.append(event_frame(g,features,**m))
    dest = analysis/split/'trends'
    result = save_metrics(dest,pd.concat(rows,ignore_index=True),
        ['expected_selection_shift','realized_selection_shift','selection_noise_shift',
         'on_score_preference_shift','off_score_preference_shift'],cfg)
    # Both endpoints of each edge must be in the observed selection event set.
    edge_map = pd.read_parquet(analysis/'edges.parquet')
    edge_rows = []
    retain = policy(cfg).write_node_details
    audit = ArrayAudit(features)
    for (rep,arm,batch),g in df[df.resampled].groupby(['representation','arm','batch'],sort=True):
        idx = g.set_index('node_id')
        e = edge_map[(edge_map.arm==arm)&(edge_map.batch==batch)].copy()
        e = e[e.source.isin(idx.index)&e.target.isin(idx.index)]
        a,b = idx.loc[e.source],idx.loc[e.target]
        dt = lineage_intervals(a.geometry_time,b.geometry_time)
        rate = (b[features].to_numpy()-a[features].to_numpy())/dt[:,None]
        audit.update([rep,arm,int(batch)],rate)
        if retain:
            e['representation'] = rep;e['geometry_delta_time'] = dt
            edge_rows.append(pd.concat([e.reset_index(drop=True),pd.DataFrame(rate,columns=features)],axis=1))
    if retain:
        write_table(dest/'parent_child_rates.parquet',pd.concat(edge_rows,ignore_index=True))
    audit.save(dest/'parent_child_diagnostics.json')
    return result
