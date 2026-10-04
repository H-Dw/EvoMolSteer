"""Selection contrasts matched on event, with same-parent controls."""
from pathlib import Path
import numpy as np
import pandas as pd
from .scope import load
from .statistics import EVENT_KEYS,event_frame,save_metrics


def run(analysis,split='discovery'):
    analysis = Path(analysis)
    df,cfg,cat,_ = load(analysis,split)
    features = list(cat['features'])
    rows = []
    for _,g in df[df.resampled].groupby(EVENT_KEYS,sort=True):
        selected,rejected = g[g.selected],g[~g.selected]
        delta = selected[features].mean()-rejected[features].mean()
        coverage = g[features].notna().mean()
        delta = delta.where(coverage>=cfg['minimum_feature_fraction'])
        s = g[g.parent_node_id!='']
        means = s.groupby(['parent_node_id','selected'],sort=True)[features].mean()
        sibling = np.full(len(features),np.nan)
        n_pairs = np.zeros(len(features),int)
        if len(means) and {True,False} <= set(means.index.get_level_values('selected')):
            a = means.xs(True,level='selected');b = means.xs(False,level='selected')
            pairs = a.index.intersection(b.index)
            diff = a.loc[pairs]-b.loc[pairs]
            sibling = diff.mean().to_numpy();n_pairs = diff.notna().sum().to_numpy()
        rows.append(event_frame(g,features,selected_minus_rejected=delta.to_numpy(),
            selected_minus_rejected_siblings=sibling,sibling_parent_pairs=n_pairs,
            n_selected=len(selected),n_rejected=len(rejected)))
    return save_metrics(analysis/split/'differential',pd.concat(rows,ignore_index=True),
                        ['selected_minus_rejected','selected_minus_rejected_siblings'],cfg)
