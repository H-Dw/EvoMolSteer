"""Same-event high-feature enrichment; expected preference and random selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from .scope import load
from .statistics import EVENT_KEYS,event_frame,weighted_mean,save_metrics
from .io import write_table


def run(analysis,split='discovery'):
    analysis = Path(analysis)
    df,cfg,cat,scope = load(analysis,split)
    reference = df if split=='discovery' else load(analysis,'discovery')[0]
    reference = reference[reference.arm==scope['background_arm']]
    features = list(cat['features'])
    cutoffs = reference.groupby(['representation','step'],sort=True)[features].quantile(.75)
    threshold_rows = cutoffs.reset_index().melt(id_vars=['representation','step'],var_name='feature',value_name='threshold')
    rows = []
    for key,g in df[df.resampled].groupby(EVENT_KEYS,sort=True):
        rep,arm,batch,step = key
        x = g[features].to_numpy(float)
        cutoff = cutoffs.loc[(rep,step)].to_numpy(float)
        finite = np.isfinite(x)&np.isfinite(cutoff)[None,:]
        high = np.where(finite,(x>cutoff).astype(float),np.nan)
        p = g.probability.to_numpy(float,copy=True);p /= p.sum()
        freq = g.offspring_count.to_numpy(float)/len(g)
        baseline,coverage = weighted_mean(high,np.ones(len(g))/len(g))
        expected,mass = weighted_mean(high,p)
        realized,rmass = weighted_mean(high,freq)
        eligible = (coverage>=cfg['minimum_feature_fraction'])&(mass>=cfg['minimum_feature_fraction'])
        label = g.selected.to_numpy(bool)[:,None]
        a = (finite&label&(high==1)).sum(0)
        b = (finite&label&(high==0)).sum(0)
        c = (finite&~label&(high==1)).sum(0)
        d = (finite&~label&(high==0)).sum(0)
        logor = np.log((a+.5)*(d+.5)/((b+.5)*(c+.5)))
        logor = np.where(eligible&((a+b)>0)&((c+d)>0),logor,np.nan)
        rows.append(event_frame(g,features,valid_fraction=coverage,
            expected_high_mass_shift=np.where(eligible,expected-baseline,np.nan),
            realized_high_mass_shift=np.where(eligible&(rmass>=cfg['minimum_feature_fraction']),realized-baseline,np.nan),
            selected_vs_rejected_log_odds=logor))
    dest = analysis/split/'enrichment'
    result = save_metrics(dest,pd.concat(rows,ignore_index=True),
        ['expected_high_mass_shift','realized_high_mass_shift','selected_vs_rejected_log_odds'],cfg)
    write_table(dest/'thresholds.csv',threshold_rows)
    return result
