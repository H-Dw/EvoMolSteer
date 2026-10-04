"""Event-first effects, vectorized batch inference, and explicit FDR families."""
from functools import lru_cache
import itertools
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .io import write_table,write_json
from .output_policy import policy,write_batch_table,validate_output_layout
from .scope import load

EVENT_KEYS = ['representation','arm','batch','step']
EFFECT_KEYS = ['representation','arm','contrast','feature']


def lineage_intervals(parent_times,child_times):
    dt = np.asarray(child_times,float)-np.asarray(parent_times,float)
    if not np.isfinite(dt).all() or np.any(dt<=0):
        raise ValueError('Nonpositive geometry-time interval')
    return dt


def bh(p):
    p = np.asarray(p,float)
    q = np.full(len(p),np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    if len(idx):
        order = idx[np.argsort(p[idx],kind='stable')]
        q[order] = np.minimum(1,np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return q


@lru_cache(maxsize=32)
def _draws(n,seed,repeats):
    rng = np.random.default_rng(seed+n*104729)
    exact = n<=12
    signs = np.array(list(itertools.product([-1.,1.],repeat=n))) if exact else rng.choice([-1.,1.],size=(repeats,n))
    indices = rng.integers(0,n,size=(repeats,n))
    counts = np.zeros((repeats,n))
    np.add.at(counts,(np.arange(repeats)[:,None],indices),1)
    return signs/n,counts/n,exact


def summarize_clusters(df,keys,value,cfg):
    rows,by_n = [],{}
    for key,g in df.groupby(keys,sort=True,dropna=False):
        if 'batch' in g:
            if g.batch.duplicated().any():
                raise ValueError('More than one effect per independent batch')
            g = g.sort_values('batch')
        v = g[value].to_numpy(float)
        v = v[np.isfinite(v)]
        key = key if isinstance(key,tuple) else (key,)
        n = len(v)
        rows.append({**dict(zip(keys,key)), 'effect':v.mean() if n else np.nan,
            'n_batches':n,'positive_batch_fraction':np.mean(v>1e-12) if n else np.nan,
            'negative_batch_fraction':np.mean(v < -1e-12) if n else np.nan,
            'p_value':np.nan,'ci_low':np.nan,'ci_high':np.nan,
            'inference':'insufficient_independent_batches'})
        if n>=cfg['minimum_inference_batches']:
            by_n.setdefault(n,[]).append((len(rows)-1,v))
    # Shared resampling draws are deterministic and valid for each marginal test.
    # Matrix chunks avoid re-generating thousands of identical bootstrap designs.
    with threadpool_limits(limits=1):
        for n,items in by_n.items():
            signs,bootstrap,exact = _draws(n,cfg['seed'],cfg['permutations'])
            for start in range(0,len(items),128):
                chunk = items[start:start+128]
                values = np.column_stack([v for _,v in chunk])
                null = signs@values
                hits = (np.abs(null)>=np.abs(values.mean(0))-1e-12).sum(0)
                p = hits/len(signs) if exact else (hits+1)/(len(signs)+1)
                ci = np.quantile(bootstrap@values,[.025,.975],axis=0)
                for j,(index,_) in enumerate(chunk):
                    rows[index].update(p_value=float(p[j]),ci_low=float(ci[0,j]),ci_high=float(ci[1,j]),
                                       inference='batch_sign_flip_and_batch_bootstrap')
    out = pd.DataFrame(rows,columns=keys+['effect','n_batches','positive_batch_fraction','negative_batch_fraction',
                                        'p_value','ci_low','ci_high','inference'])
    out['q_value'] = np.nan
    out['fdr_family'] = ''
    family = [c for c in ['representation','arm','contrast'] if c in keys]
    groups = out.groupby(family,sort=True,dropna=False) if family else [('all',out)]
    for name,g in groups:
        out.loc[g.index,'q_value'] = bh(g.p_value)
        out.loc[g.index,'fdr_family'] = str(name)
    return out


def cluster_summary(values,cfg,key=None):
    frame = pd.DataFrame({'batch':np.arange(len(values)),'group':'one','value':values})
    return summarize_clusters(frame,['group'],'value',cfg).iloc[0].to_dict()


def stage_effect_rates(batch_effects,keys,value):
    columns = keys+['batch','stage','previous_stage','delta_time','rate']
    if batch_effects.empty:
        return pd.DataFrame(columns=columns)
    d = batch_effects.sort_values(keys+['batch','stage'],kind='stable').copy()
    g = d.groupby(keys+['batch'],sort=False,dropna=False)
    d['previous_stage'] = g.stage.shift()
    d['delta_time'] = d.time-g.time.shift()
    d['rate'] = (d[value]-g[value].shift())/d.delta_time
    return d.loc[(d.stage-d.previous_stage==1)&(d.delta_time>0),columns]


def event_frame(g,features,**metrics):
    row = g.iloc[0]
    return pd.DataFrame({**{k:row[k] for k in EVENT_KEYS},'stage':row.stage,
                         'time':row.score_time,'dt':row['dt'],'feature':features,**metrics})


def weighted_mean(x,w):
    finite = np.isfinite(x)
    mass = (np.asarray(w)[:,None]*finite).sum(0)
    total = (np.asarray(w)[:,None]*np.where(finite,x,0)).sum(0)
    return np.divide(total,mass,out=np.full(x.shape[1],np.nan),where=mass>0),mass


def selection_moments(x,p,counts,minimum_fraction=.8):
    p = np.asarray(p,float)/np.sum(p)
    freq = np.asarray(counts,float)/np.sum(counts)
    base,coverage = weighted_mean(x,np.ones(len(x))/len(x))
    expected,mass = weighted_mean(x,p)
    realized,realized_mass = weighted_mean(x,freq)
    eligible = (coverage>=minimum_fraction)&(mass>=minimum_fraction)
    expected_shift = np.where(eligible,expected-base,np.nan)
    realized_shift = np.where(eligible&(realized_mass>=minimum_fraction),realized-base,np.nan)
    # Exact multinomial sampling SD is reported only for complete features.
    second,_ = weighted_mean(x*x,p)
    sd = np.where(coverage==1,np.sqrt(np.maximum(second-expected**2,0)/len(x)),np.nan)
    return {'population_mean':base,'expected_selection_shift':expected_shift,
            'realized_selection_shift':realized_shift,'selection_noise_shift':realized_shift-expected_shift,
            'valid_fraction':coverage,'retained_probability_mass':mass,'multinomial_sampling_sd':sd}


def save_metrics(dest,events,metrics,cfg):
    """Each event gets equal weight within a batch-stage; batches get equal weight."""
    dest.mkdir(parents=True,exist_ok=True)
    validate_output_layout(dest,cfg)
    events = events.sort_values(['representation','arm','batch','feature','step'],kind='stable').reset_index(drop=True)
    retention = policy(cfg)
    if retention.write_step_details:
        write_table(dest/'events.parquet',events)
    g = events.groupby(['representation','arm','batch','feature'],sort=False)
    dt = events.time-g.time.shift()
    adjacent = (events.step-g.step.shift()==1)&(dt>0)
    rates = events[EVENT_KEYS+['stage','time','feature']].copy()
    rates['delta_time'] = dt
    for metric in metrics:
        rates[metric+'_derivative'] = ((events[metric]-g[metric].shift())/dt).where(adjacent)
    if retention.write_step_details:
        write_table(dest/'event_rates.parquet',rates[adjacent])
    diagnostics = {'event_feature_rows':len(events),'adjacent_event_feature_rows':int(adjacent.sum()),
        'time_axis':'score_time','finite_metric_values':{m:int(np.isfinite(events[m]).sum()) for m in metrics},
        'finite_derivatives':{m:int(np.isfinite(rates.loc[adjacent,m+'_derivative']).sum()) for m in metrics},
        'all_derivatives_computed':True}
    if {'realized_selection_shift','expected_selection_shift','selection_noise_shift'} <= set(events):
        residual = (events.realized_selection_shift-events.expected_selection_shift-events.selection_noise_shift).abs()
        diagnostics['max_selection_noise_identity_residual'] = float(residual.max())
        if not (residual.dropna()<=1e-10).all():
            raise ValueError('Selection decomposition identity failed')
    write_json(dest/'event_diagnostics.json',diagnostics)
    del rates
    batch_rows = []
    group_keys = ['representation','arm','batch','stage','feature']
    for metric in metrics:
        good = events[np.isfinite(events[metric])]
        b = good.groupby(group_keys,as_index=False,sort=True).agg(
            value=(metric,'mean'),time=('time','mean'),n_events=('step','nunique'))
        b['contrast'] = metric
        batch_rows.append(b)
    batch = pd.concat(batch_rows,ignore_index=True)
    write_batch_table(dest/'batch_effects.csv',batch,cfg)
    summary = summarize_clusters(batch,EFFECT_KEYS+['stage'],'value',cfg)
    write_table(dest/'effects.csv',summary)
    changes = stage_effect_rates(batch,EFFECT_KEYS,'value')
    write_batch_table(dest/'batch_effect_rates.csv',changes,cfg)
    write_table(dest/'effect_rates.csv',summarize_clusters(changes,EFFECT_KEYS+['stage'],'rate',cfg))
    return summary
