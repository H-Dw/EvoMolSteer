"""Small, auditable coordinate prior from independent-batch window effects.

This does not fit a new predictor or claim a causal binding mechanism. Excluded
features remain in the original evidence tables; only the runtime prior is sparse.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from ..io import read_json,write_json,digest
from ..generation.window_reference import load_reference


def build_prior(reference_path,effects_path,functions_path,output,q_max=.05,min_effect=.1,min_batch_fraction=12/14):
    reference=load_reference(reference_path);effects=pd.read_parquet(effects_path);functions=read_json(functions_path)
    if reference['schema_version']!='affinity-endpoint-library-1.0':raise ValueError('Endpoint evidence required')
    features=reference['features'];effects=effects.set_index('feature')
    if len(reference['feature_scale'])!=len(features):raise ValueError('Feature/scale dimensions differ')
    if effects.index.duplicated().any() or set(effects.index)!=set(features) or set(functions)!=set(features):
        raise ValueError('Exact feature identity required across all sources')
    if not all(math.isfinite(v) for v in (q_max,min_effect,min_batch_fraction)) or not 0<q_max<=1 or min_effect<0 or not .5<min_batch_fraction<=1:
        raise ValueError('Invalid evidence thresholds')
    values=[];weights=[];selected_functions={};node_functions={};fit_audit={}
    grid=np.asarray(reference['times'],float)
    empirical=np.asarray([(np.asarray(f['high_scaled'])-np.asarray(f['low_scaled'])).mean(0) for f in reference['frames']])
    for feature,scale in zip(features,reference['feature_scale']):
        row=effects.loc[feature];effect=float(row.high_low_integrated_z);q=float(row.q)
        agreement=float(row.positive_batch_fraction if effect>0 else 1-row.positive_batch_fraction)
        finite=all(math.isfinite(v) for v in (effect,q,agreement,scale))
        reasons=[]
        if not finite:reasons.append('nonfinite_evidence')
        if q>=q_max:reasons.append('q_threshold')
        if abs(effect)<min_effect:reasons.append('small_integrated_effect')
        if agreement+1e-12<min_batch_fraction:reasons.append('batch_direction_inconsistent')
        if scale<=1e-5:reasons.append('scale_floor')
        accepted=not reasons;weights.append(float(accepted))
        if accepted:
            f=functions[feature];coeff=list(map(float,f['legendre_coefficients']))
            if not 0<=f['degree']<=3 or len(coeff)!=f['degree']+1 or not np.isfinite(coeff).all():raise ValueError('Invalid recorded trend function')
            if f['time_start']!=reference['times'][0] or f['time_end']!=reference['times'][-1]:raise ValueError('Trend/node support mismatch')
            selected_functions[feature]={k:f[k] for k in ('degree','legendre_coefficients','time_start','time_end')}
            actual=empirical[:,features.index(feature)]
            predicted=np.polynomial.legendre.legval(2*(grid-grid[0])/(grid[-1]-grid[0])-1,coeff)
            strong=(np.abs(actual)>=.1)&(np.abs(predicted)>=.1)
            conflicts=int(((actual*predicted<0)&strong).sum())
            relative_rmse=float(np.sqrt(np.mean((actual-predicted)**2))/max(np.sqrt(np.mean(actual**2)),1e-12))
            quiet_fraction=float(((np.abs(actual)<.1)&(np.abs(predicted)>=.1)).mean())
            fit_audit[feature]={'relative_node_rmse':relative_rmse,'strong_sign_conflict_nodes':conflicts,
                'quiet_node_large_fit_fraction':quiet_fraction,'last_empirical_z':float(actual[-1]),'last_fit_z':float(predicted[-1]),
                'legendre_runtime_allowed':bool(relative_rmse<=.35 and conflicts==0 and quiet_fraction<=.1)}
            node_functions[feature]={'values_z':actual.tolist()}
        values.append({'feature':feature,'selected':accepted,'effect_z':effect,'q':q,'batch_agreement':agreement,
                       'scale':float(scale),'excluded_reasons':reasons})
    if not any(weights):raise ValueError('No supported coordinate fields; never silently use an all-zero reward')
    prior={'schema_version':'endpoint-coordinate-prior-1.0','window':reference['window'],'times':reference['times'],
        'reference_sha256':digest(reference_path),'effects_sha256':digest(effects_path),'functions_sha256':digest(functions_path),
        'features':features,'feature_weights':weights,'direction_functions':selected_functions,'node_effect_functions':node_functions,
        'curve_fidelity_audit':fit_audit,'curve_fidelity_rule':{'relative_rmse_le':.35,'strong_sign_conflict_nodes_eq':0,'quiet_node_large_fit_fraction_le':.1},
        'evidence':values,
        'selection_rule':{'q_lt':q_max,'absolute_integrated_z_ge':min_effect,'batch_direction_fraction_ge':min_batch_fraction,'exclude_scale_floor':True},
        'selected_count':int(sum(weights)),'n_independent_batches':len(reference['discovery_batches']),
        'interpretation':'Observational coordinates associated with recorded joint head labels. Temporal trend is a coefficient, not a spatial force. No additional fitted affinity predictor.'}
    write_json(output,prior);return prior
