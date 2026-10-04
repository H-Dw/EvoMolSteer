"""Curve summaries, analytic fits/derivatives and frozen out-of-batch checks."""
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..io import read_json, write_json, write_table
from ..statistics import bh, summarize_clusters
from .functional import global_fit, quadrature, evaluate_frozen
from .events import METRICS

FIT_METRICS={
 'trends':['expected_selection_shift','window_ancestor_shift','window_ancestor_mean',
           'expected_selected_mean','population_minus_background'],
 'enrichment':['expected_low_mass_shift','expected_high_mass_shift',
               'ancestor_low_mass_shift','ancestor_high_mass_shift'],
 'differential':['selected_minus_rejected','retained_minus_extinct'],
 'dynamics':['cumulative_realized_selection','cumulative_transmission'],
 'pca':['expected_selection_shift','realized_selection_shift']}
KEYS=['representation','arm','feature']


def point_curve(metadata,metric,times,y):
    """Observed rates pair the SAME independent batch across adjacent times."""
    finite=np.isfinite(y);n=finite.sum(0)
    mean=np.divide(np.where(finite,y,0).sum(0),n,out=np.full(len(times),np.nan),where=n>0)
    variance=np.divide(np.where(finite,(y-mean)**2,0).sum(0),n-1,
        out=np.full(len(times),np.nan),where=n>1)
    rates=np.column_stack([np.full(len(y),np.nan),np.diff(y,axis=1)/np.diff(times)])
    valid=np.isfinite(rates);nr=valid.sum(0)
    rate=np.divide(np.where(valid,rates,0).sum(0),nr,out=np.full(len(times),np.nan),where=nr>0)
    ratevar=np.divide(np.where(valid,(rates-rate)**2,0).sum(0),nr-1,
        out=np.full(len(times),np.nan),where=nr>1)
    return pd.DataFrame({**metadata,'contrast':metric,'time':times,'mean':mean,
        'n_batches':n,'batch_se':np.sqrt(variance/np.maximum(n,1)),
        'observed_derivative':rate,'derivative_n_paired_batches':nr,
        'derivative_batch_se':np.sqrt(ratevar/np.maximum(nr,1)),
        'derivative_interval_start':np.r_[np.nan,times[:-1]]})


def summarize_method(analysis,method,split='discovery'):
    root=Path(analysis);cfg=read_json(root/'config.json')
    dest=root/split/'continuous'/method
    events=pd.read_parquet(dest/'batch_event_curves.parquet')
    metrics=METRICS.get(method,FIT_METRICS[method])
    metrics=[m for m in metrics if m in events]
    # Every time point is retained. Standard errors are across independent
    # batches, not particles, edges, clones, or neighboring observations.
    points=[];integrals=[];fits={};fit_rows=[];fitted=[];coef_rows=[];validation=[]
    frozen=read_json(root/'discovery/continuous'/method/'functions.json') if split!='discovery' else {}
    with threadpool_limits(limits=1):
        for key,g in events.groupby(KEYS,sort=True):
            metadata=dict(zip(KEYS,key));times=np.sort(g.time.unique())
            for metric in metrics:
                matrix=g.pivot(index='batch',columns='time',values=metric).reindex(columns=times)
                y=matrix.to_numpy()
                points.append(point_curve(metadata,metric,times,y))
                complete=np.isfinite(y).all(1); yy=y[complete]; batches=matrix.index.to_numpy()[complete]
                for batch,curve in zip(batches,yy):
                    integrals.extend([{**metadata,'contrast':metric,'batch':int(batch),'summary':name,'value':float(value)}
                        for name,value in [('window_average',quadrature(times)@curve),('end_minus_start',curve[-1]-curve[0])]])
                if metric not in FIT_METRICS[method]: continue
                model_id='|'.join([*map(str,key),metric])
                if split!='discovery':
                    if model_id in frozen and len(yy):
                        model=frozen[model_id];pred=evaluate_frozen(model,times);w=quadrature(times)
                        for batch,curve in zip(batches,yy):
                            validation.append({**metadata,'contrast':metric,'batch':int(batch),'model_id':model_id,
                                'degree':model['degree'],'rmse':float(np.sqrt(np.sum((curve-pred)**2*w))),
                                'constant_baseline_rmse':float(np.sqrt(np.sum((curve-model['discovery_constant'])**2*w))),
                                'window_average':float(w@curve),'end_minus_start':float(curve[-1]-curve[0]),
                                'cosine_to_frozen_curve':float(curve@pred/(np.linalg.norm(curve)*np.linalg.norm(pred)))
                                    if np.linalg.norm(curve)*np.linalg.norm(pred)>1e-15 else np.nan})
                    continue
                if len(yy)<cfg['minimum_inference_batches']:
                    fit_rows.append({**metadata,'contrast':metric,'model_id':model_id,
                        'n_batches':len(yy),'status':'insufficient_complete_batch_curves','global_curve_p':np.nan})
                    continue
                fit=global_fit(times,yy,cfg)
                curves=fit.pop('curves');coefficients=fit.pop('batch_coefficients')
                fit.update(metadata,contrast=metric,model_id=model_id,status='fitted',
                    discovery_constant=float(yy.mean(0)@quadrature(times)),support_batches=batches.tolist())
                fits[model_id]=fit
                fit_rows.append({k:v for k,v in fit.items() if not isinstance(v,(list,dict))})
                fitted.append(pd.DataFrame({**metadata,'contrast':metric,'model_id':model_id,**curves}))
                for batch,coefs in zip(batches,coefficients):
                    coef_rows.extend({**metadata,'contrast':metric,'batch':int(batch),'degree':fit['degree'],
                        'coefficient_index':i,'coefficient':float(c)} for i,c in enumerate(coefs))
    write_table(dest/'point_curves.parquet',pd.concat(points,ignore_index=True))
    integral=pd.DataFrame(integrals)
    write_table(dest/'batch_window_summaries.parquet',integral)
    if len(integral):
        table=summarize_clusters(integral,KEYS+['contrast','summary'],'value',cfg)
        # Correct jointly over all feature/window-summary tests within contrast.
        write_table(dest/'window_summary.csv',table)
    if split=='discovery':
        summary=pd.DataFrame(fit_rows);summary['global_curve_q']=np.nan
        for _,g in summary.groupby(['representation','arm','contrast'],sort=True):
            summary.loc[g.index,'global_curve_q']=bh(g.global_curve_p)
        for row in summary.itertuples():
            if row.model_id in fits: fits[row.model_id]['global_curve_q']=row.global_curve_q
        write_json(dest/'functions.json',fits)
        write_table(dest/'function_summary.csv',summary)
        if fitted: write_table(dest/'fitted_curves.parquet',pd.concat(fitted,ignore_index=True))
        write_table(dest/'batch_coefficients.parquet',coef_rows)
    else:
        write_table(dest/'frozen_function_validation.csv',validation)
    write_json(dest/'method.json',{'time_binning':False,'fit_selection':'leave-one-independent-batch-out; one-standard-error',
        'candidate_degrees':list(range(cfg.get('max_degree',5)+1)),
        'global_test':'Integrated squared unfitted mean curve; sign-flip entire batch curves under sign-symmetry null',
        'uncertainty':'Conditional-on-degree whole-batch bootstrap; simultaneous bands are per curve, not across features',
        'missingness':'Full-window fits/integrals require complete feature curves; all pointwise coverage and skipped fits retained',
        'absolute_level_warning':'A nonzero level is not a selection advantage. Prioritize shift/contrast curves.',
        'split':split,'metrics':metrics})
    return dest
