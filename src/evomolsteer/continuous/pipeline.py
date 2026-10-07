"""Whole-window evidence pipeline and compact, source-bound Analyst input."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..io import read_json,write_json,write_table,digest,clean
from .events import build_events,METRICS
from .summary import summarize_method


def llm_view(value):
    """Compact transport only; scientific tables/evidence keep full float64."""
    if isinstance(value,float):return float(f'{value:.8g}')
    if isinstance(value,list):return [llm_view(v) for v in value]
    if isinstance(value,dict):
        out={k:llm_view(v) for k,v in value.items()}
        if 'observed_curve' in out:
            out['observed_curve'].pop('time',None)
            out['observed_curve']['time_axis']='scope.score_times (all original events)'
        if out.get('method') in ['lineage_control','differential_control','selection_moment_control'] and out.get('data'):
            columns=list(out['data'][0]);out['data']={'columns':columns,'rows':[[r[c] for c in columns] for r in out['data']]}
        return out
    return value


def run(analysis, splits=('discovery','validation','heldout')):
    analysis=Path(analysis)
    if read_json(analysis/'config.json').get('time_analysis')!='continuous_window':
        raise ValueError('Continuous v3 requires a fresh continuous-window extraction')
    for split in splits:
        print('Whole-window events: '+split,flush=True)
        build_events(analysis,split)
        for method in METRICS:
            print('Whole-window '+method+': '+split,flush=True)
            summarize_method(analysis,method,split)
        from .. import pca
        print('Whole-window PCA: '+split,flush=True)
        pca.run(analysis,split)
    build_evidence(analysis)


def build_evidence(analysis):
    analysis=Path(analysis);cfg=read_json(analysis/'config.json');cat=read_json(analysis/'feature_catalog.json')
    scope=read_json(analysis/'selection_scope.json');evidence=[]
    for method in METRICS:
        path=analysis/'discovery/continuous'/method/'function_summary.csv'
        summary=pd.read_csv(path)
        # Ranked excerpts are a transport policy, never a deletion of null results.
        eligible=summary[summary.status=='fitted'].copy()
        contrast_priority={'expected_selection_shift','window_ancestor_shift','population_minus_background',
            'expected_low_mass_shift','expected_high_mass_shift','ancestor_low_mass_shift','ancestor_high_mass_shift',
            'selected_minus_rejected','retained_minus_extinct','cumulative_realized_selection','cumulative_transmission'}
        eligible=eligible[eligible.contrast.isin(contrast_priority)]
        eligible=eligible.sort_values(['global_curve_q','global_curve_p','feature','contrast'],kind='stable')
        models=read_json(path.parent/'functions.json')
        points=pd.read_parquet(path.parent/'point_curves.parquet')
        fitted=pd.read_parquet(path.parent/'fitted_curves.parquet')
        # Transport a range of observables rather than only nearby correlated
        # distances; every fit remains in the unfiltered statistical artifacts.
        eligible=eligible.drop_duplicates(['representation','feature'])
        def family(feature):
            kind=feature.split('::')[-1]
            for term in ['centroid','bond','charge','atom_fraction','hetero','steric','distance','contact','radius']:
                if term in kind:return term
            return 'other'
        family_counts={};indices=[];limit=cfg.get('evidence_limit_per_method',16)
        for i,row in eligible.iterrows():
            kind=family(row.feature)
            if family_counts.get(kind,0)>=3:continue
            indices.append(i);family_counts[kind]=family_counts.get(kind,0)+1
            if len(indices)>=limit:break
        selected=eligible.loc[indices]
        for row in selected.itertuples():
            curve=points[(points.feature==row.feature)&(points.representation==row.representation)&
                         (points.arm==row.arm)&(points.contrast==row.contrast)].sort_values('time')
            fc=fitted[fitted.model_id==row.model_id].sort_values('time')
            turns=[]
            for t in models[row.model_id]['turning_times']:
                near=fc.iloc[int(np.argmin(np.abs(fc.time-t)))]
                turns.append({'time':t,'nearest_observed_time':float(near.time),
                    'fitted_value':float(near.fitted),'pointwise_derivative_ci':[float(near.derivative_ci_low),float(near.derivative_ci_high)]})
            evidence.append({'evidence_id':f'continuous:{method}:{len(evidence):04d}',
                'method':method,'feature':row.feature,'definition':cat['features'][row.feature],
                'model':models[row.model_id],'source':str(path.relative_to(analysis)),
                'source_sha256':digest(path),'observed_curve':{
                    'time':curve.time.tolist(),'mean':curve['mean'].tolist(),
                    'paired_derivative':curve.observed_derivative.tolist()},
                'curve_source_sha256':digest(path.parent/'point_curves.parquet'),
                'fit_diagnostics':{'turning_points':turns,
                    'positive_derivative_support_times':fc.loc[fc.derivative_ci_low>0,'time'].tolist(),
                    'negative_derivative_support_times':fc.loc[fc.derivative_ci_high<0,'time'].tolist(),
                    'maximum_simultaneous_half_width':float(((fc.simultaneous_high-fc.simultaneous_low)/2).max()),
                    'full_curve_table':str((path.parent/'fitted_curves.parquet').relative_to(analysis)),
                    'full_curve_sha256':digest(path.parent/'fitted_curves.parquet'),
                    'caution':'Derivative support is pointwise, conditional on selected degree; small turning-point oscillations can be below residual error.'}})
    # Controls are transported alongside positive excerpts, not left for an LLM
    # to infer from an earlier analysis or a different contract.
    lineage_path=analysis/'discovery/continuous/lineage/diagnostics.csv'
    lineage=pd.read_csv(lineage_path)
    lc=lineage.groupby(['arm','time'],as_index=False)[['retained_ancestor_candidates','current_root_count',
        'ancestor_copy_ess','current_selection_ess','model_pic50_population','model_pic50_window_ancestors']].mean()
    evidence.append({'evidence_id':'continuous:lineage:control','method':'lineage_control',
        'interpretation':'Equal-independent-batch means; copies are retrospective window-end ancestry, never new independent samples.',
        'data':lc.to_dict('records'),'source':str(lineage_path.relative_to(analysis)),'source_sha256':digest(lineage_path)})
    contrasts_path=analysis/'discovery/continuous/differential/point_curves.parquet'
    controls=pd.read_parquet(contrasts_path)
    selected_features=sorted({(e['model']['representation'],e['model']['arm'],e['feature']) for e in evidence if 'model' in e})
    control_rows=[]
    for rep,arm,feature in selected_features:
        for contrast in ['same_parent_selected_minus_rejected','selected_minus_rejected','retained_minus_extinct']:
            c=controls[(controls.representation==rep)&(controls.arm==arm)&(controls.feature==feature)&(controls.contrast==contrast)]
            control_rows.append({'representation':rep,'arm':arm,'feature':feature,'contrast':contrast,
                'maximum_abs_event_mean':c['mean'].abs().max(),'mean_of_event_means':c['mean'].mean(),
                'n_event_times_with_data':int((c.n_batches>0).sum()),
                'min_batches_per_supported_time':int(c.loc[c.n_batches>0,'n_batches'].min()) if (c.n_batches>0).any() else 0})
    evidence.append({'evidence_id':'continuous:differential:controls','method':'differential_control',
        'interpretation':'Descriptive controls over all event times; no unreported pointwise significance claim.',
        'data':control_rows,'source':str(contrasts_path.relative_to(analysis)),'source_sha256':digest(contrasts_path)})
    trends_path=analysis/'discovery/continuous/trends/point_curves.parquet';controls=pd.read_parquet(trends_path)
    moments=[]
    for rep,arm,feature in selected_features:
        for contrast in ['expected_selection_shift','realized_selection_shift','selection_noise_shift',
                         'population_mean','expected_selected_mean','window_ancestor_mean','window_ancestor_shift']:
            c=controls[(controls.representation==rep)&(controls.arm==arm)&(controls.feature==feature)&(controls.contrast==contrast)].sort_values('time')
            moments.append({'representation':rep,'arm':arm,'feature':feature,'contrast':contrast,
                'mean_of_event_means':c['mean'].mean(),'first_event_mean':c['mean'].iloc[0] if len(c) else None,
                'last_event_mean':c['mean'].iloc[-1] if len(c) else None})
    evidence.append({'evidence_id':'continuous:trends:controls','method':'selection_moment_control',
        'interpretation':'Levels, instantaneous preference, realized copying, noise and retrospective ancestry remain distinct.',
        'data':moments,'source':str(trends_path.relative_to(analysis)),'source_sha256':digest(trends_path)})
    from ..skill_guidance import render
    skill_text=render('Analyst',cfg.get('agent_guidance_modules',{}).get('Analyst',[]))
    bundle={'schema_version':'continuous-3.0','scope':scope,'time_binning':False,'evidence':evidence,
        'fit_protocol':read_json(analysis/'discovery/continuous/trends/method.json'),
        'energy_status':cat.get('energy_status',{}),
        'limitations':['Window ancestry is retrospective, not an independent intervention.',
            'Clones, atoms and event times are not independent biological replicates.',
            'Function fits describe aggregate feature curves, not spatial reward gradients or optimal paths.',
            'Raw hard atom/bond categories can be chemically invalid, especially in proposals.',
            'All null/failed fits, coverage and complete tables remain available in the method directories.']}
    bundle=clean(bundle)
    write_json(analysis/'agents/continuous_evidence.json',bundle)
    from .agent_contract import SCHEMA
    import json
    transport=llm_view(bundle)
    transport['transport_note']='Only this LLM view uses 8 significant decimal digits and columnar control tables; full-precision evidence and tables remain authoritative. Time arrays are shared via scope.score_times.'
    write_json(analysis/'agents/Analyst.continuous.request.json',{'role':'Analyst','schema_version':'continuous-3.0',
        'bundle_sha256':digest(analysis/'agents/continuous_evidence.json'),'response_schema':SCHEMA,
        'messages':[{'role':'system','content':skill_text+'\nReturn JSON matching:\n'+json.dumps(SCHEMA)},
                    {'role':'user','content':json.dumps(transport,ensure_ascii=False,allow_nan=False,separators=(',',':'))}]})
    lines=['# Whole-window selection analysis', '',
        f"Observed window: {scope['window_start']}–{scope['window_end']}; {len(scope['steps'])} original event times.",
        'No subinterval averaging or separate subinterval fitting is used.',
        f"Feature count: {len(cat['features'])}. Independent batch splits are frozen in config.json.",
        '', 'See discovery/continuous/*/function_summary.csv, functions.json and fitted_curves.parquet.',
        'Validation and heldout evaluate frozen discovery functions without refitting.',
        'lineage/window_ancestry.parquet retains candidate-level window-end copy counts and extinct controls.',
        'Region steric overlap is a geometric proxy (A^2), not physical energy.']
    (analysis/'CONTINUOUS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
