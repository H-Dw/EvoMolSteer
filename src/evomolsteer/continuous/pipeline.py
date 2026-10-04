"""Whole-window evidence pipeline and compact, source-bound Analyst input."""
from pathlib import Path
import pandas as pd
from ..io import read_json,write_json,write_table,digest
from .events import build_events,METRICS
from .summary import summarize_method


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
            evidence.append({'evidence_id':f'continuous:{method}:{len(evidence):04d}',
                'method':method,'feature':row.feature,'definition':cat['features'][row.feature],
                'model':models[row.model_id],'source':str(path.relative_to(analysis)),
                'source_sha256':digest(path)})
    skill=Path(__file__).resolve().parents[3]/'skills/continuous-analyst/SKILL.md'
    bundle={'schema_version':'continuous-3.0','scope':scope,'time_binning':False,'evidence':evidence,
        'energy_status':cat.get('energy_status',{}),
        'limitations':['Window ancestry is retrospective, not an independent intervention.',
            'Clones, atoms and event times are not independent biological replicates.',
            'Function fits describe aggregate feature curves, not spatial reward gradients or optimal paths.',
            'Raw hard atom/bond categories can be chemically invalid, especially in proposals.',
            'All null/failed fits, coverage and complete tables remain available in the method directories.']}
    write_json(analysis/'agents/continuous_evidence.json',bundle)
    from .agent_contract import SCHEMA
    import json
    write_json(analysis/'agents/Analyst.continuous.request.json',{'role':'Analyst','schema_version':'continuous-3.0',
        'bundle_sha256':digest(analysis/'agents/continuous_evidence.json'),'response_schema':SCHEMA,
        'messages':[{'role':'system','content':skill.read_text(encoding='utf-8')+'\nReturn JSON matching:\n'+json.dumps(SCHEMA)},
                    {'role':'user','content':__import__('json').dumps(bundle,ensure_ascii=False,allow_nan=False)}]})
    lines=['# Whole-window selection analysis', '',
        f"Observed window: {scope['window_start']}–{scope['window_end']}; {len(scope['steps'])} original event times.",
        'No subinterval averaging or separate subinterval fitting is used.',
        f"Feature count: {len(cat['features'])}. Independent batch splits are frozen in config.json.",
        '', 'See discovery/continuous/*/function_summary.csv, functions.json and fitted_curves.parquet.',
        'Validation and heldout evaluate frozen discovery functions without refitting.',
        'lineage/window_ancestry.parquet retains candidate-level window-end copy counts and extinct controls.',
        'Region steric overlap is a geometric proxy (A^2), not physical energy.']
    (analysis/'CONTINUOUS_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
