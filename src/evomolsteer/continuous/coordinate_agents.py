"""Bounded discovery-only evidence export, strict LLM import and whitelisted design."""
from pathlib import Path
import json
import jsonschema
import numpy as np
import pandas as pd
from ..io import read_json,write_json,digest,clean

PROJECT=Path(__file__).resolve().parents[3]
STATEMENT={'type':'object','additionalProperties':False,'properties':{'finding':{'type':'string'},
    'evidence_ids':{'type':'array','items':{'type':'string'},'minItems':1},'uncertainty':{'type':'string'}},
    'required':['finding','evidence_ids','uncertainty']}
RULE={'type':'object','additionalProperties':False,'properties':{'feature':{'type':'string'},
    'status':{'enum':['selection_association_hypothesis','deferred']},'rationale':{'type':'string'},
    'evidence_ids':{'type':'array','items':{'type':'string'},'minItems':1}},
    'required':['feature','status','rationale','evidence_ids']}
ANALYST={'type':'object','additionalProperties':False,'properties':{'schema_version':{'const':'coordinate-1.0'},
    'agent':{'const':'Analyst'},'observations':{'type':'array','items':STATEMENT},'rules':{'type':'array','items':RULE},
    'counterevidence':{'type':'array','items':STATEMENT},'limitations':{'type':'array','items':{'type':'string'}}},
    'required':['schema_version','agent','observations','rules','counterevidence','limitations']}
DESIGNER={'type':'object','additionalProperties':False,'properties':{'schema_version':{'const':'coordinate-1.0'},
    'agent':{'const':'Designer'},'design_status':{'enum':['exploratory','deferred']},
    'architecture':{'enum':['spread_upper','coordinate_mixture','selection_contrast','shape_mixture']},'regions':{'type':'array','items':{'type':'string'},'minItems':1,'uniqueItems':True},
    'channel':{'enum':['all','NOS']},'window':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
    'native_rms_ratio':{'type':'number','minimum':0,'maximum':1},'mixture_temperature':{'type':'number','exclusiveMinimum':0},
    'robust_delta':{'type':'number','exclusiveMinimum':0},'rationale':{'type':'string'},
    'dose_reference':{'enum':['observed_native','predictive_flow']},'preserve_native_rigid_pose':{'type':'boolean'},
    'initial_update_dose':{'enum':['native','cap_to_flow']},
    'contrast_bound_nats':{'type':'number','exclusiveMinimum':0},
    'evidence_ids':{'type':'array','items':{'type':'string'},'minItems':1},'limitations':{'type':'array','items':{'type':'string'}}},
    'required':['schema_version','agent','design_status','architecture','regions','channel','window','native_rms_ratio','mixture_temperature','robust_delta','rationale','evidence_ids','limitations']}


def export(mining,role,landmarks=('ck2:A:ASN117','ck2:A:VAL116'),transport=None,influence=None):
    mining=Path(mining);dest=mining/'agents';dest.mkdir(exist_ok=True)
    m=read_json(mining/'manifest.json');catalog=read_json(mining/'feature_catalog.json')
    if role not in ('Analyst','Designer'):raise ValueError('Unknown role')
    d=pd.read_csv(mining/'whole_window_evidence.csv');d=d[d.split=='discovery']
    # Only compact integrated summaries, counterevidence and relevant frozen fits.
    landmark_rows=d[d.feature.map(lambda f:catalog['features'][f]['region'] in landmarks)]
    top=d.sort_values(['q','feature']).groupby('metric',sort=True).head(8)
    selected=pd.concat([landmark_rows,top]).drop_duplicates(['feature','metric']).sort_values(['metric','feature'])
    evidence=[{'evidence_id':f'coordinate:discovery:{r.feature}:{r.metric}',**r._asdict()} for r in selected.itertuples(index=False)]
    counts={metric:{'tested':len(g),'q_below_05':int(g.q.lt(.05).sum()),'minimum_q':float(g.q.min())} for metric,g in d.groupby('metric')}
    diag=pd.read_csv(mining/'lineage_diagnostics.csv');diag=diag[diag.batch.isin(m['splits']['discovery'])]
    last=diag[np.isclose(diag.time,m['window'][1])]
    evidence.extend([{'evidence_id':'D_COUNTS','metric_counts':counts},
        {'evidence_id':'D_LINEAGE','last_time':m['window'][1],'mean_unique_roots':float(last.unique_roots.mean()),
         'batches_with_one_root':int(last.unique_roots.eq(1).sum()),'lag_last_time':m['times'][-2]}])
    fits=read_json(mining/'continuous_functions.json')
    payload={'schema_version':'coordinate-evidence-1.0','window':m['window'],'times':m['times'],
        'spatial_anchor':m.get('spatial_anchor','current'),'control_representation':m.get('control_representation','current'),
        'discovery_batches':m['splits']['discovery'],'evidence':evidence,
        'features':{f:catalog['features'][f] for f in selected.feature.unique()},
        'functions':{f:fits[f] for f in selected.feature.unique() if f in fits},
        'interpretation':m['interpretation'],'source_manifest_sha256':digest(mining/'manifest.json')}
    if transport:
        extra=Path(transport);mm=read_json(extra/'manifest.json');cc=read_json(extra/'feature_catalog.json')
        if mm['window']!=m['window'] or mm['splits']['discovery']!=m['splits']['discovery'] or mm['sources']!=m['sources'] or mm.get('spatial_anchor')!=m.get('spatial_anchor'):raise ValueError('Supplementary evidence support mismatch')
        dd=pd.read_csv(extra/'whole_window_evidence.csv');dd=dd[dd.split=='discovery']
        ss=pd.concat([dd[dd.feature.map(lambda f:cc['features'][f]['region'] in landmarks)],
            dd.sort_values(['q','feature']).groupby('metric',sort=True).head(8)]).drop_duplicates(['feature','metric']).sort_values(['metric','feature'])
        if set(ss.feature)&set(payload['features']):raise ValueError('Overlapping supplementary feature definitions')
        payload['evidence'] += [{'evidence_id':f'coordinate:discovery:{r.feature}:{r.metric}',**r._asdict()} for r in ss.itertuples(index=False)]
        payload['features'].update({f:cc['features'][f] for f in ss.feature.unique()})
        ff=read_json(extra/'continuous_functions.json');payload['functions'].update({f:ff[f] for f in ss.feature.unique() if f in ff})
        payload['supplementary_transport_manifest_sha256']=digest(extra/'manifest.json')
    if influence:
        extra=Path(influence);mm=read_json(extra/'manifest.json')
        if mm['window']!=m['window'] or mm['source_manifest_sha256']!=digest(mining/'manifest.json'):raise ValueError('Influence source/window mismatch')
        dd=pd.read_csv(extra/'whole_window_node_influence.csv');dd=dd[(dd.split=='discovery')&dd.feature.isin(payload['features'])]
        payload['evidence'] += [{'evidence_id':f'influence:discovery:{r.feature}:{r.metric}',**r._asdict()} for r in dd.itertuples(index=False)]
        payload['node_influence_manifest_sha256']=digest(extra/'manifest.json')
    payload=clean(payload)
    bundle=dest/'coordinate_evidence.json';write_json(bundle,payload)
    if role=='Designer':
        analyst=read_json(dest/'Analyst.coordinate.response.json')
        validate(analyst,ANALYST,payload);payload={**payload,'analyst_document':analyst}
    schema=ANALYST if role=='Analyst' else DESIGNER
    skill=(PROJECT/'skills'/f'coordinate-{role.lower()}'/'SKILL.md').read_text(encoding='utf-8')
    request={'schema_version':'coordinate-1.0','role':role,'bundle_sha256':digest(bundle),'response_schema':schema,
        'messages':[{'role':'system','content':skill+'\nReturn only JSON matching this schema:\n'+json.dumps(schema)},
            {'role':'user','content':'Analyze only supplied discovery evidence; no causal claim or generation.\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)}]}
    write_json(dest/f'{role}.coordinate.request.json',request);return request


def validate(data,schema,bundle):
    jsonschema.validate(data,schema);known={e['evidence_id']:e for e in bundle['evidence']}
    citations=[data] if data['agent']=='Designer' else data['observations']+data['rules']+data['counterevidence']
    for item in citations:
        if set(item['evidence_ids'])-set(known):raise ValueError('Unprovided evidence IDs')
    if data['agent']=='Analyst':
        for rule in data['rules']:
            if rule['feature'] not in bundle['features']:raise ValueError('Unprovided feature')
            if not any(known[e].get('feature')==rule['feature'] for e in rule['evidence_ids']):raise ValueError('Rule/feature evidence mismatch')
    else:
        if data['window']!=bundle['window']:raise ValueError('Design/learning window mismatch')
        if not all(np.isfinite(data[k]) for k in ('native_rms_ratio','mixture_temperature','robust_delta')):raise ValueError('Nonfinite control parameter')
        if data.get('initial_update_dose')=='cap_to_flow' and data.get('dose_reference','observed_native')!='observed_native':raise ValueError('Initial cap requires native-dose parent')
        if data['architecture']=='selection_contrast' and ('contrast_bound_nats' not in data or not np.isfinite(data['contrast_bound_nats']) or data['contrast_bound_nats']<=0):raise ValueError('Contrast bound must be supplied and finite')
        known_regions={v['region'] for v in bundle['features'].values()}
        if set(data['regions'])-known_regions:raise ValueError('Unmeasured region')
        # An exploratory design is allowed, but its regional observables must be supplied.
        for r in data['regions']:
            kinds=tuple('shape_'+k for k in ('xx','yy','zz','xy','xz','yz')) if data['architecture']=='shape_mixture' else ('spread',) if data['architecture']=='spread_upper' else ('centroid_x','centroid_y','centroid_z','spread')
            prefix='proposal_' if bundle.get('control_representation')=='proposal' else ''
            needed=[f'{r}::{data["channel"]}::{prefix+k}' for k in kinds]
            if any(f not in bundle['features'] for f in needed):raise ValueError('Missing controlled observable')
    return True


def import_response(request_path,response_path,mining):
    request=read_json(request_path);dest=Path(mining)/'agents';bundle=dest/'coordinate_evidence.json'
    if digest(bundle)!=request['bundle_sha256']:raise ValueError('Evidence changed since request')
    data=read_json(response_path);validate(data,request['response_schema'],read_json(bundle))
    role=request['role'];write_json(dest/f'{role}.coordinate.response.json',data)
    write_json(dest/f'{role}.coordinate.validation.json',{'schema_valid':True,'grounding_valid':True,
        'request_sha256':digest(request_path),'response_sha256':digest(response_path),'bundle_sha256':digest(bundle)})
    return data


def compile_design(mining,dataset,campaign,output,round_number):
    from ..generation.coordinate_reference import build
    dest=Path(mining)/'agents';bundle=read_json(dest/'coordinate_evidence.json');design=read_json(dest/'Designer.coordinate.response.json')
    validate(design,DESIGNER,bundle)
    if design['design_status']!='exploratory':raise ValueError('Deferred design cannot compile')
    if not 1<=round_number<=30:raise ValueError('Round outside authorized bound')
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    if design['architecture']=='shape_mixture':
        from ..generation.coordinate_shape import build
    reference=out/'reference.json.gz';build(dataset,campaign,mining,reference,design['regions'],design['channel'])
    program={k:design[k] for k in ('window','native_rms_ratio','mixture_temperature','robust_delta')}
    program.update({k:design[k] for k in ('dose_reference','preserve_native_rigid_pose','initial_update_dose','contrast_bound_nats') if k in design})
    program.update(schema_version='current-coordinate-program-1.0',family='LLM_evidence_bound_coordinate_design',
        reward_view=design['architecture'],reference_sha256=digest(reference),core_radius_A=5.,round=round_number,seed=42,
        evidence_sha256=digest(dest/'coordinate_evidence.json'),designer_sha256=digest(dest/'Designer.coordinate.response.json'),
        constraints={'backtrack_attempts':7,'max_atom_step_A':.025,'max_cumulative_rms_A':1.25,
            'max_pair_distance_change_A':.06,'severe_receptor_clash_A':.8})
    write_json(out/'program.json',program);return program
