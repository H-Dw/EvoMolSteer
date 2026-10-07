"""Small discovery evidence packets and strict motif Analyst/Designer interfaces."""
import copy,json
from pathlib import Path
import jsonschema
import numpy as np
import pandas as pd
from .coordinate_agents import ANALYST as BASE_ANALYST,DESIGNER as BASE_DESIGNER
from ..io import read_json,digest,clean
from ..generation.prototypes import write_json

PROJECT=Path(__file__).resolve().parents[3]
ANALYST=copy.deepcopy(BASE_ANALYST);ANALYST['properties']['schema_version']={'const':'motif-agent-1.0'}
DESIGNER=copy.deepcopy(BASE_DESIGNER)
DESIGNER['properties'].update(schema_version={'const':'motif-agent-1.0'},
    architecture={'enum':['motif_mixture','motif_contrast']},channel={'enum':['all','NOS','NOS_C']},
    motif_components={'enum':['joint','pair','shell','centroid']},time_ramp_power={'type':'number','minimum':0,'maximum':4},
    target_definition={'enum':['instantaneous','boundary_survival']})
DESIGNER['required']+=['motif_components','time_ramp_power']


def export(mining,role,output,top_per_metric=6,boundary=None):
    mining,output=Path(mining),Path(output);output.mkdir(parents=True,exist_ok=True)
    if role not in ('Analyst','Designer') or not 1<=top_per_metric<=16:raise ValueError('Bounded role/evidence contract')
    m,cat=read_json(mining/'manifest.json'),read_json(mining/'feature_catalog.json')
    if m.get('feature_family')!='motif':raise ValueError('Measured motif input required')
    d=pd.read_csv(mining/'whole_window_evidence.csv');d=d[(d.split=='discovery')&d.feature.str.contains('::proposal_motif_')]
    selected=d.sort_values(['q','feature']).groupby('metric',sort=True).head(top_per_metric)
    rows=[{'evidence_id':f'motif:discovery:{r.feature}:{r.metric}',**r._asdict()} for r in selected.itertuples(index=False)]
    counts={metric:{'n_tested':len(g),'q_below_05':int(g.q.lt(.05).sum()),'min_q':float(g.q.min()),'min_coverage':float(g.mean_time_coverage.min())} for metric,g in d.groupby('metric')}
    rows.append({'evidence_id':'M_METRIC_COUNTS','counts':counts})
    diag=pd.read_csv(mining/'lineage_diagnostics.csv');last=diag[np.isclose(diag.time,m['window'][1])]
    rows.append({'evidence_id':'M_LINEAGE','end_mean_roots':float(last.unique_roots.mean()),'end_batches_one_root':int(last.unique_roots.eq(1).sum())})
    fits=read_json(mining/'continuous_functions.json')
    bundle=clean({'schema_version':'motif-evidence-1.0','window':m['window'],'times':m['times'],'discovery_batches':m['splits']['discovery'],
        'source_manifest_sha256':digest(mining/'manifest.json'),'evidence':rows,
        'features':{k:v for k,v in cat['features'].items() if '::proposal_motif_' in k},
        'functions':{k:fits[k] for k in selected.feature.unique() if k in fits},'regions':cat['regions'],
        'spatial_anchor':m['spatial_anchor'],'control_representation':m['control_representation'],
        'interpretation':m['interpretation']+['Top rows are a compact packet, not the full tested family. Read metric counts; a non-significant direction cannot become a discovered affinity mechanism.'],
        'storage':'No structures, trajectories, per-particle or per-edge feature table in LLM input'})
    if boundary is not None:
        boundary=Path(boundary);bm=read_json(boundary/'manifest.json')
        if bm['source_manifest_sha256']!=bundle['source_manifest_sha256'] or bm['window']!=bundle['window']:raise ValueError('Boundary evidence scope mismatch')
        bd=pd.read_csv(boundary/'whole_window_evidence.csv')
        top=bd.sort_values(['global_curve_q','q','feature']).groupby('metric',sort=True).head(top_per_metric)
        bundle['evidence'] += clean([{'evidence_id':f'boundary:discovery:{r.feature}:{r.metric}',**r._asdict()} for r in top.itertuples(index=False)])
        bundle['evidence'].append({'evidence_id':'M_BOUNDARY_COUNTS','counts':{
            metric:{'n_tested':len(g),'integrated_q_below_05':int(g.q.lt(.05).sum()),'whole_curve_q_below_05':int(g.global_curve_q.lt(.05).sum())}
            for metric,g in bd.groupby('metric')}})
        bf=read_json(boundary/'effect_functions.json')
        bundle['boundary_evidence']={'manifest_sha256':digest(boundary/'manifest.json'),
            'target':bm['target'],'observed_score_times':bm['observed_score_times'],'observed_proposal_boundary':bm['observed_proposal_boundary'],
            'functions':{metric:{f:bf[metric][f] for f in g.feature} for metric,g in top.groupby('metric')},
            'interpretation':'Retrospective surviving ancestry; clones are weights, no final labels; whole-curve and integrated tests are separate exploratory families'}
    bundlepath=output/'motif_evidence.json'
    if bundlepath.exists() and read_json(bundlepath)!=bundle:raise ValueError('Immutable evidence packet already differs')
    write_json(bundlepath,bundle);payload={'evidence':bundle}
    if role=='Designer':
        analyst=read_json(output/'Analyst.motif.response.json');validate(analyst,ANALYST,bundle);payload['analyst']=analyst
    schema=ANALYST if role=='Analyst' else DESIGNER
    from ..skill_guidance import render
    skill=render(role)
    request={'schema_version':'motif-agent-1.0','role':role,'bundle_sha256':digest(bundlepath),'response_schema':schema,
        'messages':[{'role':'system','content':skill+'\nReturn only JSON matching this schema:\n'+json.dumps(schema)},
                    {'role':'user','content':'Use supplied discovery evidence only. Propose code-defined hypotheses with counterevidence and no graph veto.\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)}]}
    write_json(output/f'{role}.motif.request.json',request);return request


def validate(data,schema,bundle):
    jsonschema.validate(data,schema);known={e['evidence_id']:e for e in bundle['evidence']}
    statements=[data] if data['agent']=='Designer' else data['observations']+data['rules']+data['counterevidence']
    if any(set(v['evidence_ids'])-set(known) for v in statements):raise ValueError('Unknown evidence IDs')
    if data['agent']=='Analyst':
        for r in data['rules']:
            if r['feature'] not in bundle['features'] or not any(known[e].get('feature')==r['feature'] for e in r['evidence_ids']):raise ValueError('Unmeasured rule/feature')
    else:
        if data['window']!=bundle['window'] or set(data['regions'])-set(bundle['regions']):raise ValueError('Unmeasured window/regions')
        if data.get('target_definition','instantaneous')=='boundary_survival' and 'boundary_evidence' not in bundle:raise ValueError('Measured actual-boundary evidence required')
        if data.get('initial_update_dose')=='cap_to_flow' and data.get('dose_reference','predictive_flow')!='observed_native':raise ValueError('Initial cap requires observed-native calibration')
        for key in ('native_rms_ratio','mixture_temperature','robust_delta','time_ramp_power'):
            if not np.isfinite(data[key]):raise ValueError('Nonfinite reward parameter')
        if data['architecture']=='motif_contrast' and (not np.isfinite(data.get('contrast_bound_nats',np.nan)) or data['contrast_bound_nats']<=0):raise ValueError('Finite positive contrast bound required')
        for region in data['regions']:
            needed=[f'{region}::{data["channel"]}::proposal_motif_{kind}' for kind in ('centroid_x','centroid_y','centroid_z','shell_2','shell_4','shell_6','pair_1_5','pair_3','pair_5')]
            if any(f not in bundle['features'] for f in needed):raise ValueError('Missing controlled motif observables')
    return True


def import_response(request_path,response_path,output):
    output=Path(output);req=read_json(request_path);bundlepath=output/'motif_evidence.json'
    if digest(bundlepath)!=req['bundle_sha256']:raise ValueError('Evidence changed after export')
    data=read_json(response_path);validate(data,req['response_schema'],read_json(bundlepath))
    write_json(output/f'{req["role"]}.motif.response.json',data)
    write_json(output/f'{req["role"]}.motif.validation.json',{'schema_valid':True,'grounding_valid':True,
        'request_sha256':digest(request_path),'response_sha256':digest(output/f'{req["role"]}.motif.response.json'),
        'source_response_sha256':digest(response_path),'bundle_sha256':digest(bundlepath)})
    return data


def compile_design(response,reference,output,number):
    from ..generation.window_reference import load_reference
    from ..generation.motif_reward import MotifReward
    response,reference=Path(response),Path(reference);d=read_json(response);ref=load_reference(reference)
    if not isinstance(number,int) or not 1<=number<=15:raise ValueError('Explicit fifteen-round program number required')
    # Imported validation is mandatory; arbitrary LLM Python is never executed.
    validation=read_json(response.parent/'Designer.motif.validation.json')
    if not validation['grounding_valid'] or validation['response_sha256']!=digest(response):raise ValueError('Validated Designer response required')
    bundlepath=response.parent/'motif_evidence.json';bundle=read_json(bundlepath)
    if digest(bundlepath)!=validation['bundle_sha256'] or ref['source_manifest_sha256']!=bundle['source_manifest_sha256']:raise ValueError('Frozen evidence/reference provenance mismatch')
    if d['design_status']!='exploratory' or set(d['regions'])!=set(ref['regions']) or ref['channels']!=[d['channel']]:raise ValueError('Reference/design scope mismatch or deferred design')
    if d.get('target_definition','instantaneous')!=ref.get('target_definition','instantaneous'):raise ValueError('Reference/design lineage target mismatch')
    p={'schema_version':'current-coordinate-program-1.0','family':'LLM_evidence_bound_spatial_motif_design','round':number,'seed':42,
        'window':d['window'],'reference_sha256':digest(reference),'reward_view':d['architecture'],'motif_components':d['motif_components'],
        'native_rms_ratio':d['native_rms_ratio'],'mixture_temperature':d['mixture_temperature'],'robust_delta':d['robust_delta'],
        'time_ramp_power':d['time_ramp_power'],'dose_reference':d.get('dose_reference','predictive_flow'),'core_radius_A':5.,
        'contrast_bound_nats':d.get('contrast_bound_nats',2.),'designer_sha256':digest(response),
        'target_definition':d.get('target_definition','instantaneous'),
        'constraints':{'max_atom_step_A':.025,'max_cumulative_rms_A':1.25,'severe_receptor_clash_A':.8,'backtrack_attempts':7}}
    for key in ('initial_update_dose','preserve_native_rigid_pose'):
        if key in d:p[key]=d[key]
    MotifReward(p,ref);write_json(output,p);return p
