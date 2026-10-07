"""Literal-role ablations: advice changes, contracts and evidence stay fixed.

Unlike the historical compliance audits, no expected winner, dose or formula
is prescribed. Compilation preserves the submitted choices without repair.
"""
import copy
import json
import math
from pathlib import Path
import jsonschema
from ..io import read_json, write_json, digest
from ..skill_guidance import render, modules, sha
from ..generation.window_reference import load_reference
from ..generation.coordinate_contrast import make_coordinate_reward

ANALYST_SCHEMA = {
    'type':'object','additionalProperties':False,
    'required':['agent','request_sha256','observations','hypotheses','counterevidence','limitations'],
    'properties':{
        'agent':{'const':'Analyst'},'request_sha256':{'type':'string'},
        'observations':{'type':'array','items':{'type':'object','additionalProperties':False,
            'required':['evidence_ids','finding','status'], 'properties':{
                'evidence_ids':{'type':'array','minItems':1,'items':{'type':'string'}},
                'finding':{'type':'string'},'status':{'enum':['supported','exploratory','unsupported']}}}},
        'hypotheses':{'type':'array','items':{'type':'string'}},
        'counterevidence':{'type':'array','items':{'type':'object','additionalProperties':False,
            'required':['evidence_ids','finding'],'properties':{
                'evidence_ids':{'type':'array','minItems':1,'items':{'type':'string'}},'finding':{'type':'string'}}}},
        'limitations':{'type':'array','items':{'type':'string'}}}}

DESIGNER_SCHEMA = {
    'type':'object','additionalProperties':False,
    'required':['agent','request_sha256','analyst_response_sha256','decision','base_program_id','updates','evidence_ids','rationale','failure_modes'],
    'properties':{
        'agent':{'const':'Designer'},'request_sha256':{'type':'string'},'analyst_response_sha256':{'type':'string'},
        'decision':{'enum':['retain_existing','modify_existing','defer']},
        'base_program_id':{'type':['string','null']},'updates':{'type':'object'},
        'evidence_ids':{'type':'array','minItems':1,'items':{'type':'string'}},
        'rationale':{'type':'string'},'failure_modes':{'type':'array','items':{'type':'string'}}}}

UPDATES = {
    'native_rms_ratio':(.1,3.),'teacher_neighbors':(1,28),
    'teacher_score_beta':(0.,8.),'teacher_endpoint_temperature_A2':(.25,32.),
    'mixture_temperature':(.05,4.),'pointcloud_delta_A':(.25,4.),
    'time_ramp_power':(0.,2.),'history_strength':(0.,4.),'history_time_power':(0.,4.),
    'coordinate_region_radius_A':(1.,10.),'coordinate_background_weight':(.05,2.)}


def variants():
    """Factorial role replacement plus one-at-a-time removal from full advice."""
    frozen=Path(__file__).resolve().parents[3]/'docs/experiments/skill_ablation_20261007/study_v1/treatment_modules.json'
    advice=read_json(frozen) if frozen.exists() else {r:modules(r) for r in ('Analyst','Designer')}
    all_a=list(advice['Analyst']);all_d=list(advice['Designer'])
    result=[dict(id='legacy_legacy',analyst='legacy',designer='legacy',a=[],d=[]),
            dict(id='compact_legacy',analyst='compact',designer='legacy',a=[],d=[]),
            dict(id='legacy_compact',analyst='legacy',designer='compact',a=[],d=[]),
            dict(id='compact_compact',analyst='compact',designer='compact',a=[],d=[]),
            dict(id='full_advice',analyst='compact',designer='compact',a=all_a,d=all_d)]
    for name in all_a:
        result.append(dict(id='without_analyst_'+name,analyst='compact',designer='compact',a=[n for n in all_a if n!=name],d=all_d))
    for name in all_d:
        result.append(dict(id='without_designer_'+name,analyst='compact',designer='compact',a=all_a,d=[n for n in all_d if n!=name]))
    return result


def instruction(role, variant, root):
    if variant[role.lower()]=='compact':
        if variant['id']=='standard_core':return render(role,(),root)
        frozen=Path(root)/'docs/experiments/skill_ablation_20261007/study_v1'
        if (frozen/(role+'.core.md')).exists() and (frozen/'treatment_modules.json').exists():
            content=(frozen/(role+'.core.md')).read_text(encoding='utf-8').rstrip()
            advice=read_json(frozen/'treatment_modules.json')[role]
            for name in variant['a' if role=='Analyst' else 'd']:
                content+='\n\n## Guidance: '+name+'\n\n'+advice[name]
            return content+'\n'
        return render(role,variant['a' if role=='Analyst' else 'd'],root)
    archive=Path(root)/'docs/experiments/skill_ablation_20261007/legacy_skills'
    names=['coordinate-analyst','continuous-analyst'] if role=='Analyst' else ['affinity-endpoint-pullback','affinity-regional-pointcloud','affinity-terminal-lineage']
    return '\n\n'.join((archive/(n+'.md')).read_text(encoding='utf-8') for n in names)


def export_request(root, evidence, folder, variant, role, analyst=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    packet=read_json(evidence);content=instruction(role,variant,root)
    binding={'evidence_path':str(Path(evidence).resolve())}
    if variant['id']=='standard_core':binding['scalar_update_bounds']=UPDATES
    if role=='Designer':
        if analyst is None:raise ValueError('Designer requires actual Analyst response')
        validate_response(analyst.parent/'Analyst.request.json',analyst,evidence)
        binding['analyst_path']=str(Path(analyst).resolve())
        binding['analyst_response_sha256']=digest(analyst)
        capabilities=Path(evidence).parent/'formula_registry.json'
        if capabilities.exists():
            binding['formula_registry_path']=str(capabilities.resolve())
            binding['formula_registry_sha256']=digest(capabilities)
    schema=ANALYST_SCHEMA if role=='Analyst' else DESIGNER_SCHEMA
    request={'schema_version':'literal-skill-ablation-1.0','role':role,'variant':variant,
        'evidence_sha256':digest(evidence),'instruction_sha256':sha(content),
        'instruction_hash_scope':'Role treatment before the shared output-format suffix',
        'response_schema':schema,'bindings':binding,
        'system_instruction':content+'\nReturn JSON matching the supplied response schema.'}
    path=folder/(role+'.request.json');write_json(path,request)
    (folder/(role+'.instructions.md')).write_text(content,encoding='utf-8')
    return path


def payload(request):
    """Hydrate once at call time; shared large evidence is never duplicated on disk."""
    path=Path(request['bindings']['evidence_path'])
    if digest(path)!=request['evidence_sha256']:raise ValueError('Evidence changed')
    packet=read_json(path)
    value={k:packet[k] for k in ('task','execution_contract','evidence','program_registry','incumbent_program_id')}
    if 'scalar_update_bounds' in request['bindings']:
        value['execution_contract']['allowed_scalar_updates']=request['bindings']['scalar_update_bounds']
    if 'analyst_path' in request['bindings']:
        path=Path(request['bindings']['analyst_path'])
        if digest(path)!=request['bindings']['analyst_response_sha256']:raise ValueError('Analyst response changed')
        value['analyst_document']=read_json(path)
        value['analyst_response_sha256']=digest(path)
    if 'formula_registry_path' in request['bindings']:
        path=Path(request['bindings']['formula_registry_path'])
        if digest(path)!=request['bindings']['formula_registry_sha256']:raise ValueError('Formula registry changed')
        value['execution_capabilities']=read_json(path)
    return value


def validate_response(request_path,response_path,evidence):
    request=read_json(request_path);response=read_json(response_path)
    if digest(evidence)!=request['evidence_sha256']:raise ValueError('Evidence changed')
    jsonschema.validate(response,request['response_schema'])
    if response['request_sha256']!=digest(request_path):raise ValueError('Wrong request binding')
    values=payload(request);known={e['evidence_id'] for e in values['evidence']}
    entries=response['observations']+response['counterevidence'] if response['agent']=='Analyst' else [response]
    if any(set(e['evidence_ids'])-known for e in entries):raise ValueError('Unknown evidence citation')
    if response['agent']=='Designer':
        if response['analyst_response_sha256']!=values['analyst_response_sha256']:
            raise ValueError('Designer did not consume bound Analyst output')
        if response['decision']=='defer':
            if response['base_program_id'] is not None or response['updates']:raise ValueError('Deferred response cannot silently execute a reward')
        else:
            registry=values['program_registry']
            if response['base_program_id'] not in registry:raise ValueError('Unknown registered program')
            if response['decision']=='retain_existing' and response['updates']:raise ValueError('Retain cannot modify')
            if response['decision']=='modify_existing' and not response['updates']:raise ValueError('Modify requires literal changes')
            for k,v in response['updates'].items():
                if k not in UPDATES or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
                    raise ValueError('Unregistered finite parameter update')
                if not UPDATES[k][0]<=v<=UPDATES[k][1]:raise ValueError('Parameter outside shared execution bounds')
                if k=='teacher_neighbors' and int(v)!=v:raise ValueError('Neighbor count must be integer')
                family=registry[response['base_program_id']]['reward_view']
                if k.startswith('history_') and family!='endpoint_supported_attractor':raise ValueError('Inactive history parameter')
                if k.startswith('coordinate_') and family!='endpoint_regional_pointcloud':raise ValueError('Inactive regional parameter')
    return response


def compile_response(root,request_path,response_path,evidence,output):
    response=validate_response(request_path,response_path,evidence)
    if response['decision']=='defer':return None
    request=read_json(request_path);entry=payload(request)['program_registry'][response['base_program_id']]
    source=Path(root)/entry['program_path'];reference=Path(root)/entry['reference_path']
    if digest(source)!=entry['program_sha256'] or digest(reference)!=entry['reference_sha256']:
        raise ValueError('Registered source changed')
    program=copy.deepcopy(read_json(source));program.update(response['updates'])
    if program['derivative_path']!='flowr_endpoint_vjp' or program['affinity_head_gradient'] or program['additional_per_step_affinity_calls']:
        raise ValueError('Execution contract mismatch')
    loaded=load_reference(reference)
    if program['window']!=loaded['window']:raise ValueError('Learned support mismatch')
    make_coordinate_reward(program,loaded)
    program['skill_ablation_provenance']={'request_sha256':digest(request_path),'response_sha256':digest(response_path),
        'analyst_response_sha256':response['analyst_response_sha256'],'instruction_sha256':request['instruction_sha256']}
    write_json(output,program)
    return program


def effective_signature(program):
    """Conservative identity: discard only enumerated non-execution metadata.

    Unknown fields remain in the signature, preventing an incomplete parameter
    registry from incorrectly merging distinct generated trajectories.
    """
    excluded={'round','program_id','derivation','skill_ablation_provenance','seed',
        'head_native_labels_relative_path','target_definition','objective_profile','family',
        'topology_policy','geometry_block_weights_role','record_reward_response'}
    inputs={'reference_sha256','coordinate_prior_sha256','region_prior_sha256','kinematics_prior_sha256'}
    result={k:v for k,v in program.items() if k not in excluded and
        (not k.endswith('_sha256') or k in inputs)}
    if program['reward_view'].endswith('pointcloud'):result.setdefault('pointcloud_delta_A',1.)
    return sha(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)),result
