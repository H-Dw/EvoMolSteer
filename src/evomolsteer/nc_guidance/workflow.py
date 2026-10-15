"""Real-API local-patch decision and deterministic reward compilation."""
import copy
import hashlib
import json
import jsonschema
from pathlib import Path

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj
from evomolsteer.final_guidance.reward import active_updates
from .reward import NCReward


def numerical_signature(program,teacher=None):
    excluded={'program_id','analyst_sha256','designer_sha256','source_round','patch_id','template_id',
        'candidate_id','teacher_sha256','counterfactual_labels','schema_version'}
    value={k:v for k,v in program.items() if k not in excluded}
    if not program['weights']['orientation']:value.pop('anchors',None)
    if not program['weights']['type']:
        for key in ('type_mode','type_site_weights','type_ramp'):value.pop(key,None)
    if sum(v>0 for v in program['weights'].values())==1:value.pop('gradient_balance',None)
    if teacher is not None:
        value['half_targets']=[teacher['coords_A'][s] for s in program['atom_slots']]
        value['final_targets']=[teacher['terminal_coords_A'][s] for s in program['atom_slots']]
        value['NC_vocabulary']={e:teacher['atom_vocabulary'][e] for e in ('N','C')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compile_program(cfg,patch,teacher,analyst,designer,teacher_sha,analyst_sha,designer_sha):
    contract=cfg['execution_contract']
    if designer['execution_contract']!=contract or designer['patch_id']!=patch['patch_id']:raise ValueError('Designer detached from Analyst patch or execution mode')
    if analyst['patch_id']!=patch['patch_id'] or analyst['evidence_id']!=patch['evidence_id']:raise ValueError('Analyst evidence mismatch')
    if designer['decision']=='defer':return None
    p={'schema_version':'local-nc-1','family':'local_nc_layout','scope':'local','program_id':cfg['round_id'],
        'candidate_id':teacher['candidate_id'],'template_id':patch['template_id'],'patch_id':patch['patch_id'],
        'atom_slots':patch['atom_slots'],'typed_slots':patch['typed_slots'],'typed_elements':patch['typed_elements'],
        'anchors':patch['anchors'],'max_local_points':cfg['max_local_points'],'teacher_sha256':teacher_sha,
        'teacher_time':cfg['teacher_time'],'teacher_representation':'current_world_A','steps':cfg['steps'],'terminal_time':1.,
        'window':contract['window'],'expected_active_steps':active_updates(cfg['steps'],*contract['window']),
        **{k:contract[k] for k in ['weights','dose','type_mode','gradient_balance','assignment_endpoint_fraction',
            'assignment_type_cost','type_ramp','anchor_distance_A','anchor_width_A']},
        'background_loss_weight':0,'smc':False,'affinity_head_gradient':False,'coordinate_updates_only':True,
        'analyst_sha256':analyst_sha,'designer_sha256':designer_sha,'counterfactual_labels':teacher['counterfactual_labels']}
    if cfg.get('contrast_weighting',False):
        import numpy as np
        a=np.array([max(abs(s['N_minus_C']),.05)**.5 for s in patch['site_evidence']]);p['type_site_weights']=(a/a.sum()).tolist()
    NCReward(p,teacher);return p


def analyze(repo,cfg,feedback):
    repo=Path(repo);study=repo/cfg['campaign_study'];packet=read_json(study/'evidence.json')
    patches=packet['patches'];allowed=patches
    if cfg.get('r13_context_only',False):allowed=[p for p in allowed if p.get('r13_context',False)]
    elif not cfg.get('allow_r13_context',False):allowed=[p for p in allowed if not p.get('r13_context',False)]
    allowed=[p for p in allowed if len(p['atom_slots'])<=cfg['max_local_points']]
    if cfg.get('template_filter'):allowed=[p for p in allowed if cfg['template_filter'](p)]
    # All serializable eligibility constraints are explicit runtime metadata.
    if cfg.get('template_kind'):allowed=[p for p in allowed if p['counterfactual_labels']==(cfg['template_kind']=='counterfactual')]
    if cfg.get('candidate_node'):allowed=[p for p in allowed if p['node_id']==cfg['candidate_node']]
    if cfg.get('minimum_N',0):allowed=[p for p in allowed if p['typed_elements'].count('N')>=cfg['minimum_N']]
    if cfg.get('minimum_C',0):allowed=[p for p in allowed if p['typed_elements'].count('C')>=cfg['minimum_C']]
    if cfg.get('require_compensation'):allowed=[p for p in allowed if p.get('joint_layout_compensation',False)]
    if cfg['execution_contract']['weights']['orientation']>0:allowed=[p for p in allowed if p['anchors']]
    if cfg.get('frozen_patch'):allowed=[p for p in allowed if p['patch_id']==cfg['frozen_patch']]
    if cfg.get('exclude_patches'):allowed=[p for p in allowed if p['patch_id'] not in cfg['exclude_patches']]
    if not allowed:raise ValueError('No supported local patches for declared intervention')
    folder=repo/cfg['study'];folder.mkdir(parents=True,exist_ok=True);text={'type':'string','maxLength':2800}
    skills={r:(repo/cfg[r.lower()+'_skill']).read_text(encoding='utf-8') for r in ('Analyst','Designer')}
    schema=obj({'role':{'const':'Analyst'},'patch_id':{'enum':[p['patch_id'] for p in allowed]},
        'evidence_id':{'type':'string'},'hypothesis':text,'comparison_and_counterexample':text,
        'expected_NC_and_geometry_response':text,'falsifier':text})
    def validate_a(a):
        patch=next(p for p in allowed if p['patch_id']==a['patch_id'])
        if a['evidence_id']!=patch['evidence_id']:raise jsonschema.ValidationError('Unknown patch evidence',validator='evidence_binding',validator_value=patch['evidence_id'])
    analyst=call_json([{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'intervention':cfg['intervention'],'patches':allowed,'candidate_summaries':packet['candidate_summaries'],
        'report_limits':packet['report_limits'],'support_definition':packet.get('support_definition'),
        'feedback':feedback,'frozen_patch':cfg.get('frozen_patch'),
        'frozen_parameters':cfg['execution_contract'],'task':'Select one supported local N/C patch; do not copy slot lists.'},sort_keys=True,separators=(',',':'))}],
        schema,folder/'Analyst.json',input_store=study/'inputs',max_output_tokens=3500,response_validator=validate_a,max_format_repairs=2)
    patch=next(p for p in allowed if p['patch_id']==analyst['patch_id']);templates=read_json(study/'templates.json');teacher=templates[patch['template_id']]
    target=folder/'candidates'/teacher['candidate_id'];target.mkdir(parents=True,exist_ok=True);write_json(target/'teacher.json',teacher)
    write_json(folder/'selection.json',patch)
    schema=obj({'role':{'const':'Designer'},'patch_id':{'const':patch['patch_id']},'decision':{'enum':['activate','defer']},
        'execution_contract':{'const':cfg['execution_contract']},'scientific_rationale':text,'observable_falsifier':text})
    selected={'half_coords_A':[teacher['coords_A'][s] for s in patch['atom_slots']],
        'final_coords_A':[teacher['terminal_coords_A'][s] for s in patch['atom_slots']],
        'slots':patch['atom_slots'],'counterfactual_labels':teacher['counterfactual_labels'],'teacher_time':teacher['time']}
    designer=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
        'analyst':analyst,'patch':patch,'selected_teacher':selected,'execution_contract':cfg['execution_contract'],
        'direct_geometry_slots':patch['atom_slots'] if cfg['execution_contract']['weights']['half']+cfg['execution_contract']['weights']['endpoint']>0 else [],
        'direct_type_slots':patch['typed_slots'] if cfg['execution_contract']['weights']['type']>0 else [],
        'task':'Bind the Analyst local N/C targets to enabled registered terms. Never add other points.'},sort_keys=True,separators=(',',':'))}],
        schema,folder/'Designer.json',input_store=study/'inputs',max_output_tokens=3500,max_format_repairs=2)
    p=compile_program(cfg,patch,teacher,analyst,designer,digest(target/'teacher.json'),digest(folder/'Analyst.json'),digest(folder/'Designer.json'))
    if p is None:raise ValueError('Designer deferred the supported local experiment')
    path=target/'programs'/(p['program_id']+'.json');write_json(path,p)
    write_json(target/'bindings.json',{'programs':[{'program':path.relative_to(repo).as_posix(),'sha256':digest(path),
        'operative_text':f"LOCAL {patch['patch_id']}; ONLY slots {p['atom_slots']}; typed {list(zip(p['typed_slots'],p['typed_elements']))}; weights {p['weights']}; coordinates-only VJP; window {p['window']}; no affinity gradient or SMC"}],
        'teacher':(target/'teacher.json').relative_to(repo).as_posix(),
        'instruction_sha256':{r:hashlib.sha256(s.encode()).hexdigest() for r,s in skills.items()},
        'analyst_patch_sha256':digest(folder/'selection.json'),'designer_bound_to_analyst':True})
    return p
