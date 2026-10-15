"""Sequential, evidence-bound local-guidance research with real LLM calls.

The task file defines ONE intervention family per round. Models query registered
evidence and select a local hypothesis and a numeric variant. No SMC/head VJP.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import jsonschema
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evomolsteer.io import read_json, write_json, write_table, digest
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj
from evomolsteer.final_guidance.reward import active_updates
from evomolsteer.partial_search.engine import Remote, infer, local, emit
from evomolsteer.partial_search.campaign import condition_metrics
from evomolsteer.partial_search.report import paired_comparison, api_audit
from .reward import NCReward
from .workflow import numerical_signature

NATIVE_DEFAULTS={'coord_noise_level':.2,'cat_noise_level':1.,'cat_noise_euler_guard':False,
                 'use_sde_simulation':True,'final_corr_pred':True}
PARAMETER_SEMANTICS={
    'assignment_type_cost':'Coefficient in DETACHED Hungarian assignment cost (A^2 per log-probability). Changes matched indices; NOT a type-gradient coefficient and has no derivative through the discrete assignment.',
    'dose.native_rms_ratio':'Requested coordinate RMS dose relative to predictive native-flow increment, before clipping. Not a mixture weight.',
    'weights':'Relative normalized gradient mixture; total physical dose remains separately fixed.',
    'weights.pair':'Adds a local endpoint pair-distance gradient to the mixture; does not freeze the graph.',
    'type_ramp':'Within-window scaling of type/orientation contributions, not guidance after the window.',
    'native_options.cat_noise_level':'Categorical transition noise during native generation, including suffix. Does not overwrite or freeze labels.',
    'native_options.final_corr_pred':'Select final predicted output versus last integrated current. Changes terminal output protocol, not control window.',
    'geometry_mode':'Aggregate versus per-point robust geometry changes relative spatial gradients.',
    'assignment_endpoint_fraction':'Mixture of half/endpoint spatial costs for the detached correspondence.',
    'type_confidence_weighting':'Continuously multiply normalized type contribution by detached desired-probability deficit; optimizer weighting, no graph restriction.',
    'patch':'Analyst-local teacher/support/label hypothesis; scalar controller settings stay fixed.',
    'calibration':'Frozen numerical program; no new reward or new source-score promise.'}
TOOLS=['joint_layout_contrasts','local_role_context','native_retention_evidence','window_lineage_features',
       'reward_response_probe','matched_counterexamples']


def factor_value(p,factor):
    if factor=='calibration':return None
    if factor=='patch':return p['patch_id']
    value=p
    for key in factor.split('.'):
        if key not in value:return 'none' if factor=='type_confidence_weighting' else 0.
        value=value[key]
    return value


def set_path(value,path,new):
    keys=path.split('.');target=value
    for key in keys[:-1]:target=target[key]
    target[keys[-1]]=new


def validate_program(p,teacher,learning_window):
    if not learning_window[0]<=p['window'][0]<p['window'][1]<=learning_window[1]:
        raise ValueError('Control outside actual learned window')
    if p['teacher_time']!=teacher['time']:raise ValueError('Teacher time was not propagated')
    native=p['native_options']
    if set(native)!=set(NATIVE_DEFAULTS):raise ValueError('Unknown or missing native option')
    if not 0<=native['cat_noise_level']<=1 or not 0<native['coord_noise_level']<=.4:raise ValueError('Noise outside registered bounds')
    if not 0<p['dose']['native_rms_ratio']<=2:raise ValueError('Dose outside registered bounds')
    if set(p['weights'])-{'half','endpoint','type','orientation','role','pair'}:raise ValueError('Unknown reward term')
    if any(not np.isfinite(v) or v<0 for v in p['weights'].values()) or not sum(p['weights'].values()):raise ValueError('Invalid mixture')
    if p.get('geometry_mode') not in ('aggregate','pointwise'):raise ValueError('Unknown geometry shape')
    if p.get('type_confidence_weighting','none') not in ('none','linear_deficit'):raise ValueError('Unknown type optimizer weighting')
    p['expected_active_steps']=active_updates(p['steps'],*p['window'])
    if not p['expected_active_steps']:raise ValueError('Empty control grid')
    NCReward(p,teacher)
    return p


def execution_identity(p,teacher,cfg,source_identity):
    return hashlib.sha256(json.dumps({'reward':numerical_signature(p,teacher),
        'native':p['native_options'],'batch':cfg['pilot_initial_batch'],'n':cfg['pilot_samples'],
        'seed':cfg['seed'],'source':source_identity},sort_keys=True).encode()).hexdigest()


def prepare(repo,master):
    study=repo/master['campaign_study'];study.mkdir(parents=True,exist_ok=True)
    old=repo/master['prior_study'];packet=read_json(old/'evidence.json');templates=read_json(old/'templates.json')
    patches=copy.deepcopy(packet['patches'])
    source=next(p for p in patches if p['patch_id']=='L109')
    xyz=np.asarray(templates[source['template_id']]['terminal_coords_A'])
    essentials=set(source['typed_slots'])
    for s in source['typed_slots']:essentials.update(templates[source['template_id']]['local_adjacency'].get(str(s),[]))
    distances=np.linalg.norm(xyz[:,None]-xyz[source['typed_slots']][None],axis=-1).min(1)
    for count in (8,12,16):
        selected=set(essentials)
        for s in sorted(source['atom_slots'],key=lambda s:(distances[s],s)):
            if len(selected)>=count:break
            selected.add(s)
        patches.append({**copy.deepcopy(source),'patch_id':f'A{count:02d}','evidence_id':f'A{count:02d}/local_context',
                        'atom_slots':sorted(selected),'claim':'Nested local chemical-role context; measured source geometry, new support hypothesis'})
        swapped=next(p for p in patches if p['patch_id']=='L110')
        patches.append({**copy.deepcopy(swapped),'patch_id':f'S{count:02d}','evidence_id':f'S{count:02d}/local_context','atom_slots':sorted(selected)})
    members=pd.read_csv(Path(master['layout_report'])/'reference_inputs/inputs/frozen_membership.csv')
    quality_nodes=set(members.loc[members.top10,'node_id'])
    atoms=pd.read_csv(Path(master['layout_report'])/'all_atoms.csv.gz')
    native_sources=[p for p in patches if p['node_id'] in quality_nodes and not p['counterfactual_labels'] and len(p['typed_slots'])==2]
    nodes=sorted({p['node_id'] for p in native_sources},key=lambda node:max(p['native_score'] for p in native_sources if p['node_id']==node),reverse=True)[:3]
    for node in nodes:
        choices=[p for p in native_sources if p['node_id']==node]
        src=min(choices,key=lambda p:(-len(p['anchors']),len(p['atom_slots']),p['patch_id']))
        t=templates[src['template_id']];coords=np.asarray(t['terminal_coords_A']);g=atoms[atoms.node_id.eq(node)]
        oxygen=g.loc[g.element.eq('O'),'atom_index'].astype(int).tolist()
        required=set(src['atom_slots'])|set(oxygen)
        centers=coords[sorted(set(src['typed_slots'])|set(oxygen))]
        distances=np.linalg.norm(coords[:,None]-centers[None],axis=-1).min(1)
        for count in (8,12,16,20):
            if len(required)>count or count>=len(coords):continue
            support=set(required)
            for s in np.argsort(distances,kind='stable'):
                if len(support)>=count:break
                support.add(int(s))
            rid=f'Q{t["batch"]:03d}_{count:02d}'
            patches.append({**copy.deepcopy(src),'patch_id':rid,'evidence_id':rid+'/observed_quality_context',
                'atom_slots':sorted(support),'O_context_slots':oxygen,
                'claim':'Observed Quality/Top10 native teacher; local N/C and O spatial context, proper subset, no whole-graph constraint'})
    for t in templates.values():t['bond_vocabulary']={'aromatic':4}
    write_json(study/'templates.json',templates)
    # Do not put large candidate/history tables in every prompt.
    compact=[]
    for p in patches:
        compact.append({k:p[k] for k in ('patch_id','template_id','node_id','atom_slots','typed_slots','typed_elements',
            'anchors','layout_score','native_score','counterfactual_labels','site_evidence')})
        compact[-1]['O_context_slots']=p.get('O_context_slots',[])
        compact[-1]['in_original_TOP10']=p['node_id'] in quality_nodes
    evidence={'patches':compact,'candidate_summaries':packet['candidate_summaries'],
        'source_sha256':packet['source_sha256'],'limits':packet['report_limits'],
        'context':{'O_fields_heldout_increment':'Supported score explanation; local O intervention not established',
            'F_reference_membership':'User-specified extra teacher; not original Quality67/Top10',
            'counterfactual_C':'Compensating replacement, not independent negative N energy'},
        'retention_prior':read_json(old/'R18_identity_replay/identity_report.json')}
    if (study/'energy_probe_reproducible.json').exists():
        requests=read_json(study/'energy_probe_inputs.json');results=read_json(study/'energy_probe_reproducible.json')['cases']
        evidence['source_quality']={q['node_id']:r for q,r in zip(requests,results) if q.get('node_id')}
    prior_path=study/'evidence.json'
    if prior_path.exists():
        version=study/'evidence_versions'/(digest(prior_path)+'.json');version.parent.mkdir(exist_ok=True)
        if not version.exists():version.write_bytes(prior_path.read_bytes())
    write_json(prior_path,evidence)
    return evidence,templates


def variants(parent,task,patch,teacher,master):
    base=copy.deepcopy(parent)
    for key in ('template_id','candidate_id','patch_id','atom_slots','typed_slots','typed_elements','anchors'):
        base[key]=copy.deepcopy(patch[key] if key in patch else teacher[key])
    base['max_local_points']=master['max_local_points'];base['teacher_time']=teacher['time']
    base['counterfactual_labels']=teacher['counterfactual_labels']
    base['native_options']=base.get('native_options',copy.deepcopy(NATIVE_DEFAULTS))
    base['geometry_mode']=base.get('geometry_mode','aggregate')
    result={}
    for i,value in enumerate(task['values']):
        p=copy.deepcopy(base)
        if task['factor']=='weights':p['weights']=copy.deepcopy(value)
        elif task['factor']=='patch':pass
        elif task['factor']!='calibration':set_path(p,task['factor'],value)
        try:validate_program(p,teacher,master['learning_window'])
        except ValueError as error:
            if task['factor']=='weights.orientation' and not p['anchors']:continue
            raise error
        result[f'V{i:02d}']=p
    return result


def compact_history(state):
    return [{'round':r['round'],'factor':r['factor'],'patch':r.get('patch'),'chosen_value':r.get('chosen_value'),
             'actual_dose':r.get('program',{}).get('dose'),'actual_weights':r.get('program',{}).get('weights'),
             'actual_native_options':r.get('program',{}).get('native_options'),
             'mean_pic50':r.get('metrics',{}).get('mean_all_pic50'),'best_pic50':r.get('metrics',{}).get('best_qualified_pic50'),
             'mean_vs_unguided':r.get('mean_vs_unguided'),'realization':r.get('realization'),
             'rollback':r.get('rollback',False),'status':r['status']} for r in state['rounds'][-8:]]


def ask(repo,master,cfg,task,parent,evidence,templates,state):
    folder=repo/cfg['study'];folder.mkdir(parents=True,exist_ok=True)
    allowed=[p for p in evidence['patches'] if p['patch_id'] in task['patches']]
    if not allowed:raise ValueError('No actual patch choices')
    brief=[{'patch_id':p['patch_id'],'template_id':p['template_id'],'n_points':len(p['atom_slots']),
            'targets':list(zip(p['typed_slots'],p['typed_elements'])),'source_score':p['layout_score'],
            'counterfactual':p['counterfactual_labels'],'in_original_TOP10':p.get('in_original_TOP10',False)} for p in allowed]
    skills={r:(repo/master[r.lower()+'_skill']).read_text(encoding='utf-8') for r in ('Analyst','Designer')}
    parent_scores=next((r['metrics'] for r in reversed(state['rounds']) if r.get('program',{}).get('program_id')==parent['program_id']),state['best'])
    parent_history=[r for r in compact_history(state) if any(q.get('program',{}).get('program_id')==parent['program_id'] and q['round']==r['round'] for q in state['rounds'])]
    for role,text in skills.items():(folder/(role+'.SKILL.md')).write_text(text,encoding='utf-8')
    short={'type':'string','maxLength':6000,'description':'Keep the explanation concise; refer to evidence rather than repeating every field.'}
    query=call_json([{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'task':task,'available_patches':brief,'history':compact_history(state),
        'tools':TOOLS,
        'instruction':'Request one registered tool and at most three patch IDs. This is a real deterministic tool lookup.'})}],
        obj({'tool':{'enum':TOOLS},
             'patch_ids':{'type':'array','items':{'enum':[p['patch_id'] for p in allowed]},'minItems':1,'maxItems':3,'uniqueItems':True},
             'question':short}),folder/'Analyst.tool.json',input_store=repo/master['campaign_study']/'inputs',max_output_tokens=1000,max_format_repairs=2)
    tool_rows=[p for p in allowed if p['patch_id'] in query['patch_ids']]
    tool_result={'tool':query['tool'],'source_hash':digest(repo/master['campaign_study']/'evidence.json'),
        'rows':tool_rows,'limits':evidence['limits'],'context':evidence['context']}
    from .research_tools import reward_response_probe
    parent_entry=next((r for r in reversed(state['rounds']) if r.get('program',{}).get('program_id')==parent['program_id']),None)
    response_folder=repo/(parent_entry['result_folder'] if parent_entry else master['initial_results'])
    tool_result['measured_parent_response']=reward_response_probe(response_folder)
    if task['factor']=='patch' or query['tool']=='matched_counterexamples':
        history=compact_history(state)
        tool_result['queried_patch_generated_feedback']=[r for r in history if r['patch'] in query['patch_ids']]
    quality_nodes={p['node_id'] for p in tool_rows}|{templates[parent['template_id']]['evidence_origin']}
    tool_result['source_quality']={k:v for k,v in evidence.get('source_quality',{}).items() if k in quality_nodes}
    if query['tool']=='native_retention_evidence':
        matches=any(p['typed_slots']==[12,22] and p['typed_elements']==['N','C'] for p in tool_rows)
        tool_result['retention_prior_applicability']={'prior_hypothesis':'L110: N at 12 and C at 22',
            'matches_requested_labels':matches,'do_not_transfer_to_native_C12_N22':True}
        if matches:
            tool_result['different_controller_counterexample']={'prior':evidence['retention_prior'],
                'control_difference':'Legacy mixed categorical controller, strength 8, dose .33; not the current coordinate-only parent'}
    from .research_tools import joint_layout_contrasts,local_role_context
    if query['tool']=='joint_layout_contrasts':tool_result['actual_layout_table']=joint_layout_contrasts(master['layout_report'],tool_rows)
    if query['tool']=='local_role_context':tool_result['chemical_context']=local_role_context(master['layout_report'],tool_rows)
    if query['tool']=='window_lineage_features':
        trends=read_json(repo/master['campaign_study']/'window_trends.json')
        candidate_ids={templates[p['template_id']]['candidate_id'] for p in tool_rows}
        tool_result['window_trends']={'domain':trends['domain'],'limits':trends['limits'],
            'fits':[r for r in trends['fits'] if r['candidate_id'] in candidate_ids]}
    write_json(folder/'tool_result.json',tool_result)
    analyst=call_json([{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'task':task,'allowed_patches':brief,'tool_request':query,'actual_tool_result':tool_result,
        'parent':{'patch':parent['patch_id'],'weights':parent['weights'],'dose':parent['dose'],'native':parent['native_options']},
        'parent_generated_scores':{'mean':parent_scores['mean_all_pic50'],'best_qualified':parent_scores['best_qualified_pic50'],
            'meaning':'Measured active generated reference, not the source teacher score'},
        'current_parent_program_id':parent['program_id'],'comparator_history':parent_history,
        'parameter_semantics':PARAMETER_SEMANTICS[task['factor']],'actual_parent_factor_value':factor_value(parent,task['factor']),
        'instruction':'Select a local hypothesis, separate realization from value and identify one falsifiable consequence. Compare to the selected parent; other controllers are contextual counterexamples, not this operating point.'})}],
        obj({'role':{'const':'Analyst'},'patch_id':{'enum':query['patch_ids']},
             'hypothesis':short,'evidence_and_counterexample':short,'expected_response':short,'falsifier':short}),
        folder/'Analyst.json',input_store=repo/master['campaign_study']/'inputs',max_output_tokens=2500,max_format_repairs=2)
    patch=next(p for p in allowed if p['patch_id']==analyst['patch_id']);teacher=templates[patch['template_id']]
    options=variants(parent,task,patch,teacher,master)
    def validate_d(answer):
        if answer['decision']=='defer':return
        expected=sorted(k for k,v in options[answer['variant_id']]['weights'].items() if v>0)
        if answer['enabled_terms']!=expected:
            raise jsonschema.ValidationError('enabled_terms must equal positive-weight executed terms',
                validator='enum',validator_value=expected,instance=answer['enabled_terms'])
        wanted=factor_value(options[answer['variant_id']],task['factor'])
        if answer['chosen_factor_value']!=wanted:
            raise jsonschema.ValidationError('Chosen factor value differs from executable variant',validator='const',validator_value=wanted,instance=answer['chosen_factor_value'])
    designer=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
        'analyst':analyst,'selected_patch':patch,'experiment':task,'allowed_variants':options,
        'parent_generated_scores':{'mean':parent_scores['mean_all_pic50'],'best_qualified':parent_scores['best_qualified_pic50']},
        'current_parent_program_id':parent['program_id'],'comparator_history':parent_history,'fixed_parent_dose':parent['dose'],
        'parameter_semantics':PARAMETER_SEMANTICS[task['factor']],'actual_parent_factor_value':factor_value(parent,task['factor']),
        'instruction':'This is an authorized prospective experiment to improve final affinity. Choose one legal untested variant using a falsifiable hypothesis; lack of previous outcomes at that exact value is the reason to experiment, not a reason to defer. Defer for missing necessary chemistry, mathematical invalidity or a demonstrated ineffective operator. Source layout_score is NOT a generated score forecast; zero-weight terms are inactive. Compare only to the declared parent, not a different recent round. Explain direction, dose and expected measurable response.'})}],
        obj({'role':{'const':'Designer'},'patch_id':{'const':patch['patch_id']},'decision':{'enum':['activate','defer'] if options else ['defer']},
             'variant_id':{'enum':list(options) or ['UNSUPPORTED']},
             'enabled_terms':{'type':'array','items':{'enum':['half','endpoint','type','orientation','role','pair']},'uniqueItems':True},
             'parent_factor_value':{'const':factor_value(parent,task['factor'])},
             'chosen_factor_value':{'enum':[factor_value(p,task['factor']) for p in options.values()] or ['UNSUPPORTED']},
             'source_score_is_generation_forecast':{'const':False},
             'scientific_rationale':short,'expected_execution_change':short,'falsifier':short}),
        folder/'Designer.json',input_store=repo/master['campaign_study']/'inputs',max_output_tokens=2500,max_format_repairs=2,response_validator=validate_d)
    delivery={r:api_audit(folder,r,hashlib.sha256(skills[r].encode()).hexdigest()) for r in skills}
    if not all(v['receipt_and_file_hashes_match'] and v['input_file_hashes_match'] and v['skill_delivered_exactly'] for v in delivery.values()):raise ValueError('Role delivery failed')
    write_json(folder/'delivery.json',{'roles':delivery,'allowed_patches':len(allowed),'designer_variants':len(options),'tool':query['tool']})
    write_json(folder/'selection.json',patch)
    if designer['decision']=='defer':return None,teacher,patch,designer
    p=options[designer['variant_id']];p['program_id']=cfg['round_id']
    p['analyst_sha256']=digest(folder/'Analyst.json');p['designer_sha256']=digest(folder/'Designer.json')
    return p,teacher,patch,designer


def commit(repo,master,message):
    paths=[master['campaign_study'],master['config_dir'],'src/evomolsteer/nc_guidance','src/evomolsteer/final_guidance/native.py',
           'src/evomolsteer/partial_search/campaign.py','scripts/run_nc_autoresearch.py','skills/nc-research-analyst','skills/nc-research-designer','tests/test_nc_autoresearch.py','.gitattributes']
    local(repo,['git','add','--',*paths])
    if local(repo,['git','diff','--cached','--name-only']):local(repo,['git','commit','-m',message])
    for i in range(3):
        try:local(repo,['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main']);break
        except RuntimeError:
            if i==2:raise
    return local(repo,['git','rev-parse','HEAD'])


def bind(repo,cfg,p,teacher,folder=None):
    target=(folder or repo/cfg['study'])/'candidates'/teacher['candidate_id'];target.mkdir(parents=True,exist_ok=True)
    tpath=target/'teacher.json';write_json(tpath,teacher);p=copy.deepcopy(p);p['teacher_sha256']=digest(tpath)
    path=target/'programs'/(p['program_id']+'.json');write_json(path,p)
    write_json(target/'bindings.json',{'teacher':tpath.relative_to(repo).as_posix(),
        'programs':[{'program':path.relative_to(repo).as_posix(),'sha256':digest(path),
                     'operative_text':'Analyst-bound proper local subset; actual FLOWR VJP; no SMC or affinity gradient'}]})
    return p


def realization(folder,p,teacher):
    audit=read_json(folder/'guidance_audit.json');geometry=read_json(folder/'geometry.json')
    boundary=next((r for r in audit['steps'] if 'boundary_typed_generated_slots' in r),None)
    if boundary is None:return {'status':'missing_actual_boundary'}
    ids=np.asarray(boundary['boundary_typed_generated_slots']);actual=np.asarray(boundary['boundary_current_atomic_indices'])
    target=np.asarray([teacher['atom_vocabulary'][e] for e in p['typed_elements']])
    integration=np.take_along_axis(np.asarray(geometry['integration_current']['atomic_indices']),ids,1)
    final=np.take_along_axis(np.asarray(geometry['terminal']['atomic_indices']),ids,1)
    at=(actual==target).all(1);integrated=(integration==target).all(1);last=(final==target).all(1)
    return {'joint_formation':float(at.mean()),'joint_at_integration':float(integrated.mean()),'joint_final_same_slots':float(last.mean()),
        'retention_after_formation':float(last[at].mean()) if at.any() else None,
        'final_prediction_changed_typed_fraction':float((integration!=final).mean()),
        'boundary_N_probability':boundary.get('target_N_probability'),'boundary_C_probability':boundary.get('target_C_probability'),
        'mean_assignment_switch':float(np.mean([r['assignment_switch_fraction'] for r in audit['steps'] if r.get('assignment_switch_fraction') is not None])),
        'mean_control_A':float(np.mean([r['injected_rms_A_mean'] for r in audit['steps'] if r['active']])),
        'actual_active_steps':len(audit['actual_active_steps'])}


def controls(repo,master,cfg,p,teacher,remote,state,source):
    native=p['native_options'];n=cfg['pilot_samples'];batch=cfg['pilot_initial_batch']
    if native==NATIVE_DEFAULTS and n in (24,50) and batch==0:
        return {'unguided':repo/'data/generated/partial_region_search_20261014'/('controls_screen/unguided' if n==24 else 'controls_batch0_n50/unguided'),
                'R13':repo/'data/generated/partial_region_search_20261014'/('R13/R13_0.1' if n==24 else 'R19/R19_0.1')}
    key=hashlib.sha256(json.dumps({'native':native,'n':n,'batch':batch,'source':source},sort_keys=True).encode()).hexdigest()[:12]
    label='controls_'+key;out=repo/master['generated_root']/label
    if not (out/'COMPLETE.json').exists():
        cc=copy.deepcopy(cfg);cc.update(study=master['campaign_study']+'/'+label,round_id=label,
            remote_output=master['remote_campaign']+'/'+label,include_unguided=True,only_program_ids=['R13'])
        baseline=copy.deepcopy(state['initial_program']);baseline.update(program_id='R13',native_options=native)
        baseline['expected_active_steps']=active_updates(baseline['steps'],*baseline['window'])
        native_teacher=read_json(repo/master['prior_study']/'templates.json')[baseline['template_id']]
        bind(repo,cc,baseline,native_teacher);path=repo/master['config_dir']/(label+'.json');write_json(path,cc)
        sha=commit(repo,master,'control(nc-research): freeze matched native protocol '+label)
        out=infer(repo,master,cc,path,remote,sha)
    return {'unguided':out/'unguided','R13':out/'R13'}


def summarise(repo,master,state):
    study=repo/master['campaign_study'];rows=[]
    for r in state['rounds']:
        m=r.get('metrics',{})
        rows.append({'round':r['round'],'factor':r['factor'],'status':r['status'],'patch':r.get('patch'),
            'value':json.dumps(r.get('chosen_value')),'mean':m.get('mean_all_pic50'),'max_qualified':m.get('best_qualified_pic50'),
            'qualified':m.get('n_qualified'),'n':m.get('n'),'mmff':m.get('median_mmff_relief_per_heavy'),
            'mean_vs_unguided':r.get('mean_vs_unguided'),'rollback':r.get('rollback',False),
            'formation':r.get('realization',{}).get('joint_formation'),'retention':r.get('realization',{}).get('retention_after_formation'),
            'commit':r.get('inference_commit')})
    write_table(study/'summary/rounds.csv',rows)
    write_json(study/'summary/state.json',state)
    body='|轮次|改动|状态|均值|最高合格|合格/尝试|相对无引导均值|回滚|\n|---|---|---|---:|---:|---|---:|---|\n'
    for r in rows:
        body+='|'+ '|'.join(str(r[k]) for k in ('round','factor','status','mean','max_qualified'))+f"|{r['qualified']}/{r['n']}|{r['mean_vs_unguided']}|{r['rollback']}|\n"
    (study/'summary/report.zh-CN.md').write_text('**新一轮局部 guidance 顺序研究**\n\n'+body+f"\n活动最佳：{state['best_round']}；目标8.704806。全部为预测分数。确认结果与预算/谱系不同的历史Steer分开解释。\n",encoding='utf-8')


def run(repo,master_path,stop_after=20):
    repo=Path(repo).resolve();master=read_json(master_path);study=repo/master['campaign_study'];sp=study/'state.json'
    if (study/'COMPLETE.json').exists():return read_json(sp)
    evidence,templates=prepare(repo,master)
    from .window_features import build
    build(repo,master)
    base=read_json(repo/master['initial_program']);base.update(native_options=copy.deepcopy(NATIVE_DEFAULTS),geometry_mode='aggregate')
    base.pop('type_site_weights',None)
    if sp.exists():state=read_json(sp)
    else:
        old=condition_metrics(repo/master['initial_results'])
        state={'next_round':1,'rounds':[],'initial_program':base,'best_program':base,'best':old,'best_round':'historical_R13',
               'declines':0,'cache':{},'best_typed':None}
    if 'tail_program' not in state:
        tail=max([r for r in state['rounds'] if r.get('metrics')],key=lambda r:r['metrics']['best_qualified_pic50'],default=None)
        state.update(tail_program=copy.deepcopy(tail['program'] if tail else base),tail_round=tail['round'] if tail else 'historical_R13',
                     tail_score=tail['metrics']['best_qualified_pic50'] if tail else state['best']['best_qualified_pic50'])
    remote=Remote()
    try:
        # Remote files are read once before cache decisions; no inference here.
        metadata=remote.command("python - <<'PY'\nimport hashlib,json\nfrom pathlib import Path\nroot=Path('/root/private_data/MolSteer/flowr_root')\npaths=[root/'flowr/models/fm_pocket.py',root/'flowr/models/integrator.py',root/'checkpoints/flowr_root_v2.ckpt']\npaths += [root/('experiments/ck2_clk3_lineage_20261003/results/main1000_w050/single/batch_%03d/initial_state.pt.gz'%b) for b in (0,2)]\nprint(json.dumps({str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}))\nPY")
        source={'remote':json.loads(metadata),'local':{str(p.relative_to(repo)):digest(p) for p in [repo/'src/evomolsteer/nc_guidance/reward.py',repo/'src/evomolsteer/final_guidance/native.py']}}
        write_json(study/'source_identity.json',source)
        for number in range(state['next_round'],min(stop_after,master['max_rounds'])+1):
            task=copy.deepcopy(master['tasks'][number-1]);label=f'R{number:02d}'
            parent=copy.deepcopy(state['best_program'])
            if number>=19:parent=copy.deepcopy(state['tail_program'])
            if task.get('parent')=='best_typed' and state.get('best_typed'):parent=copy.deepcopy(state['best_typed']['program'])
            if task.get('parent') in ('formed_counterfactual','saturated_native'):
                counter=task['parent']=='formed_counterfactual'
                eligible=[r for r in state['rounds'] if r.get('program',{}).get('weights',{}).get('type',0)>0
                    and bool(r['program'].get('counterfactual_labels'))==counter and r.get('realization',{}).get('joint_formation',0)>0]
                if not eligible:raise ValueError('Diagnostic parent needs observed formation, not an unrealized high score')
                chosen=max(eligible,key=lambda r:(r['realization']['joint_formation'],r['metrics']['best_qualified_pic50']))
                parent=copy.deepcopy(chosen['program'])
            if task.get('patches')=='parent':task['patches']=[parent['patch_id']]
            if task.get('patches')=='quality_matched':
                task['patches']=[p['patch_id'] for p in evidence['patches'] if p['patch_id'].startswith('Q') and len(p['atom_slots'])==len(parent['atom_slots'])]
            if task.get('patches')=='nested_parent':
                task['patches']=[p['patch_id'] for p in evidence['patches'] if p['template_id']==parent['template_id'] and len(p['atom_slots']) in (8,12,16,20)]
            if task.get('values')=='dose_neighbors':task['values']=[max(.05,parent['dose']['native_rms_ratio']*.5),min(2.,parent['dose']['native_rms_ratio']*1.5)]
            if task['factor']=='patch':task['values']=[None]
            n=master['screen_samples'] if number<19 else master['confirmation_samples']
            cfg={'study':master['campaign_study']+'/'+label,'campaign_study':master['campaign_study'],
                'round_id':label,'remote_output':master['remote_campaign']+'/'+label,
                'checkpoint':master['checkpoint'],'remote_dataset':master['remote_dataset'],'campaign':master['campaign'],'arm':master['arm'],
                'pilot_initial_batch':master['validation_batch'] if number==20 else 0,'pilot_samples':n,'seed':master['seed'],
                'steps':master['steps'],'teacher_time':master['teacher_time'],'terminal_time':1.,'include_unguided':False,'only_program_ids':[label]}
            revision=master.get('interface_revisions',{}).get(label)
            if revision:cfg['study']+='_interface_v'+str(revision)
            folder=repo/cfg['study'];path=repo/master['config_dir']/(label+'.json')
            write_json(folder/'proposal.json',{'task':task,'parent':parent,'history':compact_history(state),'max_rounds':master['max_rounds']})
            emit('research_role_calls',round=label,factor=task['factor'],active_incumbent=state['best_round'],actual_parent_program=parent['program_id'])
            p,teacher,patch,d=ask(repo,master,cfg,task,parent,evidence,templates,state)
            if p is None:
                entry={'round':label,'factor':task['factor'],'status':'designer_deferred','patch':patch['patch_id']}
            else:
                cfg['native_options']=p['native_options'];write_json(path,cfg);p=bind(repo,cfg,p,teacher)
                resolved={'program':p,'runtime':cfg,'source':source,'learning_window':master['learning_window'],
                    'parameter_consumers':{'dose':'Control.after','weights':'NCControl.forward','geometry_mode':'NCReward.terms',
                        'assignment_type_cost':'NCReward.assignment','native_options':'native.run','window':'NCReward.active'},
                    'single_changed_factor':task['factor'],'patch':patch['patch_id']}
                write_json(folder/'resolved_config.json',resolved)
                key=execution_identity(p,teacher,cfg,source);write_json(folder/'signature.json',{'numeric_execution_signature':key})
                sha=commit(repo,master,f'experiment(nc-research): {label} {task["factor"]} Analyst-bound numeric proposal')
                output=repo/master['generated_root']/label
                reused=key in state['cache']
                if reused:result=repo/state['cache'][key]
                else:
                    if not (output/'COMPLETE.json').exists():
                        write_json(folder/'pending_inference.json',{'inference_commit':sha})
                        output=infer(repo,master,cfg,path,remote,sha)
                    else:sha=read_json(folder/'pending_inference.json')['inference_commit']
                    result=output/label;state['cache'][key]=result.relative_to(repo).as_posix()
                scores=condition_metrics(result);obs=realization(result,p,teacher)
                control_paths=controls(repo,master,cfg,p,teacher,remote,state,source)
                comparisons={k:paired_comparison(result,f,label+' vs '+k) for k,f in control_paths.items()}
                control_metrics={k:condition_metrics(f) for k,f in control_paths.items()}
                score=scores['best_qualified_pic50'] or -1e20;delta=score-state['best']['best_qualified_pic50']
                rollback=False;confirmation=number>=19
                if not confirmation:
                    if score>state['tail_score']:
                        state.update(tail_score=score,tail_program=copy.deepcopy(p),tail_round=label)
                    # Protect affinity distribution while pursuing a high-quality tail.
                    viable=scores['mean_all_pic50']>=state['best']['mean_all_pic50']-master['mean_regression_tolerance']
                    if delta>master['decline_tolerance'] and viable:
                        state.update(best_program=p,best=scores,best_round=label,declines=0)
                    else:state['declines']=state['declines']+1 if delta < -master['decline_tolerance'] else 0
                    if p['weights']['type']>0 and (not state['best_typed'] or score>state['best_typed']['score']):state['best_typed']={'program':p,'score':score}
                    if state['declines']>=master['rollback_declines']:
                        rollback=True;state['declines']=0
                        write_json(folder/'rollback.json',{'restored_program':state['best_program'],'restored_round':state['best_round'],
                            'next_default_parent':'best_program','retained_correctness_fixes':True})
                entry={'round':label,'factor':task['factor'],'status':'reused' if reused else 'completed','patch':patch['patch_id'],
                    'chosen_value':task['values'][int(d['variant_id'][1:])],'program':p,'inference_commit':sha,
                    'metrics':scores,'realization':obs,'controls':control_metrics,'paired':comparisons,
                    'mean_vs_unguided':comparisons['unguided']['mean_pic50_difference'],'rollback':rollback,
                    'confirmation':confirmation,'result_folder':result.relative_to(repo).as_posix(),
                    'incumbent_after':state['best_round'],'additional_attempts':0 if reused else n}
            write_json(folder/'results.json',entry);state['rounds'].append(entry);state['next_round']=number+1
            entry['study']=cfg['study']
            write_json(sp,state);write_json(repo/master['config_dir']/'active_best.json',state['best_program']);summarise(repo,master,state)
            commit(repo,master,f'results(nc-research): {label} {entry["status"]}; incumbent {state["best_round"]}')
            emit('research_round_complete',round=label,status=entry['status'],metrics=entry.get('metrics'),realization=entry.get('realization'),rollback=entry.get('rollback'))
            if (study/'STOP_AFTER_CURRENT').exists():return state
        if state['next_round']>master['max_rounds']:
            write_json(study/'COMPLETE.json',{'rounds':len(state['rounds']),'best_round':state['best_round'],
                'best':state['best'],'target_reached':any(r.get('metrics',{}).get('n_above_reference',0)>0 for r in state['rounds'])})
            commit(repo,master,'results(nc-research): complete twenty sequential rounds')
        return state
    finally:remote.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--stop-after',type=int,default=20)
    a=p.parse_args();run(a.repo,a.config,a.stop_after)


if __name__=='__main__':main()
