"""Real-API tool loop; Analyst fixes support/mode, Designer chooses bounded function."""
import copy
import json
import hashlib
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj
from .bank import regional_evidence
from .programs import validate,numerical_identity


MODULES={'interface_support':['C1'],'NC_support':['C1','C2'],'group_support':['C1','C2','C3'],
    'orientation':['C1','C3'],'pocket_field':['C1'],'pair_geometry':['C3'],'charge_field':['C1','C3','C4'],
    'typed_channels':['C1','C2'],'type_reference':['C1','C2']}

CLOCKS={'fixed':{'observable':'predictive_bridge_to_learning_end','teacher':'actual_ancestor_at_learning_end'},
    'current':{'observable':'current_state','teacher':'actual_ancestor_at_same_current_time'},
    'endpoint_prediction':{'observable':'predicted_endpoint','teacher':'teacher_predicted_endpoint_at_same_forward_time'},
    'lookahead':{'observable':'predictive_bridge_to_lookahead','teacher':'actual_ancestor_at_same_lookahead_time'},
    'decaying_lookahead':{'observable':'predictive_bridge_to_decaying_lookahead','teacher':'actual_ancestor_at_same_lookahead_time'}}


def factual_review(repo,m,folder,role,answer,contract):
    """A bounded source-consistency review, never a numerical reward redesign."""
    schema=obj({'violations':{'type':'array','maxItems':5,'items':obj({'quote':{'type':'string','maxLength':1500},
        'code':{'enum':['clock_confusion','unsupported_active_force','source_or_comparator_confusion','unmeasured_result_claim']},
        'explanation':{'type':'string','maxLength':1600}})}})
    def validate(v):
        raw=json.dumps(answer,ensure_ascii=False)
        for item in v['violations']:
            if item['quote'] not in raw and not any(item['quote'] in str(x) for x in answer.values()):
                import jsonschema
                raise jsonschema.ValidationError('Reviewer quote is not verbatim role text',validator='verbatim_quote',validator_value='actual role response',instance=item['quote'])
    return call_json([{'role':'system','content':'Check factual consistency against the supplied executable/source contract. Flag only explicit contradictions. A prospective hypothesis is not a claimed measurement. Do not redesign the reward or prefer a candidate. Copy exact short quotes for violations.'},
        {'role':'user','content':json.dumps({'role':role,'response':answer,'authoritative_contract':contract},ensure_ascii=False)}],
        schema,folder/(role+'.factual_review.json'),input_store=repo/m['campaign_study']/'inputs',response_validator=validate,max_output_tokens=3000)


def instructions(repo,m,role,modules):
    text=(repo/m[role.lower()+'_skill']).read_text(encoding='utf-8')
    if role=='Analyst' and modules:
        ref=(repo/m['analyst_skill']).parent/'references/chemistry-modules.zh-CN.md';source=ref.read_text(encoding='utf-8')
        sections=source.split('\n## ')
        text+='\n\nEnabled optional scientific modules only:\n'+'\n\n'.join('## '+s for s in sections[1:] if s.split('.',1)[0] in modules)
    return text


def ask(repo,m,folder,bank,regions,state,actions,phase,modules_override=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    modules=MODULES.get(phase,[]) if modules_override is None else modules_override
    skills={r:instructions(repo,m,r,modules) for r in ('Analyst','Designer')}
    for r,s in skills.items():(folder/(r+'.SKILL.md')).write_bytes(s.encode('utf-8'))
    groups={};options={a['action_id']:a for a in actions}
    for a in actions:
        key=json.dumps([a['parent_id'],a['region_ids'],a['phase']])
        groups.setdefault(key,[]).append(a)
    groups={f'G{i:03d}':v for i,v in enumerate(groups.values())}
    catalog=[]
    for gid,values in groups.items():
        a=values[0];p=a['program']
        catalog.append({'group_id':gid,'parent':a['parent_id'],'region_ids':a['region_ids'],'reason':a['reason'],
            'mode_ids':[s['template_id'] for s in p['mode_specs']],'clock_policy':p['clock_policy'],
            'operator':p['operator'],'observable_contract':CLOCKS[p['clock_policy']],
            'support_sizes':[len(s['atom_slots']) for s in p['mode_specs']],
            'numerical_options':[{'option_id':v['action_id'],'program':v['program']} for v in values]})
    allowed_regions=sorted({rid for a in actions for rid in a['region_ids']});byregion={r['region_id']:r for r in regions}
    history=[{k:r.get(k) for k in ('round','phase','parent','metrics','response','rollback','status')} for r in state.get('rounds',[])]
    bundle={'evidence':{'campaign/history':history,'source/prior_campaign':state.get('prior_campaign_evidence'),
        'parent/outcomes':state['best'].get('metrics'),
        'parent/response':state['best'].get('response'),'source/limits':bank['limits']},
        'phase':phase,'enabled_chemistry_modules':modules,'catalog':catalog,'learning_window':m['learning_window'],
        'contract':{'no_affinity_derivative':True,'no_smc':True,'scope':'coordinate only; graph free',
            'teacher_roles':'Terminal source roles are not early-state actual identities.',
            'density_ratio':'Descriptive explicit local feature proxy, not future-success probability.',
            'dose':'Post-RMS actual displacement; common reward scaling can cancel.',
            'router':'One active local mode per seed, state compatibility is detached, no full-cloud union.',
            'budget_normalization':'Calibrated to frozen reference drift, actual budgets can differ and are measured.'}}
    bundle['contract']['clock_definitions']=CLOCKS
    bundle['contract']['registered_active_forces']='pointwise/pair/band/pocket/orientation/selection operator as program says; typed/charge only when positive weights. Detached element assignment is not direct type force. A polar-selected support with pointwise reward is still geometry-only.'
    tools=['ancestor','selection_trend','controller_response','failure_history']
    if modules:tools.append('chemical_context')
    def query(tool,rid):
        r=byregion[rid];mode=bank['modes'][r['template_id']]
        if tool=='ancestor':
            path=mode['path'];indices=sorted(set([0,len(path['times'])//4,len(path['times'])//2,len(path['times'])-1]))
            return {'candidate':mode['candidate_id'],'root_id':mode['root_id'],'source':mode['trajectory_sha256'],
                'domain':m['learning_window'],'current_clock_times':[path['times'][i] for i in indices],
                'selected_support':r['atom_slots'],'source_terminal_rescore':mode['terminal_score'],
                'selected_terminal_elements':[mode['native_elements'][j] for j in r['atom_slots']],
                'actual_ancestor_slots':[path['ancestor_slots'][i] for i in indices],
                'selected_ancestor_coords_A':[np.asarray(path['coords_A'][i])[r['atom_slots']].tolist() for i in indices],
                'selected_teacher_endpoint_A':[np.asarray(path['endpoint_A'][i])[r['atom_slots']].tolist() for i in indices],
                'representations':['actual current world A','same-forward teacher predicted endpoint world A'],
                'limits':'Current versus endpoint targets must not be exchanged; early identity remains uncertain.'}
        if tool=='chemical_context':
            from .chemistry import terminal_group_context
            xyz=np.asarray(mode['terminal_coords_A'])[r['atom_slots']];sites=np.asarray([s['coords_A'] for s in r['pocket_sites']])
            prior=read_json(repo/m['prior_study']/'evidence.json');card=next((v for k,v in prior['cards'].items() if k.startswith(mode['candidate_id']+'__')),None)
            groups=terminal_group_context(m['layout_report'],mode['evidence_origin'],json.dumps(mode['terminal_roles'],sort_keys=True))
            groups=copy.deepcopy(groups);groups['groups']=[g for g in groups['groups'] if set(g['atoms']).intersection(r['atom_slots'])]
            return {'region':r,'terminal_roles':[x for x in mode['terminal_roles'] if x['atom_index'] in r['atom_slots']],
                'terminal_coords_A':xyz.tolist(),'site_distance_matrix_A':np.linalg.norm(xyz[:,None]-sites[None],axis=-1).tolist(),
                'conditional_layout_evidence':None if card is None else card['conditional_layout'],
                **({'functional_groups':groups} if any(k in modules for k in ('C3','C4','C5')) else {}),
                'pocket_sites':r['pocket_sites'],'full_adjacency':{str(i):mode['full_adjacency'][str(i)] for i in r['atom_slots']},
                'limits':'Chemical labels are terminal observations; nearest polar pocket atoms do not establish H bonds or energetic contributions.'}
        if tool=='selection_trend':
            e=regional_evidence(repo,m,bank,r)
            return {k:v for k,v in e.items() if k!='rows'}|{'n_times':len(e['rows']),
                'representative_rows':e['rows'][::max(1,len(e['rows'])//5)],'full_curve_reference':str((repo/m['campaign_study']/'regional_evidence'/(rid+'.json')).relative_to(repo))}
        if tool=='controller_response':return {'parent_response':state['best'].get('response'),'actual_program':state['best']['program']}
        if tool=='failure_history':return {'current':history,'previous_campaign':state.get('prior_campaign_evidence')}
        raise ValueError('Unknown tool')
    messages=[{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'bundle':bundle,'available_tools':tools,'allowed_region_ids':allowed_regions,
        'task':'Choose discriminating evidence before the scientific handoff. Query real tools for candidate regions; then select a support/mode hypothesis group. You cannot invent evidence. One principal intervention, not a guarantee of improvement. Tool loop only now.'},ensure_ascii=False)}]
    query_schema=obj({'requests':{'type':'array','minItems':1,'maxItems':2,'items':obj({'tool':{'enum':tools},'region_id':{'enum':allowed_regions}})},
        'purpose':{'type':'string','maxLength':2000}})
    receipts=[]
    for turn in range(2):
        qs=copy.deepcopy(query_schema)
        if turn==1 and modules:
            # One additional explicit local chemistry query makes final region evidence bindable.
            qs['properties']['requests']['items']['properties']['tool']={'const':'chemical_context'}
        ans=call_json(messages,qs,folder/f'Analyst.query{turn}.json',input_store=repo/m['campaign_study']/'inputs',max_output_tokens=2500)
        got={}
        for request in ans['requests']:
            key=request['region_id']+'/'+request['tool'];got[key]=query(request['tool'],request['region_id']);bundle['evidence'][key]=got[key]
        messages.extend([{'role':'assistant','content':json.dumps(ans,ensure_ascii=False)},
            {'role':'user','content':json.dumps({'observations':got,'task':'One more evidence query or final handoff when requested; do not infer numerical results not returned by tools.'},ensure_ascii=False)}])
        receipts.append(digest(folder/f'Analyst.query{turn}.request.json'))
    refs={'type':'array','minItems':1,'items':{'enum':sorted(bundle['evidence'])}}
    text={'type':'string','minLength':1,'maxLength':4000}
    eligible={gid:group for gid,group in groups.items() if any(rid+'/chemical_context' in bundle['evidence'] for rid in group[0]['region_ids'])} if modules else groups
    if not eligible:raise ValueError('No group has its own chemical evidence; query phase must be repaired')
    schema=obj({'group_id':{'enum':list(eligible)},'evidence_refs':refs,'counterevidence_refs':refs,
        'scientific_hypothesis':text,'observable_change':text,'formation_retention_value_test':text,'falsifier':text,
        'active_chemistry_modules':{'const':modules},'chemical_assumptions':text,'clock_and_representation':text,
        'relationship_to_failed_families':text,'support_reason':text,
        'observable_contract':obj({'observable':{'enum':sorted({c['observable'] for c in CLOCKS.values()})},
            'teacher':{'enum':sorted({c['teacher'] for c in CLOCKS.values()})}})})
    def validate_a(a):
        group=groups[a['group_id']]
        if not any(ref.split('/')[0] in group[0]['region_ids'] for ref in a['evidence_refs']):
            import jsonschema
            raise jsonschema.ValidationError('Chosen group must cite a queried local source',validator='support_binding',validator_value=group[0]['region_ids'],instance=a['evidence_refs'])
        if modules and not any(rid+'/chemical_context' in a['evidence_refs'] for rid in group[0]['region_ids']):
            import jsonschema
            raise jsonschema.ValidationError('Selected support needs its OWN chemical context citation',validator='chemical_support_binding',validator_value=group[0]['region_ids'],instance=a['evidence_refs'])
        expected=CLOCKS[group[0]['program']['clock_policy']]
        if a['observable_contract']!=expected:
            import jsonschema
            raise jsonschema.ValidationError('Selected observable and teacher representation disagree',validator='clock_binding',validator_value=expected,instance=a['observable_contract'])
    messages.append({'role':'user','content':json.dumps({'evidence_mapping':bundle['evidence'],'response_task':'Return the scientific Analyst handoff using a group whose actual regional evidence you queried. Designer will choose the numerical option inside that group only.'},ensure_ascii=False)})
    a=call_json(messages,schema,folder/'Analyst.json',input_store=repo/m['campaign_study']/'inputs',response_validator=validate_a,max_output_tokens=5500)
    review=factual_review(repo,m,folder,'Analyst',a,{'execution':groups[a['group_id']][0]['program'],
        'clock':CLOCKS[groups[a['group_id']][0]['program']['clock_policy']], 'source_contract':bundle['contract'],
        'evidence':bundle['evidence']})
    if review['violations']:
        a=call_json(messages+[{'role':'user','content':json.dumps({'neutral_factual_diagnostics':review,
            'task':'Correct factual contradictions while preserving supported scientific choices; return the same schema.'},ensure_ascii=False)}],schema,
            folder/'Analyst.corrected.json',input_store=repo/m['campaign_study']/'inputs',response_validator=validate_a,max_output_tokens=5500)
        # Keep original immutable; downstream identity binds the extracted corrected handoff explicitly.
        analyst_file=folder/'Analyst.corrected.json'
        final_review=factual_review(repo,m,folder,'Analyst.corrected',a,{'execution':groups[a['group_id']][0]['program'],
            'clock':CLOCKS[groups[a['group_id']][0]['program']['clock_policy']],'source_contract':bundle['contract']})
        if final_review['violations']:raise ValueError('Analyst remains factually inconsistent after bounded correction')
    else:analyst_file=folder/'Analyst.json'
    selected=groups[a['group_id']];write_json(folder/'evidence_bundle.json',bundle)
    designer_schema=obj({'option_id':{'enum':[x['action_id'] for x in selected]},'hypothesis_group':{'const':a['group_id']},
        'evidence_refs':refs,'design_grounds':text,'operative_terms':text,'derivative_path':text,'actual_dose_expectation':text,'falsifier':text})
    d=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
        'Analyst':a,'shared_evidence':bundle['evidence'],'allowed_numerical_options':selected,
        'task':'Choose one executable numerical variant implementing this Analyst support/mode hypothesis. Do not alter points, mode sources or add terms. Explain operative derivative and actual dose, with a testable negative result. Registry is bounded; choosing parameters is not unrestricted code synthesis.'},ensure_ascii=False)}],
        designer_schema,folder/'Designer.json',input_store=repo/m['campaign_study']/'inputs',max_output_tokens=5000)
    action=options[d['option_id']];p=copy.deepcopy(action['program']);validate(p,bank)
    d_review=factual_review(repo,m,folder,'Designer',d,{'execution':p,'clock':CLOCKS[p['clock_policy']],
        'source_contract':bundle['contract'],'Analyst':a})
    if d_review['violations']:
        d=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
            'Analyst':a,'allowed_options':selected,'authoritative_clock':CLOCKS[p['clock_policy']],
            'previous_response':d,'neutral_factual_diagnostics':d_review,'task':'Correct factual contradictions; keep this schema and Analyst-bound hypothesis.'},ensure_ascii=False)}],
            designer_schema,folder/'Designer.corrected.json',input_store=repo/m['campaign_study']/'inputs',max_output_tokens=5000)
        action=options[d['option_id']];p=copy.deepcopy(action['program']);validate(p,bank);designer_file=folder/'Designer.corrected.json'
        final_review=factual_review(repo,m,folder,'Designer.corrected',d,{'execution':p,'clock':CLOCKS[p['clock_policy']],
            'source_contract':bundle['contract'],'Analyst':a})
        if final_review['violations']:raise ValueError('Designer remains factually inconsistent after bounded correction')
    else:designer_file=folder/'Designer.json'
    if action not in selected:raise ValueError('Designer escaped Analyst hypothesis')
    p.update(analyst_sha256=digest(analyst_file),designer_sha256=digest(designer_file),evidence_bundle_sha256=digest(folder/'evidence_bundle.json'))
    delivery={'model':'deepseek-flash','real_api':True,'temperature':0,'active_modules':modules,
        'analyst_response_path':analyst_file.name,'designer_response_path':designer_file.name,
        'skill_sha256':{r:hashlib.sha256(v.encode()).hexdigest() for r,v in skills.items()},
        'query_request_sha256':receipts,'analyst_sha256':p['analyst_sha256'],'designer_sha256':p['designer_sha256'],
        'bound_region_ids':action['region_ids'],'numeric_identity':numerical_identity(p,bank),
        'analysis_scope':'Evidence, decisions and falsifiers; no private reasoning transcript.'}
    write_json(folder/'delivery.json',delivery);write_json(folder/'decision.json',{'Analyst':a,'Designer':d,'action':{k:v for k,v in action.items() if k!='program'}})
    write_json(folder/'selected_program.json',p)
    return p,action,delivery
