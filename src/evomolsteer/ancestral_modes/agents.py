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
from .programs import validate,numerical_identity,program_delta,scientific_group_key,executable_view,compile_designer_response,controlled_feature


MODULES={'interface_support':['C1'],'NC_support':['C1','C2'],'group_support':['C1','C2','C3'],
    'field_response_calibration':['C1','C2'],
    'orientation':['C1','C3'],'pocket_field':['C1'],'pair_geometry':['C3'],'charge_field':['C1','C3','C4'],
    'typed_channels':['C1','C2'],'type_reference':['C1','C2']}

CLOCKS={'fixed':{'observable':'predictive_bridge_to_learning_end','teacher':'actual_ancestor_at_learning_end'},
    'current':{'observable':'current_state','teacher':'actual_ancestor_at_same_current_time'},
    'endpoint_prediction':{'observable':'predicted_endpoint','teacher':'teacher_predicted_endpoint_at_same_forward_time'},
    'lookahead':{'observable':'predictive_bridge_to_lookahead','teacher':'actual_ancestor_at_same_lookahead_time'},
    'decaying_lookahead':{'observable':'predictive_bridge_to_decaying_lookahead','teacher':'actual_ancestor_at_same_lookahead_time'}}


def factual_review(repo,m,folder,role,answer,contract):
    """A bounded source-consistency review, never a numerical reward redesign."""
    contract=copy.deepcopy(contract)
    if 'execution' in contract:contract['execution']=executable_view(contract['execution'])
    if 'evidence' in contract:
        cited=set(answer.get('evidence_refs',[]))|set(answer.get('counterevidence_refs',[]))
        contract['evidence']={k:v for k,v in contract['evidence'].items() if k in cited and k not in ('campaign/history','source/prior_campaign') and not k.endswith('/failure_history')}
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
        key=scientific_group_key(a)
        groups.setdefault(key,[]).append(a)
    groups={f'G{i:03d}':v for i,v in enumerate(groups.values())}
    parent_ids={a['parent_id'] for a in actions};parents={}
    for entry in [state['reference'],state['best'],*state.get('rounds',[])]:
        if entry.get('program',{}).get('program_id') in parent_ids:
            parents[entry['program']['program_id']]={k:entry.get(k) for k in ('program','metrics','response','local_outcomes','result_folder')}
    for pid,record in parents.items():
        if not record.get('result_folder'):continue
        result=repo/record['result_folder'];audit=read_json(result/'guidance_audit.json')
        if audit.get('program',{}).get('family')!='ancestral_modes_v3':continue
        from .layout import analyze as layout_analysis
        from .verify import observe
        destination=repo/m['campaign_study']/pid/'delivered_measurements'
        record['element_layout']=layout_analysis(result,bank,destination/'element_layout')
        record['execution_response']=observe(result,destination/'execution_observation.json')
    for action in actions:action['authoritative_program_delta']=program_delta(parents[action['parent_id']]['program'],action['program'])
    catalog=[]
    for gid,values in groups.items():
        a=values[0];p=a['program']
        catalog.append({'group_id':gid,'parent':a['parent_id'],'region_ids':a['region_ids'],'reason':a['reason'],
            'parent_metrics':parents[a['parent_id']]['metrics'],
            'mode_ids':[s['template_id'] for s in p['mode_specs']],'clock_policy':p['clock_policy'],
            'operator':p['operator'],'observable_contract':CLOCKS[p['clock_policy']],
            'support_sizes':[len(s['atom_slots']) for s in p['mode_specs']],
            'numerical_options':[{'option_id':v['action_id'],'program':executable_view(v['program']),'authoritative_program_delta':v['authoritative_program_delta']} for v in values]})
    allowed_regions=sorted({rid for a in actions for rid in a['region_ids']});byregion={r['region_id']:r for r in regions}
    history=[]
    for r in state.get('rounds',[]):
        brief={k:r.get(k) for k in ('round','phase','parent','metrics','rollback','status')}
        response=r.get('response',{});statistics=response.get('statistics',{})
        brief['response_summary']={k:statistics[k] for k in ('dose_gain_mean','predictive_flow_rms_A_mean','injected_rms_A_mean','gradient_native_observed_cosine_mean','assignment_switch_fraction') if k in statistics}
        brief['execution_updates']={k:response.get(k) for k in ('active_updates','nonzero_updates')}
        brief['full_results_reference']=m['campaign_study']+'/'+r['round']+'/results.json'
        history.append(brief)
    bundle={'evidence':{'campaign/history':history,'source/prior_campaign':state.get('prior_campaign_evidence'),
        'source/limits':bank['limits']},
        'phase':phase,'enabled_chemistry_modules':modules,'catalog':catalog,'learning_window':m['learning_window'],
        'contract':{'no_affinity_derivative':True,'no_smc':True,'scope':'coordinate only; graph free',
            'teacher_roles':'Terminal source roles are not early-state actual identities.',
            'density_ratio':'Descriptive explicit local feature proxy, not future-success probability.',
            'dose':'Post-RMS actual displacement; common reward scaling can cancel.',
            'router':'One active local mode per seed, state compatibility is detached, no full-cloud union.',
            'budget_normalization':'Calibrated to frozen reference drift, actual budgets can differ and are measured.'}}
    bundle['objective']={'primary':'Discover a basic-qualified terminal candidate approaching or exceeding the original Steer best predicted affinity.',
        'original_steer_best':state['original']['best'],'tail_candidate_program':state.get('tail_candidate',state['reference'])['program']['program_id'],
        'tail_candidate_metrics':state.get('tail_candidate',state['reference'])['metrics'],
        'separate_criterion':'Guarded mean/U_tau incumbent promotion remains fixed. That guard does not ban experiments around a measured near-target tail branch. Do not replace the requested tail-discovery goal with maximizing mean alone.',
        'uncertainty':'Development maximum is search biased; final frozen new-pool confirmation and quality diagnostics are separate.'}
    if m.get('experiment_contract'):bundle['contract']['current_experiment']=m['experiment_contract']
    for pid,v in parents.items():
        bundle['evidence']['parents/'+pid+'/outcomes']={'parent_program_id':pid,'metrics':v['metrics'],'local_outcomes':v.get('local_outcomes')}
        bundle['evidence']['parents/'+pid+'/response']={'parent_program_id':pid,'response':v['response'],'program':executable_view(v['program'])}
        if 'element_layout' in v:
            bundle['evidence']['parents/'+pid+'/element_layout']={'parent_program_id':pid,**v['element_layout']}
            bundle['evidence']['parents/'+pid+'/component_response']={'parent_program_id':pid,**v['execution_response']}
    correction=repo/m['campaign_study']/'feature_definition_correction.json'
    if correction.exists():bundle['evidence']['source/feature_revision']=read_json(correction)
    routing_evidence=repo/m['campaign_study']/'paired_path_preferences_R02.json'
    if routing_evidence.exists():bundle['evidence']['source/measured_path_preferences']=read_json(routing_evidence)
    bundle['contract']['clock_definitions']=CLOCKS
    bundle['contract']['registered_active_forces']='pointwise/pair/band/pocket/orientation/selection operator as program says; typed/charge only when positive weights. Detached element assignment is not direct type force. A polar-selected support with pointwise reward is still geometry-only.'
    if m.get('local_support_registry'):
        bundle['contract']['local_support_selection']={'policy':m['local_support_policy'],
            'rule':'Atom count is an upper bound from molecule coverage, never a target. Query and cite the selected region local_support_evidence; compare local contrast beyond whole-cloud similarity. Teacher trends are not cohort enrichment. Explain why every included neighbor is needed; broad historical support is a baseline, not an automatically inherited hypothesis.'}
    bundle['contract']['response_routing']='Response routing uses sparse per-mode coordinate VJPs of this surrogate, from the same generator forward. Gap-normalized local controllability is NOT future affinity. Nearest geometry, surrogate controllability and final outcome are separate hypotheses. Two probe times are configurable fractions of the control window; no extra generator/affinity forwards.'
    bundle['contract']['measurement_stages']='Nonzero component VJP establishes activation only, not geometric/elemental formation. Formation uses actual boundary geometry/layout. Retention compares boundary and terminal layout of the same tracked support after the unguided suffix. Zero clipping and constant dose during control are NOT suffix-retention evidence. Nonzero raw component gradient does not establish a significant change in the normalized total direction: inspect component/total fraction and cosine. Final value uses terminal affinity/quality and matched controls.'
    bundle['contract']['feature_profile']='Online feature moments are linearly interpolated saved rows. Whole-window quadratic fits and derivatives are diagnostic, not the online control curve. Post-RMS historical contrast attenuation measures standardized descriptive separation, not calibrated reliability or future success.'
    bundle['contract']['surrogate_units']='Geometry distance losses and sigma-scaled type/charge fields have A^2 units. Orientation and standardized feature log-ratios are dimensionless; relative chemical coefficients then carry compensating A^-2 units. These are optimization surrogates, not physical energies. Kernel mass is dimensionless; signed displacement components are A.'
    tools=['ancestor','selection_trend','controller_response','failure_history']
    if m.get('local_support_registry'):tools.append('local_support_evidence')
    if any(len(a['region_ids'])>1 for a in actions):tools.append('mode_comparison')
    if modules:tools.append('chemical_context')
    def query(tool,rid):
        r=byregion[rid];mode=bank['modes'][r['template_id']]
        if tool=='local_support_evidence':
            if 'local_support_evidence' not in r:
                return {'region_id':rid,'status':'historical_reference_only','evidence':None,
                    'limits':'No newly mined local support evidence; do not infer it from the teacher score.'}
            result=read_json(repo/m['local_support_registry']/r['local_support_evidence'])
            if result['region_id']!=rid:raise ValueError('Measured evidence belongs to a different local support')
            return {'region':r,'measured_evidence':result,
                'active_operator_contract':'Only the selected registered operator is active. Role labels select regions; geometric operators do not supply chemical-role forces.'}
        if tool=='mode_comparison':
            ids=sorted({v for a in actions if rid in a['region_ids'] for v in a['region_ids']})
            rows=[]
            for other in ids:
                ancestry=query('ancestor',other)
                rows.append(ancestry|{'region_id':other,'terminal_local_coords_A':np.asarray(bank['modes'][byregion[other]['template_id']]['terminal_coords_A'])[byregion[other]['atom_slots']].tolist()})
            return {'actual_mode_sources':rows,'limit':'Different successful local solutions are kept separate. Terminal scores are retrospective, not online route scores.'}
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
            result={k:v for k,v in e.items() if k!='rows'}|{'n_times':len(e['rows']),
                'representative_rows':e['rows'][::max(1,len(e['rows'])//5)],'full_curve_reference':e['cache_reference']}
            if any(a['program']['operator']=='selection_advantage' for a in actions):
                from .forecast_evidence import evidence as forecast_evidence
                f=forecast_evidence(repo,m,bank,r)
                result['endpoint_forecast_evidence']={k:v for k,v in f.items() if k!='rows'}|{'n_times':len(f['rows']),
                    'representative_rows':f['rows'][::max(1,len(f['rows'])//5)]}
                result['representation_contract']='Use actual-current profiles for current/bridge observables and forecast profiles for endpoint_prediction; prediction selection shifts do not describe proposal geometry.'
            return result
        if tool=='controller_response':
            ids={a['parent_id'] for a in actions if rid in a['region_ids']}
            return {'candidate_parent_records':{pid:{**parents[pid],'program':executable_view(parents[pid]['program'])} for pid in ids},'rule':'Bind the SELECTED action parent, never substitute incumbent measurements.'}
        if tool=='failure_history':return {'current':history,'previous_campaign':state.get('prior_campaign_evidence')}
        raise ValueError('Unknown tool')
    messages=[{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'bundle':bundle,'available_tools':tools,'allowed_region_ids':allowed_regions,
        'task':'Choose discriminating evidence before the scientific handoff. Query real tools for candidate regions; then select a support/mode hypothesis group. You cannot invent evidence. One principal intervention, not a guarantee of improvement. Tool loop only now.'},ensure_ascii=False)}]
    query_schema=obj({'requests':{'type':'array','minItems':1,'maxItems':2,'items':obj({'tool':{'enum':tools},'region_id':{'enum':allowed_regions}})},
        'purpose':{'type':'string','maxLength':2000}})
    receipts=[]
    query_turns=3 if m.get('local_support_registry') else 2
    for turn in range(query_turns):
        qs=copy.deepcopy(query_schema)
        if turn==1 and modules:
            # One additional explicit local chemistry query makes final region evidence bindable.
            qs['properties']['requests']['items']['properties']['tool']={'const':'chemical_context'}
        if m.get('local_support_registry') and turn==query_turns-1:
            # Guaranteed measured support delivery without displacing the chemistry query.
            qs['properties']['requests']['items']['properties']['tool']={'const':'local_support_evidence'}
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
    if m.get('local_support_registry'):
        eligible={gid:group for gid,group in eligible.items() if all(
            'local_support_evidence' not in byregion[rid] or rid+'/local_support_evidence' in bundle['evidence']
            for rid in group[0]['region_ids'])}
    if not eligible:raise ValueError('No group has its own chemical evidence; query phase must be repaired')
    schema=obj({'group_id':{'enum':list(eligible)},'parent_program_id':{'enum':sorted(parent_ids)},'evidence_refs':refs,'counterevidence_refs':refs,
        'primary_operator':{'enum':sorted({g[0]['program']['operator'] for g in eligible.values()})},
        'scientific_hypothesis':text,'observable_change':text,'formation_retention_value_test':text,'falsifier':text,
        'active_chemistry_modules':{'const':modules},'chemical_assumptions':text,'clock_and_representation':text,
        'relationship_to_failed_families':text,'support_reason':text,
        'component_response_interpretation':text,
        'observable_contract':obj({'observable':{'enum':sorted({c['observable'] for c in CLOCKS.values()})},
            'teacher':{'enum':sorted({c['teacher'] for c in CLOCKS.values()})}})})
    if m.get('strict_feature_handoff'):
        schema['properties']['controlled_feature']={'enum':sorted({controlled_feature(g[0]['program']) for g in eligible.values()}),
            'description':'The actual active geometric reward observable. Other measured features are diagnostic/background; do not describe them as separately activated forces.'}
        schema['required'].append('controlled_feature')
    def validate_a(a):
        group=groups[a['group_id']]
        if m.get('strict_feature_handoff') and a['controlled_feature']!=controlled_feature(group[0]['program']):
            import jsonschema
            raise jsonschema.ValidationError('Analyst observable is not implemented by the selected operator',validator='active_feature_binding',
                validator_value=controlled_feature(group[0]['program']),instance=a['controlled_feature'])
        if a['primary_operator']!=group[0]['program']['operator']:
            import jsonschema
            raise jsonschema.ValidationError('Hypothesis primary operator must be the selected registered mechanism',validator='operator_binding',validator_value=group[0]['program']['operator'],instance=a['primary_operator'])
        if a['parent_program_id']!=group[0]['parent_id'] or 'parents/'+a['parent_program_id']+'/outcomes' not in a['evidence_refs']:
            import jsonschema
            raise jsonschema.ValidationError('Chosen group must bind and cite its ACTUAL parent outcomes',validator='parent_binding',validator_value=group[0]['parent_id'],instance=a['parent_program_id'])
        if not any(ref.split('/')[0] in group[0]['region_ids'] for ref in a['evidence_refs']):
            import jsonschema
            raise jsonschema.ValidationError('Chosen group must cite a queried local source',validator='support_binding',validator_value=group[0]['region_ids'],instance=a['evidence_refs'])
        if m.get('local_support_registry'):
            required=[rid+'/local_support_evidence' for rid in group[0]['region_ids'] if 'local_support_evidence' in byregion[rid]]
            if not set(required)<=set(a['evidence_refs']):
                import jsonschema
                raise jsonschema.ValidationError('Selected local support must cite its measured mining evidence',validator='local_support_evidence_binding',validator_value=required,instance=a['evidence_refs'])
        if modules and not any(rid+'/chemical_context' in a['evidence_refs'] for rid in group[0]['region_ids']):
            import jsonschema
            raise jsonschema.ValidationError('Selected support needs its OWN chemical context citation',validator='chemical_support_binding',validator_value=group[0]['region_ids'],instance=a['evidence_refs'])
        parent=parents[a['parent_program_id']]
        if 'execution_response' in parent and (parent['program']['typed_weight'] or parent['program']['charge_weight']):
            required=['parents/'+a['parent_program_id']+'/'+key for key in ('component_response','element_layout')]
            if not set(required)<=set(a['evidence_refs']):
                import jsonschema
                raise jsonschema.ValidationError('A chemical parent needs its OWN measured component and class-balanced layout evidence',validator='measured_chemical_parent',validator_value=required,instance=a['evidence_refs'])
        expected=CLOCKS[group[0]['program']['clock_policy']]
        if a['observable_contract']!=expected:
            import jsonschema
            raise jsonschema.ValidationError('Selected observable and teacher representation disagree',validator='clock_binding',validator_value=expected,instance=a['observable_contract'])
    messages.append({'role':'user','content':json.dumps({'valid_evidence_reference_ids':sorted(bundle['evidence']),
        'response_task':'Use the evidence ALREADY delivered in this conversation; it is not duplicated here. Return the scientific Analyst handoff using a group whose actual regional evidence you queried. State the primary registered operator. Designer chooses numerical variants inside that semantic group only.'},ensure_ascii=False)})
    a=call_json(messages,schema,folder/'Analyst.json',input_store=repo/m['campaign_study']/'inputs',response_validator=validate_a,max_output_tokens=5500)
    review=factual_review(repo,m,folder,'Analyst',a,{'execution':groups[a['group_id']][0]['program'],
        'authoritative_program_delta':groups[a['group_id']][0]['authoritative_program_delta'],
        'actual_parent_execution':executable_view(parents[a['parent_program_id']]['program']),
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
    visible_options=[{**v,'program':executable_view(v['program'])} for v in selected]
    cited=set(a['evidence_refs'])|set(a['counterevidence_refs'])|{'source/limits','source/feature_revision'}
    designer_evidence={k:v for k,v in bundle['evidence'].items() if k in cited}
    designer_schema=obj({'option_id':{'enum':[x['action_id'] for x in selected]},'hypothesis_group':{'const':a['group_id']},
        'parent_program_id':{'const':a['parent_program_id']},
        'evidence_refs':refs,'design_grounds':{'type':'string','minLength':1,'description':'One concise scientific reason; do not repeat canonical program/derivative/dose objects.'},
        'falsifier':{'type':'string','minLength':1,'description':'One concise measurable negative outcome.'}})
    d=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
        'Analyst':a,'shared_evidence':designer_evidence,'allowed_numerical_options':visible_options,
        'task':'Return ONLY the six schema fields: selected option_id, fixed hypothesis_group, fixed parent_program_id, evidence_refs, one concise design_grounds paragraph and one measurable falsifier. The deterministic compiler generates formulas, derivative paths, parameter changes and actual-dose descriptors from your selected ID; DO NOT output or rewrite these computed fields. Registry is bounded; choosing parameters is not unrestricted code synthesis.'},ensure_ascii=False)}],
        designer_schema,folder/'Designer.response.json',input_store=repo/m['campaign_study']/'inputs',max_output_tokens=3000)
    api_file=folder/'Designer.response.json';d=compile_designer_response(d,options[d['option_id']],api_file);write_json(folder/'Designer.json',d)
    action=options[d['option_id']];p=copy.deepcopy(action['program']);validate(p,bank)
    d_review=factual_review(repo,m,folder,'Designer',d,{'execution':p,'clock':CLOCKS[p['clock_policy']],'authoritative_program_delta':action['authoritative_program_delta'],
        'source_contract':bundle['contract'],'Analyst':a})
    if d_review['violations']:
        d=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
            'Analyst':a,'allowed_options':visible_options,'authoritative_clock':CLOCKS[p['clock_policy']],'authoritative_program_delta':action['authoritative_program_delta'],
            'previous_response':{k:d[k] for k in designer_schema['properties']},'neutral_factual_diagnostics':d_review,'task':'Correct factual contradictions; return only the six raw schema fields and keep the Analyst-bound hypothesis.'},ensure_ascii=False)}],
            designer_schema,folder/'Designer.corrected.response.json',input_store=repo/m['campaign_study']/'inputs',max_output_tokens=3000)
        api_file=folder/'Designer.corrected.response.json';d=compile_designer_response(d,options[d['option_id']],api_file);write_json(folder/'Designer.corrected.json',d)
        action=options[d['option_id']];p=copy.deepcopy(action['program']);validate(p,bank);designer_file=folder/'Designer.corrected.json'
        final_review=factual_review(repo,m,folder,'Designer.corrected',d,{'execution':p,'clock':CLOCKS[p['clock_policy']],'authoritative_program_delta':action['authoritative_program_delta'],
            'source_contract':bundle['contract'],'Analyst':a})
        if final_review['violations']:raise ValueError('Designer remains factually inconsistent after bounded correction')
    else:designer_file=folder/'Designer.json'
    if action not in selected:raise ValueError('Designer escaped Analyst hypothesis')
    p.update(analyst_sha256=digest(analyst_file),designer_sha256=digest(designer_file),evidence_bundle_sha256=digest(folder/'evidence_bundle.json'))
    delivery={'model':'deepseek-flash','real_api':True,'temperature':0,'active_modules':modules,
        'analyst_response_path':analyst_file.name,'designer_response_path':designer_file.name,
        'designer_api_response_path':api_file.name,'designer_api_response_sha256':digest(api_file),
        'skill_sha256':{r:hashlib.sha256(v.encode()).hexdigest() for r,v in skills.items()},
        'query_request_sha256':receipts,'analyst_sha256':p['analyst_sha256'],'designer_sha256':p['designer_sha256'],
        'bound_region_ids':action['region_ids'],'numeric_identity':numerical_identity(p,bank),
        'analysis_scope':'Evidence, decisions and falsifiers; no private reasoning transcript.'}
    write_json(folder/'delivery.json',delivery);write_json(folder/'decision.json',{'Analyst':a,'Designer':d,'action':{k:v for k,v in action.items() if k!='program'}})
    write_json(folder/'selected_program.json',p)
    return p,action,delivery
