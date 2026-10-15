"""Bounded real-API query/observe loop with a shared immutable evidence bundle."""
import copy
import hashlib
import json
import jsonschema
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.partial_search.report import api_audit
from .reward import OPERATORS
from .semantic_contract import references,variants,validate_program

FACTORS=['region','operator','dose','band_multiplier','controller','matching','representation','window_start','gradient_floor','curvature']
TOOLS=['conditional_layout','regional_context','regional_trajectories','control_response','history']
OPERATOR_DEFINITIONS={
    'pointwise':'Implemented and derivative-tested: local Hungarian assignment, robust per-point distance to observed teacher coordinates. Units A; no chemical-role or energy assumption.',
    'point_band':'Implemented and derivative-tested: squared excess of local point distance outside a calibrated spatial tolerance. A coordinate intervention to be tested, not a proven affinity rule.',
    'pocket_distance':'Implemented and derivative-tested: local selected-point distances to fixed pocket sites, compared with source distances within a proposed tolerance. Uses provided coordinates, NOT an interaction energy or established H bond.',
    'orientation':'Implemented and derivative-tested: directed local neighbor rays toward observed teacher rays; dimensionless angular mismatch. Requires supplied orientation_triplets; no direct element force.',
    'pair_distance':'Implemented and derivative-tested: local pair-distance intervals around observed teacher distances, units A. Does not freeze bonds or graph.'}


def ask(repo,master,folder,evidence,templates,parents,feedback,stage='from_zero',fixed=None):
    folder.mkdir(parents=True,exist_ok=True);skills={}
    for role in ['Analyst','Designer']:
        skills[role]=(repo/master[role.lower()+'_skill']).read_text(encoding='utf-8')
        (folder/(role+'.SKILL.md')).write_text(skills[role],encoding='utf-8')
    store=repo/master['campaign_study']/'inputs';regions={r['region_id']:r for r in evidence['regions']}
    # Every round can discover another region. Frozen confirmations are explicit.
    catalog=evidence['catalog']
    if fixed is not None:
        regions={fixed['region_id']:regions[fixed['region_id']]}
        catalog=[r for r in catalog if r['region_id']==fixed['region_id']]
    bundle={'evidence':{'report/O-intervention':evidence['O_evidence'],'report/limits':evidence['limits'],
        'parent/response':feedback.get('parent_response'), 'parent/outcomes':feedback.get('parent_outcomes')},
        'learning_window':master['learning_window'],'stage':stage,'implemented_operators':OPERATOR_DEFINITIONS}
    query_schema=obj({'action':{'enum':['query','finish']},'requests':{'type':'array','maxItems':2,'items':obj({
        'tool':{'enum':TOOLS},'region_id':{'enum':list(regions)}})},'question':{'type':'string','maxLength':6000}})
    messages=[{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps({
        'catalog':catalog,'parents':parents,'feedback':feedback,'available_tools':TOOLS,'shared_initial_evidence':bundle,
        'research_stage':stage,'fixed_confirmation':fixed,'budget':{'max_tool_rounds':3,'max_calls_each':2},
        'task':'You are in the tool-planning phase. Return a brief tool question, not the final scientific handoff. Query at least one registered tool for support/counterevidence, then finish. The listed geometry calculators ARE implemented and tested; they do not require a binding-energy calculator. Missing previous terminal benefit is a reason for a prospective experiment, not a capability gap. New operators outside this registry require capability_gap. One principal intervention per from-zero experiment.'},ensure_ascii=False)}]
    queried=set();query_count=0
    for turn in range(3):
        q=call_json(messages,query_schema,folder/f'Analyst.query{turn}.json',input_store=store,max_output_tokens=1200,max_format_repairs=2)
        if q['action']=='finish' and queried:break
        if not q['requests']:raise ValueError('A query needs actual registered tools')
        observations={}
        for request in q['requests']:
            rid=request['region_id'];tool=request['tool'];queried.add(rid);card=evidence['cards'][rid]
            if tool in ('conditional_layout','regional_context'):value=card[tool if tool=='conditional_layout' else 'context']
            elif tool=='regional_trajectories':value=evidence['trajectory']['fits'].get(rid,{'status':'missing_trajectory'})
            elif tool=='control_response':
                value={'matching_parents':{pid:entry for pid,entry in parents.items() if entry['program']['region_id']==rid},
                    'active_incumbent_context':feedback.get('parent_response'),
                    'rule':'Use the response whose program ID and region match the chosen parent; active-incumbent context is not a substitute.'}
            else:value={'historical':evidence['history'],'current_campaign':feedback.get('history',[])}
            eid=rid+'/'+tool;observations[eid]=value;bundle['evidence'][eid]=value;query_count+=1
        messages.extend([{'role':'assistant','content':json.dumps(q)}, {'role':'user','content':json.dumps({'registered_observations':observations,'remaining_tool_rounds':2-turn})}])
    write_json(folder/'evidence_bundle.json',bundle);bundle_hash=digest(folder/'evidence_bundle.json')
    text={'type':'string','maxLength':3000};refs={'type':'array','items':{'enum':list(bundle['evidence'])},'uniqueItems':True}
    schema=obj({'role':{'const':'Analyst'},'status':{'enum':['hypothesis','capability_gap','scientific_defer']},
        'hypothesis_id':{'type':'string','maxLength':120},'parent_program_id':{'enum':list(parents)},
        'region_id':{'enum':sorted(queried)},'factor':{'enum':FACTORS},'observable_id':{'enum':list(OPERATORS)},
        'observable_definition':{'enum':list(OPERATOR_DEFINITIONS.values())},
        'evidence_time_domain':{'const':master['learning_window']},
        'evidence_refs':{**refs,'minItems':1},'counterevidence_refs':refs,'claim_class':{'enum':['inferred','proposed']},
        'conditioning':text,'contrast_definition':text,'scientific_hypothesis':text,
        'predicted_execution_response':text,'predicted_terminal_response':text,'falsifier':text,
        'source_layout_implies_generated_benefit':{'const':False},'temporal_slope_is_spatial_gradient':{'const':False}})
    def validate_a(a):
        references(a,bundle,a['region_id'])
        if a['observable_definition']!=OPERATOR_DEFINITIONS[a['observable_id']]:
            raise jsonschema.ValidationError('Observable definition differs from registered operator',validator='const',validator_value=OPERATOR_DEFINITIONS[a['observable_id']],instance=a['observable_definition'])
        if fixed is not None and a['observable_id']!=fixed['operator']:
            raise jsonschema.ValidationError('Frozen confirmation cannot redesign the operator',validator='const',validator_value=fixed['operator'],instance=a['observable_id'])
        parent=parents[a['parent_program_id']]['program']
        if stage=='from_zero' and a['factor']!='region' and a['region_id']!=parent['region_id']:
            raise jsonschema.ValidationError('Only the region intervention may change support or teacher',validator='region_binding',validator_value=parent['region_id'])
        if a['factor']!='operator' and a['observable_id']!=parent['operator'] and stage=='from_zero':
            raise jsonschema.ValidationError('Changing observable requires operator intervention',validator='operator_binding',validator_value=parent['operator'])
        if stage=='from_zero' and a['factor']=='region' and a['region_id']==parent['region_id']:
            raise jsonschema.ValidationError('Changing a region to itself is a numerical no-op; choose an actual registered intervention',validator='numerical_noop',validator_value='different region or a different factor')
    payload={'stage':stage,'parents':parents,'feedback':feedback,'shared_complete_evidence_bundle':bundle,
        'fixed_confirmation':fixed,'task':'Return one conditional coordinate hypothesis with explicit counterevidence, executable observable and a discriminating test. All listed coordinate calculators are implemented; an unproved head benefit does not make geometry unavailable. Do not demand a physical interaction energy to test a geometric hypothesis. Use source IDs. For a region-only change keep the parent operator. The evidence clock domain is the complete learning_window, distinct from the control window. Source scores are not outcome forecasts.'}
    a=call_json([{'role':'system','content':skills['Analyst']},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}],
        schema,folder/'Analyst.json',input_store=store,max_output_tokens=3500,max_format_repairs=2,response_validator=validate_a)
    if a['status']!='hypothesis':return None,a,None,{'status':a['status'],'query_count':query_count}
    r=regions[a['region_id']];t=templates[r['template_id']];parent=parents[a['parent_program_id']]['program']
    if stage=='suffix':
        # One newly specified local intervention versus no intervention; not a
        # claim of single-factor comparison against an unrelated teacher.
        from .semantic_contract import bind_region
        p=bind_region(copy.deepcopy(parent),r,t);p['operator']=a['observable_id']
        options={'V00':validate_program(p,t,master['learning_window'])}
    elif fixed is not None:options={'V00':copy.deepcopy(fixed)}
    else:options=variants(parent,a,r,t,master)
    if not options:return None,a,None,{'status':'capability_gap','reason':'No supported numeric variant','query_count':query_count}
    dschema=obj({'role':{'const':'Designer'},'decision':{'enum':['execute','defer']},'hypothesis_id':{'const':a['hypothesis_id']},
        'evidence_bundle_hash':{'const':bundle_hash},'variant_id':{'enum':list(options)},
        'executed_operator':{'const':a['observable_id']},'enabled_terms':{'const':[a['observable_id']]},
        'operator_definition':{'const':OPERATOR_DEFINITIONS[a['observable_id']]},'controlled_atom_slots':{'const':r['atom_slots']},
        'evidence_refs':{**refs,'minItems':1},'counterevidence_refs':refs,
        'scientific_rationale':text,'parameter_source':text,'predicted_control_response':text,'falsifier':text,
        'direct_type_force':{'const':False}})
    def validate_d(d):references(d,bundle)
    design_messages=[{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
        'Analyst':a,'selected_region':r,'parent':parent,'numeric_variants':options,
        'measured_selected_parent_response':parents[a['parent_program_id']].get('response'),
        'measured_selected_parent_outcomes':parents[a['parent_program_id']].get('metrics'),
        'shared_complete_evidence_bundle':bundle,'evidence_bundle_hash':bundle_hash,
        'parameter_semantics':{'dose':'Coordinate RMS relative to predictive flow, before caps.',
            'band_multiplier':'Scales source geometry calibration band, not a proved optimum.',
            'controller':'unit_rms or first-active-gradient-calibrated bounded dose attenuation.',
            'matching':'Detached element likelihood cost for assignment, not an elemental force.',
            'representation':'bridge uses actual intermediate teacher; endpoint uses observed terminal coordinates.'},
        'task':'Choose one executable variant bound to Analyst. Your executed_operator/enabled_terms/controlled_atom_slots are authoritative and prose must describe that same function. Inactive pocket-distance, type or orientation terms cannot be claimed as direct forces. No type, affinity or categorical-logit gradient. Give an execution-level falsifier and terminal consequence. A new hypothesis does not require already demonstrated benefit.'},ensure_ascii=False)}]
    d=call_json(design_messages,dschema,folder/'Designer.json',input_store=store,max_output_tokens=3000,max_format_repairs=2,response_validator=validate_d)
    accepted_path=folder/'Designer.json'
    audit_schema=obj({'consistent':{'type':'boolean'},'issues':{'type':'array','maxItems':3,'items':obj({
        'field':{'enum':['scientific_rationale','parameter_source','predicted_control_response','falsifier']},
        'quote':{'type':'string','maxLength':800},'contradiction':{'type':'string','maxLength':1200}})}})
    for revision in range(2):
        def validate_audit(c):
            if c['consistent'] and c['issues']:raise jsonschema.ValidationError('Consistent audit cannot list contradictions')
            for issue in c['issues']:
                if not issue['quote'] or issue['quote'] not in d[issue['field']]:
                    raise jsonschema.ValidationError('Audit quote must occur verbatim in the declared Designer field',validator='verbatim_quote',validator_value='exact Designer quote',instance=issue['quote'])
        check=call_json([{'role':'system','content':skills['Designer']},{'role':'user','content':json.dumps({
            'mode':'Implementation consistency audit, not an efficacy review','Designer_output':d,
            'actual_compiled_program':options[d['variant_id']], 'registered_operator':OPERATOR_DEFINITIONS[a['observable_id']],
            'task':'Check only whether prose claims a different active operator, target, direct chemical force, support, time or dose than the actual program. Merely mentioning alternatives or unproved future benefit is not a contradiction. Quote each actual contradiction verbatim. Do not invent an issue or require known terminal success.'})}],
            audit_schema,folder/f'Designer.audit{revision}.json',input_store=store,max_output_tokens=1800,max_format_repairs=2,response_validator=validate_audit)
        if check['consistent']:break
        if revision==1:return None,a,d,{'status':'semantic_inconsistency','query_count':query_count}
        d=call_json(design_messages+[{'role':'assistant','content':json.dumps(d)},{'role':'user','content':json.dumps({
            'implementation_audit':check,'task':'Correct the implementation claims, keeping Analyst binding and legal variants. If the mapping cannot support the Analyst hypothesis, defer. This is an explicit scientific correction, separate from neutral JSON format repair.'})}],
            dschema,folder/'Designer.revision.json',input_store=store,max_output_tokens=3000,max_format_repairs=2,response_validator=validate_d)
        accepted_path=folder/'Designer.revision.json'
    delivery={'Analyst':api_audit(folder,'Analyst',hashlib.sha256(skills['Analyst'].encode()).hexdigest()),
        'Designer':api_audit(folder,accepted_path.stem,hashlib.sha256(skills['Designer'].encode()).hexdigest())}
    write_json(folder/'delivery.json',{'roles':delivery,'query_count':query_count,'regions_queried':sorted(queried),
        'complete_shared_evidence_hash':bundle_hash,'n_numeric_variants':len(options),'accepted_designer_file':accepted_path.name,
        'implementation_audit_consistent':check['consistent']})
    if not all(v['skill_delivered_exactly'] and v['receipt_and_file_hashes_match'] and v['input_file_hashes_match'] for v in delivery.values()):raise ValueError('Agent instructions were not delivered')
    if d['decision']=='defer':return None,a,d,{'status':'scientific_defer','query_count':query_count}
    p=options[d['variant_id']];p.update(hypothesis_id=a['hypothesis_id'],analyst_sha256=digest(folder/'Analyst.json'),designer_sha256=digest(accepted_path),
        evidence_bundle_sha256=bundle_hash,enabled_terms=[p['operator']])
    return p,a,d,{'status':'execute','query_count':query_count}
