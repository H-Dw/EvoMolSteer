"""Strict agent contracts and semantic grounding checks, shared by all providers."""
from jsonschema import Draft202012Validator

def obj(properties,required=None):
    return {'type':'object','properties':properties,'required':list(properties) if required is None else required,'additionalProperties':False}
def arr(items):return {'type':'array','items':items}
S={'type':'string'};N={'type':'number'};A=arr(S)
RULE=obj({'rule_id':S,'region':S,'feature':S,'representation':{'enum':['predicted_endpoint','proposal_state']},'stage_start':N,'stage_end':N,
    'direction':{'enum':['increase','decrease','toward_window','defer']},'target_id':{'type':['string','null']},
    'status':{'enum':['exploratory','supported','deferred']},'evidence_ids':A,'counterevidence_ids':A,'rationale':S})
ANALYST=obj({'schema_version':{'const':'2.0'},'agent':{'const':'Analyst'},'dataset_id':S,
    'observations':arr(obj({'observation_id':S,'finding':S,'evidence_ids':A,'uncertainty':S})),
    'selection_comparison':S,'stage_dynamics':S,'rules':arr(RULE),'limitations':A})
TERM=obj({'term_id':S,'rule_id':S,'target_id':S,'feature':S,'lower':N,'upper':N,
    'scale':{'type':'number','exclusiveMinimum':0},'weight':{'type':'number','exclusiveMinimum':0,'maximum':10},
    'stage_start':{'type':'number','minimum':0,'maximum':1},'stage_end':{'type':'number','minimum':0,'maximum':1},
    'gate_width':{'type':'number','minimum':.005,'maximum':.1},'evidence_ids':A,'parameter_status':{'const':'exploratory_pilot'}})
PROGRAM=obj({'version':{'const':'2.0'},'representation':{'const':'predicted_endpoint_world_A'},
    'selection_window':obj({'start':N,'end':N,'time_axis':{'const':'score_time'}}),
    'direction':{'const':'reward_ascent'},'operator':{'const':'negative_smooth_window_sum'},'terms':arr(TERM),
    'constraints':obj({'max_atom_displacement_A':{'type':'number','exclusiveMinimum':0,'maximum':.1},
        'max_rms_displacement_A':{'type':'number','exclusiveMinimum':0,'maximum':.1},
        'clash_threshold_A':{'const':1.2},'max_new_clash_pairs':{'const':0},'preserve_fixed_atoms':{'const':True}})})
DESIGNER=obj({'schema_version':{'const':'2.0'},'agent':{'const':'Designer'},'dataset_id':S,
    'status':{'enum':['exploratory_offline_candidate','deferred']},'program':PROGRAM,
    'selected_rule_ids':A,'deferred_rules':arr(obj({'rule_id':S,'reason':S})),
    'design_rationale':S,'gradient_path':{'const':'saved_endpoint_only_live_generator_jacobian_not_validated'},
    'failure_modes':A,'validation_plan':A})

def validate(role,data,bundle,analyst=None):
    if role not in ('Analyst','Designer'):raise ValueError('Unknown agent role')
    import math
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite agent number')
        if isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    finite(data)
    if bundle.get('schema_version')!='2.0' or bundle['scope'].get('analysis_scope')!='actual_resampling_events':
        raise ValueError('Only selection-window evidence is supported')
    Draft202012Validator(ANALYST if role=='Analyst' else DESIGNER).validate(data)
    if data['dataset_id']!=bundle['dataset_id']:raise ValueError('Dataset identity mismatch')
    evidence_rows={r['evidence_id']:r for r in bundle['evidence']}
    evidence=set(evidence_rows); targets={r['target_id']:r for r in bundle['targets']}
    features=bundle['feature_catalog']['features']
    scope=bundle['scope']
    windows={(s['stage_start'],s['stage_end']) for s in scope['stages']}
    def ids(values):
        if not set(values)<=evidence:raise ValueError('Unknown evidence ID')
    if role=='Analyst':
        seen=set()
        for o in data['observations']:ids(o['evidence_ids'])
        for r in data['rules']:
            if r['rule_id'] in seen:raise ValueError('Duplicate rule ID')
            seen.add(r['rule_id']);ids(r['evidence_ids']);ids(r['counterevidence_ids'])
            if not r['evidence_ids']:raise ValueError('Rule has no evidence')
            if r['feature'] not in features or features[r['feature']]['region']!=r['region']:raise ValueError('Unknown feature/region')
            if not any(evidence_rows[e].get('feature')==r['feature'] and evidence_rows[e].get('representation')==r['representation'] for e in r['evidence_ids']):raise ValueError('No evidence for rule feature/representation')
            if (r['stage_start'],r['stage_end']) not in windows:raise ValueError('Rule is outside observed selection stages')
            rule_stage=next(s['stage'] for s in scope['stages'] if (s['stage_start'],s['stage_end'])==(r['stage_start'],r['stage_end']))
            if not any(evidence_rows[e].get('feature')==r['feature'] and evidence_rows[e].get('representation')==r['representation']
                       and evidence_rows[e].get('stage')==rule_stage for e in r['evidence_ids']):
                raise ValueError('Rule lacks same-stage evidence')
            if r['target_id'] is not None:
                target=targets.get(r['target_id'])
                if not target or target['feature']!=r['feature'] or target['representation']!=r['representation']:raise ValueError('Invalid target reference')
                if (r['stage_start'],r['stage_end'])!=(target['stage_start'],target['stage_end']):raise ValueError('Target stage mismatch')
            if r['status']=='supported':
                target=targets.get(r['target_id'],{})
                corrected=[evidence_rows[e].get('q_value') for e in r['evidence_ids']
                    if evidence_rows[e].get('feature')==r['feature'] and evidence_rows[e].get('representation')==r['representation']
                    and evidence_rows[e].get('statistic')=='effects'
                    and evidence_rows[e].get('contrast') in ('expected_selection_shift','expected_high_mass_shift')
                    and any(s['stage']==evidence_rows[e].get('stage') and (s['stage_start'],s['stage_end'])==(r['stage_start'],r['stage_end']) for s in scope['stages'])]
                if target.get('support_batches',0)<4 or not any(q is not None and q<=.05 for q in corrected):
                    raise ValueError('Supported rule requires >=4 batches and corrected same-stage expected-selection evidence')
    else:
        if analyst is None:raise ValueError('Designer requires validated Analyst output')
        window=data['program']['selection_window']
        if (window['start'],window['end'])!=(scope['window_start'],scope['window_end']):raise ValueError('Reward window differs from observed selection window')
        rules={r['rule_id']:r for r in analyst['rules']};seen=set()
        for t in data['program']['terms']:
            if t['term_id'] in seen:raise ValueError('Duplicate term ID')
            seen.add(t['term_id']);ids(t['evidence_ids'])
            if not t['evidence_ids']:raise ValueError('Term lacks evidence')
            r=rules.get(t['rule_id']);target=targets.get(t['target_id'])
            if not r or r['status']=='deferred' or r['target_id']!=t['target_id'] or target is None:raise ValueError('Unsupported Analyst rule')
            if r['representation']!='predicted_endpoint':raise ValueError('Reward compiler supports only predicted endpoint rules')
            if not set(t['evidence_ids'])&set(r['evidence_ids']):raise ValueError('Reward term must preserve rule evidence')
            if not features.get(t['feature'],{}).get('differentiable_supported'):raise ValueError('Unsupported differentiable feature')
            for field in ['feature','lower','upper','scale','stage_start','stage_end']:
                if t[field]!=target[field]:raise ValueError('Term differs from empirical target: '+field)
            if t['lower']>t['upper']:raise ValueError('Invalid window')
        if set(data['selected_rule_ids'])!={t['rule_id'] for t in data['program']['terms']}:raise ValueError('Selected-rule inventory mismatch')
        if data['status']=='deferred' and data['program']['terms']:raise ValueError('Deferred program must be empty')
    return data
