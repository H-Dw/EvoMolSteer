"""Registered, single-factor research actions and operative numerical identities."""
import copy
import hashlib
import json
import inspect
import numpy as np
from evomolsteer.io import read_json,digest
from evomolsteer.final_guidance.reward import active_updates
from .reward import AncestorReward,temporal_gain,horizon


def program_delta(parent,candidate):
    """Executable declarations, with large source payloads represented by hashes."""
    ignored={'program_id','expected_active_steps','analyst_sha256','designer_sha256','evidence_bundle_sha256','teacher_sha256'}
    changed={}
    for k in sorted((set(parent)|set(candidate))-ignored):
        a=parent.get(k);b=candidate.get(k)
        if a==b:continue
        def compact(value):
            raw=json.dumps(value,sort_keys=True,separators=(',',':'))
            return value if len(raw)<2000 else {'sha256':hashlib.sha256(raw.encode()).hexdigest(),'bytes':len(raw.encode())}
        changed[k]={'before':compact(a),'after':compact(b)}
    return changed


def scientific_group_key(action):
    p=action['program'];chemical=bool(p['typed_weight'] or p['charge_weight'])
    semantics={k:p[k] for k in ('clock_policy','operator','routing','attainment_response')}
    semantics.update(typed_enabled=bool(p['typed_weight']),charge_enabled=bool(p['charge_weight']))
    semantics['element_aware_matching']=bool(p['assignment_element_cost'])
    if chemical:semantics['sensing']=p['sensing']
    if p['typed_weight']:
        semantics.update(typed_channels=p['typed_channels'],type_target=p['type_target'],type_balance=p.get('type_balance','none'))
    return json.dumps([action['parent_id'],action['region_ids'],action['phase'],semantics],sort_keys=True)


def numerical_payload(p,teacher):
    keys=['clock_policy','window','learning_window','steps','operator','dose','assignment_element_cost','routing','native_options']
    v={k:copy.deepcopy(p[k]) for k in keys}
    if p['operator']!='pointwise':v['dose'].pop('pointcloud_delta_A',None)
    v['modes']=[{'template_id':s['template_id'],'atom_slots':s['atom_slots'],
        'trajectory_sha256':teacher['modes'][s['template_id']]['trajectory_sha256']} for s in p['mode_specs']]
    v['dose_schedule']=p['dose_schedule'];v['schedule_normalization']=p['schedule_normalization'];v['attainment_response']=p['attainment_response']
    if p['clock_policy'] in ('lookahead','decaying_lookahead'):v['lookahead_fraction']=p['lookahead_fraction']
    if p['operator'] in ('point_band','pair_distance','pocket_distance') or p['attainment_response']:v['target_band_A']=p['target_band_A']
    if p['operator']=='orientation':v['triplets']=[s['orientation_triplets'] for s in p['mode_specs']]
    if p['operator']=='pocket_distance':v['sites']=[s['pocket_sites'] for s in p['mode_specs']]
    if p['routing']=='persistent':v.update(mode_memory_t=p['mode_memory_t'],switch_margin_A=p['switch_margin_A'])
    if p['routing'] in ('response','persistent_response'):
        v.update(response_refresh_fraction=p.get('response_refresh_fraction',.5),response_gap_floor=p.get('response_gap_floor',.01))
        if p['routing']=='persistent_response':v.update(mode_memory_t=p['mode_memory_t'],response_switch_fraction=p.get('response_switch_fraction',.2))
    if p['typed_weight'] or p['charge_weight']:
        for k in ('typed_weight','charge_weight','sensing','field_sigma_A'):v[k]=p[k]
    if p['typed_weight']:
        for k in ('typed_channels','type_target'):v[k]=p[k]
        v['type_balance']=p.get('type_balance','none')
        if p.get('type_balance')=='target_mass':v['type_mass_floor']=p.get('type_mass_floor',.05)
    if p['operator']=='selection_advantage':
        v.update(field_sigma_A=p['field_sigma_A'],density_scale_floor=p['density_scale_floor'],advantage_cap=p['advantage_cap'])
        v['advantage_evidence_gain']=p.get('advantage_evidence_gain',False)
        if p.get('advantage_evidence_gain'):v['advantage_evidence_scale']=p.get('advantage_evidence_scale',1.)
        v['profiles']=[s['selection_profile'] for s in p['mode_specs']]
    return v


def numerical_identity(p,teacher):
    return hashlib.sha256(json.dumps(numerical_payload(p,teacher),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def executable_view(p):
    """Same active-parameter registry as identity; omit inherited inactive fields."""
    placeholder={'modes':{s['template_id']:{'trajectory_sha256':'bound_source'} for s in p['mode_specs']}}
    view=numerical_payload(p,placeholder)
    for mode in view['modes']:mode.pop('trajectory_sha256')
    if 'profiles' in view:
        view['profiles']=[{'feature_representation':s.get('feature_representation','current'),'domain':[s['times'][0],s['times'][-1]],
            'feature_dim':4,'full_data_sha256':hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()} for s in view['profiles']]
    view['program_id']=p.get('program_id');view['operator_formula']={
        'pointwise':'R_geo = mean[-delta^2 * (sqrt(1+distance^2/delta^2)-1)]',
        'point_band':'R_geo = -mean[max(distance-band,0)^2]',
        'pair_distance':'R_geo = -mean[max(abs(student_pair_distance-teacher_pair_distance)-band,0)^2]',
        'pocket_distance':'R_geo = -mean[max(abs(student_site_distance-teacher_site_distance)-band,0)^2]',
        'orientation':'R_geo = -mean[(1-cos(student_neighbor_vector,teacher_neighbor_vector))^2] over declared triplets only',
        'selection_advantage':'R_geo = clip(mean[-0.5*((phi-mu_focal)/sd)^2+0.5*((phi-mu_background)/sd)^2],-cap,cap); phi = kernel occupancy and signed dx/dy/dz to fixed terminal centers. NO pointwise/pseudo-Huber term.'}[p['operator']]
    view['actual_terms']=['registered_'+p['operator']]+(['typed_spatial_field'] if p['typed_weight'] else [])+(['charge_spatial_field'] if p['charge_weight'] else [])
    if p.get('advantage_evidence_gain',False):view['dose_response_formula']='After gradient RMS normalization multiply displacement by s/sqrt(s^2+evidence_scale^2), where s=RMS((mu_focal-mu_background)/sd); dose is NOT the parent constant full displacement.'
    return view


def canonical_description(p):
    view=executable_view(p);view.pop('program_id',None)
    return json.dumps(view,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def derivative_binding(p):
    paths=[] if p['clock_policy']=='endpoint_prediction' else ['direct_current_coordinates']
    if p['clock_policy']!='current':paths.append('live_generator_endpoint_coordinate_VJP')
    if p['typed_weight'] or p['charge_weight']:
        paths.append('live_generator_endpoint_coordinate_VJP')
        if p['sensing']=='full':paths.append('live_generation_probability_head_coordinate_VJP')
    return {'coordinate_paths':sorted(set(paths)),'matching_and_routing':'stop_gradient','affinity_derivative':False,'extra_affinity_forward':False}


def dose_binding(p):
    gains=['temporal_schedule']
    if p['attainment_response']:gains.append('attainment_response')
    if p.get('advantage_evidence_gain',False):gains.append('historical_standardized_contrast')
    return {'post_RMS_gains':gains,'dose':{k:v for k,v in p['dose'].items() if k!='pointcloud_delta_A'},
        'schedule':p['dose_schedule'],'schedule_normalization':p['schedule_normalization'],'actual_budget_depends_on_student_trajectory':True}


def compile_designer_response(raw,action,api_path):
    expected={'option_id','hypothesis_group','parent_program_id','evidence_refs','design_grounds','falsifier'}
    if set(raw)!=expected or raw['option_id']!=action['action_id'] or raw['parent_program_id']!=action['parent_id']:
        raise ValueError('Raw Designer response escaped the option/parent contract')
    if read_json(api_path)!=raw:raise ValueError('Designer response differs from recorded real API output')
    p=action['program']
    return {**raw,'primary_operator':p['operator'],'execution_changes':sorted(action['authoritative_program_delta']),
        'operative_terms':json.loads(canonical_description(p)),'derivative_path':derivative_binding(p),'actual_dose_expectation':dose_binding(p),
        'api_response_path':api_path.name,'api_response_sha256':digest(api_path),
        'binding_origin':'deterministic compiler from validated option_id; rationale/falsifier from real API'}


def implementation_identity():
    from .reward import AncestorControl,attainment_gain
    from .bank import lookup
    return hashlib.sha256(('\n'.join(inspect.getsource(x) for x in (AncestorReward,AncestorControl,temporal_gain,horizon,attainment_gain,lookup))).encode()).hexdigest()


def validate(p,teacher):
    AncestorReward(p,teacher)
    if p['routing'] in ('response','persistent_response'):
        if not 0<p.get('response_refresh_fraction',.5)<=1 or p.get('response_gap_floor',.01)<=0:raise ValueError('Invalid response routing scale')
        if p.get('response_switch_fraction',.2)<0:raise ValueError('Invalid response routing switch margin')
    if p['operator']=='selection_advantage' and p['assignment_element_cost']!=0:raise ValueError('Learned feature definition requires geometry-only correspondence')
    if p.get('advantage_evidence_gain',False) and (p['operator']!='selection_advantage' or p.get('advantage_evidence_scale',1.)<=0):raise ValueError('Invalid feature-evidence dose calibration')
    if p['dose_schedule'] not in ('constant','early','late','bell'):raise ValueError('Invalid dose schedule')
    for k in ('native_rms_ratio','max_atom_step_A','max_cumulative_rms_A'):
        if not 0<p['dose'][k]<=20:raise ValueError('Nonphysical/unsupported dose')
    if p['typed_weight']<0 or p['charge_weight']<0:raise ValueError('Negative chemical field weight')
    if p.get('type_balance')=='target_mass' and not 0<p.get('type_mass_floor',.05)<=1:raise ValueError('Invalid class-mass floor')
    p['expected_active_steps']=active_updates(p['steps'],*p['window'])
    return p


def spec(region):
    keys=['template_id','region_id','atom_slots','pocket_sites','orientation_triplets']
    return {k:copy.deepcopy(region[k]) for k in keys}


def base(repo,m,bank,regions):
    old=read_json(repo/m['initial_program']);r=next(x for x in regions if x['region_id']==old['region_id'])
    p={'family':'ancestral_modes_v3','scope':'local','program_id':'R15_V3_COMPATIBLE','clock_policy':'fixed',
        'learning_window':m['learning_window'],'window':old['window'],'steps':m['steps'],'teacher_time':m['learning_window'][1],
        'mode_specs':[spec(r)],'routing':'single','mode_memory_t':.05,'switch_margin_A':.15,
        'lookahead_fraction':.2,'operator':'pointwise','dose':copy.deepcopy(old['dose']),
        'dose_schedule':'constant','schedule_normalization':1.,'attainment_response':False,
        'assignment_element_cost':old['assignment_element_cost'],'target_band_A':old['base_band_A'],
        'typed_weight':0.,'charge_weight':0.,'sensing':'detached','field_sigma_A':m['field_sigma_A'],
        'typed_channels':'all','type_target':'predicted','density_scale_floor':[.03,.1,.1,.1],'advantage_cap':20.,
        'native_options':old['native_options'],'coordinate_updates_only':True,'smc':False,'affinity_head_gradient':False}
    return validate(p,bank)


PHASES={1:'current_clock',2:'endpoint_clock',3:'lookahead',4:'decaying_lookahead',5:'dose_schedule',6:'attainment',
    7:'window_start',8:'dose_scale',9:'mode_diversity',10:'compatible_routing',11:'persistent_routing',12:'response_routing',
    13:'typed_field',14:'typed_vjp',15:'field_width',16:'interface_support',17:'NC_support',18:'group_support',
    19:'size_matched_support',20:'orientation',21:'pocket_field',22:'pair_geometry',23:'charge_field',
    24:'selection_advantage',25:'density_scale',26:'selection_phase',27:'typed_channels',28:'type_reference',
    29:'curvature',30:'matching_cost',31:'adaptive_revisit',32:'module_deletion',33:'deterministic_selector',
    34:'confirmation_A',35:'confirmation_B'}


def normalize_schedule(p,reference_rows):
    p['schedule_normalization']=1.
    if p['dose_schedule']=='constant':return
    weights=[];gains=[]
    for r in reference_rows:
        t=r['time'];dt=1/p['steps']
        if p['window'][0]-2e-6<=t and t+dt<=p['window'][1]+2e-6:
            weights.append(r.get('predictive_flow_rms_A_mean',r.get('native_predictive_rms_A_mean',1.)))
            gains.append(temporal_gain(p,t,dt))
    if not weights or sum(weights)<=0:raise ValueError('Missing dose-budget reference')
    p['schedule_normalization']=float(np.average(gains,weights=weights))


def actions(repo,m,bank,regions,state,number):
    """A principal module per round; the API selects hypothesis and parameter option."""
    phase=PHASES[number];best=state['best'];parents=[best,state['reference']]
    if state.get('tail_candidate'):parents.append(state['tail_candidate'])
    prior=[r for r in state['rounds'] if r.get('program') and r.get('stage')=='discovery']
    if prior and not prior[-1].get('rollback'):parents.append(prior[-1])
    prerequisites={10:'mode_diversity',11:'compatible_routing',12:'persistent_routing',14:'typed_field',15:'typed_vjp',
        21:'interface_support',22:'group_support',23:'group_support',25:'selection_advantage',26:'selection_advantage',27:'typed_field',28:'typed_field'}
    if number in prerequisites:
        candidates=[r for r in prior if r.get('phase')==prerequisites[number]]
        if candidates:parents.append(candidates[-1])
    if number==27:
        for family in ('typed_vjp','field_width'):
            candidates=[r for r in prior if r.get('phase')==family]
            if candidates:parents.append(candidates[-1])
    # Canonicalize optional inactive defaults under the current identity schema;
    # preserve historical stored identities as dispatch provenance, not a no-op loophole.
    seen={numerical_identity(r['program'],bank) for r in state['rounds'] if r.get('program') and r.get('stage')=='discovery'}
    seen.add(numerical_identity(state['reference']['program'],bank));result=[];seen_options=set()
    all_modes=sorted(bank['modes'],key=lambda k:bank['modes'][k]['terminal_score'],reverse=True)
    def add(parent,changes,reason,region=None,modes=None):
        p=copy.deepcopy(parent['program']);p.update(copy.deepcopy(changes))
        if region:p['mode_specs']=[spec(region)];p['routing']='single'
        if modes:p['mode_specs']=modes
        if p['routing']=='single':p['mode_specs']=p['mode_specs'][:1]
        if p['operator']=='selection_advantage':
            from .bank import regional_evidence
            from .forecast_evidence import evidence as forecast_evidence
            for s in p['mode_specs']:
                rr=next(r for r in regions if r['region_id']==s['region_id']);ev=(forecast_evidence if p['clock_policy']=='endpoint_prediction' else regional_evidence)(repo,m,bank,rr)
                valid=[r for r in ev['rows'] if r['unique_controls']>0]
                if len(valid)<3 or valid[0]['time']>p['learning_window'][0]+2e-6 or valid[-1]['time']<p['learning_window'][1]-2e-6:return
                s['selection_profile']={'schema_version':2,'feature_centers_A':ev['feature_centers_A'],
                    'feature_representation':'endpoint_prediction' if p['clock_policy']=='endpoint_prediction' else 'current',
                    'times':[r['time'] for r in valid],**{k:[r[k] for r in valid] for k in ('focal','background_mean','background_sd')}}
        normalize_schedule(p,state['dose_reference_steps'])
        try:validate(p,bank)
        except ValueError:return
        nid=numerical_identity(p,bank)
        if nid in seen or nid in seen_options:return
        seen_options.add(nid);idx=f'A{len(result):03d}'
        result.append({'action_id':idx,'parent_id':parent['program']['program_id'],'phase':phase,'reason':reason,
            'region_ids':[s['region_id'] for s in p['mode_specs']],'numeric_identity':nid,'program':p})
    for parent in parents:
        p=parent['program'];width=m['learning_window'][1]-m['learning_window'][0]
        if phase=='current_clock':add(parent,{'clock_policy':'current'},'Current state matched to actual ancestor at the same clock; not an endpoint Jacobian claim')
        elif phase=='endpoint_clock':add(parent,{'clock_policy':'endpoint_prediction'},'Model endpoint prediction matched to teacher endpoint prediction at same forward time')
        elif phase in ('lookahead','decaying_lookahead'):
            for f in (.1,.2,.4):add(parent,{'clock_policy':'lookahead' if phase=='lookahead' else 'decaying_lookahead','lookahead_fraction':f},'Same-horizon predictive bridge and actual ancestor; configured phase advance')
        elif phase=='dose_schedule':
            for schedule in ('early','late','bell'):add(parent,{'dose_schedule':schedule},'Post-normalization schedule calibrated to parent-reference predicted-step budget')
        elif phase=='attainment':add(parent,{'attainment_response':not p['attainment_response']},'Attenuate actual displacement as geometric target is attained')
        elif phase=='window_start':
            for fraction in (0.,.2,.6):add(parent,{'window':[m['learning_window'][0]+fraction*width,p['window'][1]]},'Configured control onset within actual learning window')
        elif phase=='dose_scale':
            for f in (.5,1.5,2.):
                d=copy.deepcopy(p['dose']);d['native_rms_ratio']*=f;add(parent,{'dose':d},'Change total physical dose separately from time allocation')
        elif phase in ('mode_diversity','compatible_routing','persistent_routing','routing_memory','response_routing'):
            if phase=='mode_diversity':
                focal=p['mode_specs'][0];others=[k for k in all_modes if k!=focal['template_id'] and bank['modes'][k]['root_id']!=bank['modes'][focal['template_id']]['root_id']]
                for offset in (0,2,4):
                    modes=[copy.deepcopy(focal)]
                    for tid in others[offset:offset+2]:
                        candidates=[r for r in regions if r['template_id']==tid and len(r['atom_slots'])==len(focal['atom_slots'])]
                        if candidates:modes.append(spec(candidates[0]))
                    if len(modes)>1:add(parent,{'routing':'static'},'Multiple observed terminal modes with identical support count and static seed assignment',modes=modes)
            elif len(p['mode_specs'])>1:
                if phase=='compatible_routing':
                    add(parent,{'routing':'compatible'},'Each seed selects lowest local required spatial displacement; not a future-quality estimate')
                    add(parent,{'routing':'response','response_refresh_fraction':.5,'response_gap_floor':.01},'Two same-forward coordinate VJP probes select maximal fractional surrogate progress per RMS displacement; not future success probability')
                elif phase=='persistent_routing':
                    response=p['routing'] in ('response','persistent_response')
                    add(parent,{'routing':'persistent_response' if response else 'persistent','response_switch_fraction':.2},'Per-seed compatibility or measured response plus bounded switch memory')
                elif phase=='response_routing':
                    add(parent,{'routing':'persistent_response' if p['routing']=='persistent' else 'response',
                        'response_refresh_fraction':.5,'response_gap_floor':.01,'response_switch_fraction':.2},
                        'Replace nearest-distance compatibility with sparse same-forward real coordinate controllability probes; preserve support and existing memory')
                else:
                    for f in (.05,.15,.3):add(parent,{'mode_memory_t':width*f,'routing':'persistent_response' if p['routing']=='persistent_response' else 'persistent'},'Memory measured in configured generation time')
        elif phase in ('typed_field','charge_field'):
            key='typed_weight' if phase=='typed_field' else 'charge_weight'
            for w in (.5,2.,8.):add(parent,{key:w},'Local soft probability field; no direct categorical editing or affinity derivative')
        elif phase=='typed_vjp' and p['typed_weight']>0:add(parent,{'sensing':'full'},'Differentiate the same type field through generation probabilities and coordinates')
        elif phase=='field_width' and (p['typed_weight'] or p['charge_weight']):
            for f in (.5,2.):add(parent,{'field_sigma_A':p['field_sigma_A']*f},'Change spatial response width alone')
        elif phase in ('interface_support','NC_support','group_support'):
            family={'interface_support':'interface_role','NC_support':'NC_layout','group_support':'functional_group'}[phase]
            candidates=[r for r in regions if r.get('family')==family and r['template_id']==p['mode_specs'][0]['template_id']]
            for r in candidates:add(parent,{},'Analyst-selectable chemically grounded local support, needed neighbors included',region=r)
        elif phase=='size_matched_support':
            k=len(p['mode_specs'][0]['atom_slots']);tid=p['mode_specs'][0]['template_id'];v=bank['modes'][tid]
            x=np.asarray(v['terminal_coords_A']);slots=p['mode_specs'][0]['atom_slots'];center=x[slots].mean(0)
            # Deterministic geometric comparator, same point count and parent operator.
            for sign in (1.,-1.):
                ids=np.argsort(sign*np.square(x-center).sum(-1),kind='stable')[:k].tolist()
                r=copy.deepcopy(next(r for r in regions if r['region_id']==p['mode_specs'][0]['region_id']))
                r.update(region_id=r['region_id']+'_sizecontrol'+str(int(sign)),atom_slots=sorted(ids),family='size_matched_geometry',orientation_triplets=[])
                if not any(v['region_id']==r['region_id'] for v in regions):regions.append(r)
                add(parent,{},'Same-size geometric support comparator; not a chemical-role hypothesis',region=r)
        elif phase in ('orientation','pocket_field','pair_geometry'):
            op={'orientation':'orientation','pocket_field':'pocket_distance','pair_geometry':'pair_distance'}[phase]
            add(parent,{'operator':op},'Change registered local observable keeping support/dose fixed')
        elif phase=='selection_advantage':add(parent,{'operator':'selection_advantage','assignment_element_cost':0.,'field_sigma_A':m['field_sigma_A'],
            'typed_weight':0.,'charge_weight':0.,'attainment_response':False,'advantage_evidence_gain':True,'advantage_evidence_scale':1.},
            'Explicit fixed-space focal/background ratio with representation-matched profiles, geometry-only correspondence, analysis-calibrated width and post-RMS standardized-contrast attenuation; one complete geometry objective package, not an affinity probability')
        elif phase=='density_scale' and p['operator']=='selection_advantage':
            for f in (.5,2.):add(parent,{'density_scale_floor':(np.asarray(p['density_scale_floor'])*f).tolist()},'Calibrate uncertainty floor, not a claimed success probability')
        elif phase=='selection_phase' and p['operator']=='selection_advantage':
            for policy in ('current','lookahead'):add(parent,{'clock_policy':policy},'Separate current-state and forward local-feature advantage')
        elif phase=='typed_channels' and p['typed_weight']:
            for channels in ('NC','NO'):add(parent,{'typed_channels':channels},'Restrict soft typed fields to selected elemental channels, not universal acceptor roles')
            add(parent,{'type_balance':'target_mass','type_mass_floor':.05},'Normalize each elemental spatial field by bounded teacher field mass; a separate class-balance alternative against majority-element domination, without changing channels')
        elif phase=='type_reference' and p['typed_weight']:add(parent,{'type_target':'terminal'},'Contrast retrospective terminal identity field with same-time predicted identity field')
        elif phase=='curvature':
            for f in (.5,2.):
                d=copy.deepcopy(p['dose']);d['pointcloud_delta_A']*=f;add(parent,{'dose':d},'Change geometric robust response curvature separately from actual dose')
        elif phase=='matching_cost':
            for cost in (0.,.2,1.):add(parent,{'assignment_element_cost':cost},'Detached element-aware correspondence only')
        elif phase in ('module_deletion','adaptive_revisit','deterministic_selector'):
            for changes in ({'clock_policy':'fixed'},{'dose_schedule':'constant'},{'typed_weight':0.},{'charge_weight':0.},{'routing':'single'},{'attainment_response':False}):
                add(parent,changes,'Single-module deletion or return to simpler measured mechanism')
            for cost in (.1,.8):add(parent,{'assignment_element_cost':cost},'Revisit correspondence in changed scientific context')
            if phase in ('adaptive_revisit','deterministic_selector'):
                add(parent,{'dose_schedule':'late'},'Revisit late allocation after earlier-dose branch failed; calibrated reference budget, actual trajectory budget reported separately')
                for factor in (.95,1.05):
                    dose=copy.deepcopy(p['dose']);dose['native_rms_ratio']*=factor
                    add(parent,{'dose':dose},'Small physical-dose perturbation of a measured branch; distinguish fragile tail response from a large mechanism change')
                from .bank import joint_supports
                for r in joint_supports(bank,regions,p['mode_specs'][0]['template_id'],m['max_new_local_points']):
                    if not any(v['region_id']==r['region_id'] for v in regions):regions.append(r)
                    add(parent,{'operator':'pointwise','typed_weight':0.,'charge_weight':0.,'attainment_response':False,'advantage_evidence_gain':False},
                        'Local joint-anchor geometry package motivated by isolated-anchor failures and same-count point sensitivity; not proven synergy or a chemical graph constraint',region=r)
    return result
