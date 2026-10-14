"""Strict real-API handoff and compiler for partial-region interventions."""
import copy
import hashlib
import json
from pathlib import Path
import jsonschema
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj,arr
from evomolsteer.final_guidance.reward import HalfStateReward,active_updates
from evomolsteer.final_guidance.workflow import CONTROL_SEMANTICS


STRATEGIES=[
('incremental_discrimination','Prioritize positive incremental regional affinity association over raw whole-pose similarity; compare several spatial extents.'),
('hard_negatives','Use the nearest low-scoring final counterexamples to reject geometry that is shared with failures; explain the discriminating addition or omission.'),
('context_preserving_core','Test a broader coherent core that preserves surrounding spatial organization while freeing a smaller region; contrast with tiny-point overfitting.'),
('leave_one_out','Consider near-complete partial clouds and free the point with weakest or adverse incremental evidence; avoid interpreting one free point as guaranteed improvement.'),
('pocket_context','Compare untyped receptor placement and local burial with incremental evidence; choose region placement with a plausible spatial role, without typed-contact claims.'),
('core_rim_tradeoff','Compare buried core and exposed rim constraints. Preserve supported spatial organization and explicitly identify what remains adaptable.'),
('tail_specificity','Focus on regional geometry that distinguishes higher-scoring remaining finals from lower-scoring near-matches, not similarity to the one teacher alone.'),
('dependence_robustness','Let lineage-cluster intervals and excluded-teacher-batch sensitivity change the selection, rather than quoting them as a disclaimer.'),
('negative_region_evidence','Actively avoid a point or region whose conditional geometric association is adverse, while maintaining a supported spatial context.'),
('complementary_motifs','Consider two complementary registered regions whose union preserves distinct spatial placements; justify why their union adds context instead of redundant points.'),
('minimal_sufficient_context','Use actual terminal feedback to reduce unnecessary points without losing the spatial constraints needed for high-affinity valid tails.'),
('observable_transfer','Identify which final observable a half-cloud proxy previously failed to reproduce. Choose points that have a testable route to reproducing it, not merely a low matching loss.'),
('conditional_point_importance','Use point-level incremental evidence to compare equal weights with supported importance weights; explain every weight through a delivered point evidence ID.'),
('distribution_compatibility','Use observed validity and tail-versus-mean changes to choose a region that several starting seeds can accommodate, rather than optimizing one isolated high score.'),
('counterexample_refinement','Revisit a previously selected region using its actual terminal counterexamples; remove or change a spatial constraint whose expected response was contradicted.'),
('context_expansion','Expand the best partial region only where the added points have useful geometry evidence, and explain the remaining freedom.'),
('supported_weighting','Keep a supported region and propose geometry-grounded point importance; do not treat an affinity association coefficient as a calibrated causal weight.'),
('synthesis','Combine only region-selection rules that produced distinct real executable changes and improved terminal feedback; discard unsupported narrative additions.'),
('confirmation','Re-evaluate the best frozen selection on more matched random inputs; do not invent a new region to make the confirmation pass.'),
('independent_validation','Validate the frozen best reward on an untouched initial batch; do not tune using that batch before reporting its result.')]


def fail(message):raise jsonschema.ValidationError(message,validator='partial_binding',validator_value=message,schema={})


def validate_analysis(answer,packet):
    regions={r['region_id']:r for r in packet['regions']}
    if answer['candidate_id']!=packet['candidate']['candidate_id']:fail('Candidate mismatch')
    chosen=answer['region_ids']
    if len(chosen)!=len(set(chosen)) or not chosen:fail('Unique actual registered regions required')
    slots=sorted({s for key in chosen for s in regions[key]['atom_slots']})
    if slots!=answer['atom_slots'] or not 1<len(slots)<len(packet['terminal_coords_A']):fail('Exact proper-subset region union required')
    if {h['region_id'] for h in answer['hypotheses']}!=set(chosen):fail('Every activated region needs a hypothesis')
    for h in answer['hypotheses']:
        if h['evidence_id']!=regions[h['region_id']]['evidence_id']:fail('Unknown or wrong-region evidence')
    weights=answer['point_importance']
    if [w['slot'] for w in weights]!=slots or any(w['evidence_id']!=f"point/{w['slot']}" for w in weights):fail('Point importance must bind every approved slot in order')


def compile_programs(cfg,analyst,designer,teacher,analyst_sha,designer_sha,teacher_sha):
    slots=analyst['atom_slots'];weights=np.array([r['relative_weight'] for r in analyst['point_importance']],float)
    if designer['decision']=='defer':return []
    if designer['candidate_id']!=analyst['candidate_id'] or teacher['candidate_id']!=analyst['candidate_id']:raise ValueError('Designer changed candidate')
    if designer['atom_slots']!=slots or designer['region_ids']!=analyst['region_ids']:raise ValueError('Designer detached from Analyst spatial selection')
    expected=weights/weights.sum()
    if not np.allclose(designer['atom_weights'],expected,atol=1e-8,rtol=0):raise ValueError('Designer changed Analyst point importance')
    if teacher['representation']!='current_world_A' or teacher['time']!=cfg['teacher_time']:raise ValueError('Actual ancestor clock required')
    if designer['dose']!=cfg['dose'] or designer['teacher_time']!=cfg['teacher_time']:raise ValueError('Unregistered dose/clock intervention')
    semantics=copy.deepcopy(CONTROL_SEMANTICS);semantics['teacher_weights']='analyst_importance_normalized'
    if designer['control_semantics']!=semantics:raise ValueError('Reward sign or correspondence mismatch')
    if not all(0<=s<cfg['guidance_end']<=cfg['teacher_time']<1 for s in cfg['guidance_starts']):raise ValueError('Invalid dynamic guidance clocks')
    programs=[]
    for start in cfg['guidance_starts']:
        p={'schema_version':'analyst-bound-partial-1.0','program_id':f"{cfg['round_id']}_{start:g}",
            'candidate_id':analyst['candidate_id'],'family':'actual_half_state_pointcloud','scope':'local',
            'atom_slots':slots,'atom_weights':expected.tolist(),'region_ids':analyst['region_ids'],
            'teacher_time':cfg['teacher_time'],'teacher_representation':'current_world_A','teacher_sha256':teacher_sha,
            'window':[start,cfg['guidance_end']],'terminal_time':1.,'steps':cfg['steps'],'dose':cfg['dose'],
            'background_loss_weight':0,'expected_active_steps':active_updates(cfg['steps'],start,cfg['guidance_end']),
            'analyst_sha256':analyst_sha,'designer_sha256':designer_sha,'smc':False,'affinity_head_gradient':False,
            'graph_equality_gate':False,'coordinate_evaluation':'linear_bridge_to_actual_teacher_clock',
            'control_semantics':semantics,'formula':'weighted_negative_pseudo_Huber_matched_point_error'}
        HalfStateReward(p,teacher);programs.append(p)
    return programs


def signature(program):
    keys=['candidate_id','family','atom_slots','atom_weights','teacher_sha256','window','steps','dose','coordinate_evaluation']
    value={k:program[k] for k in keys}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def analyze(repo,cfg,feedback,caller=call_json):
    repo=Path(repo);out=repo/cfg['study']/'candidates'/cfg['candidate_id'];out.mkdir(parents=True,exist_ok=True)
    packet=read_json(repo/cfg['campaign_study']/'evidence.json');text={'type':'string','maxLength':2200}
    skill_path=repo/cfg['analyst_skill'];designer_path=repo/cfg['designer_skill']
    skill=skill_path.read_text(encoding='utf-8');dskill=designer_path.read_text(encoding='utf-8')
    ids=[r['region_id'] for r in packet['regions']];points=len(packet['terminal_coords_A'])
    schema=obj({'role':{'const':'Analyst'},'candidate_id':{'const':cfg['candidate_id']},'region_ids':arr({'enum':ids},1,2),
        'atom_slots':arr({'type':'integer','minimum':0,'maximum':points-1},2,points-1),
        'hypotheses':arr(obj({'region_id':{'enum':ids},'evidence_id':{'type':'string'},'advantage':text,'alternative':text,'falsifier':text}),1,2),
        'point_importance':arr(obj({'slot':{'type':'integer'},'relative_weight':{'type':'number','minimum':.1,'maximum':5},'evidence_id':{'type':'string'}}),2,points-1),
        'selection_rationale':text,'free_region_rationale':text,'expected_final_response':text})
    def validate(a):
        validate_analysis(a,packet)
        if not cfg.get('allow_importance',False) and any(w['relative_weight']!=1 for w in a['point_importance']):fail('Region-only intervention requires all relative weights exactly 1')
        if 'frozen_selection' in cfg and any(a[k]!=v for k,v in cfg['frozen_selection'].items()):fail('Confirmation cannot change the frozen Analyst selection')
        w=np.array([r['relative_weight'] for r in a['point_importance']],float);w=w/w.sum()
        for previous in cfg.get('previous_selections',[]):
            if a['atom_slots']==previous['atom_slots'] and np.allclose(w,previous['atom_weights'],atol=1e-10,rtol=0):
                fail('Exploratory round must test a different evidence-grounded partial program; exact replay is reserved for confirmation')
    analyst=caller([{'role':'system','content':skill},{'role':'user','content':json.dumps({'task':'Choose an evidence-grounded partial terminal region, before seeing its intermediate ancestor.',
        'packet':packet,'feedback':feedback,'frozen_dose':cfg['dose'],'relative_weights_allowed':cfg.get('allow_importance',False),
        'frozen_selection':cfg.get('frozen_selection'),'already_tested_selections':cfg.get('previous_selections',[]),
        'experiment_goal':'High-affinity valid final molecules approaching or exceeding the reference final score; intermediate matching is not the performance metric'},separators=(',',':'))}],
        schema,out/'Analyst.json',input_store=repo/cfg['campaign_study']/'inputs',max_output_tokens=5000,response_validator=validate,max_format_repairs=2)
    validate(analyst)
    # Teacher is supplied only after actual terminal decision is frozen.
    teacher=read_json(repo/cfg['base_study']/'candidates'/cfg['candidate_id']/'teacher.json');write_json(out/'teacher.json',teacher)
    terminal=np.asarray(packet['terminal_coords_A']);half=np.asarray(teacher['coords_A']);slots=analyst['atom_slots']
    transfer={'selected_slots':slots,'terminal_to_half_RMS_A':float(np.sqrt(np.square(terminal[slots]-half[slots]).sum(1).mean())),
        'terminal_Rg_A':float(np.sqrt(np.square(terminal[slots]-terminal[slots].mean(0)).sum(1).mean())),
        'half_Rg_A':float(np.sqrt(np.square(half[slots]-half[slots].mean(0)).sum(1).mean()))}
    weights=np.array([w['relative_weight'] for w in analyst['point_importance']],float);weights=np.round(weights/weights.sum(),12).tolist()
    semantics=copy.deepcopy(CONTROL_SEMANTICS);semantics['teacher_weights']='analyst_importance_normalized'
    dschema=obj({'role':{'const':'Designer'},'candidate_id':{'const':cfg['candidate_id']},'decision':{'enum':['activate','defer']},
        'region_ids':{'const':analyst['region_ids']},'atom_slots':{'const':slots},'atom_weights':{'const':weights},
        'teacher_time':{'const':cfg['teacher_time']},'dose':{'const':cfg['dose']},'control_semantics':{'const':semantics},
        'scientific_rationale':{'type':'string','maxLength':5000},'observable_falsifier':text})
    def validate_d(d):
        try:compile_programs(cfg,analyst,d,teacher,digest(out/'Analyst.json'),'pending',digest(out/'teacher.json'))
        except ValueError as e:fail(str(e))
    design=caller([{'role':'system','content':dskill},{'role':'user','content':json.dumps({'analyst':analyst,
        'selected_terminal_regions':[r for r in packet['regions'] if r['region_id'] in analyst['region_ids']],
        'actual_teacher':teacher,'transfer':transfer,'frozen_dose':cfg['dose'],'normalized_weights':weights,
        'required_control_semantics':semantics,'clock':{'starts':cfg['guidance_starts'],'end':cfg['guidance_end']},
        'task':'Design the declared robust partial reward using exactly the Analyst selection. No autonomous region substitution.'},separators=(',',':'))}],
        dschema,out/'Designer.json',input_store=repo/cfg['campaign_study']/'inputs',max_output_tokens=4000,response_validator=validate_d,max_format_repairs=2)
    programs=compile_programs(cfg,analyst,design,teacher,digest(out/'Analyst.json'),digest(out/'Designer.json'),digest(out/'teacher.json'))
    bindings=[]
    for p in programs:
        path=out/'programs'/(p['program_id']+'.json');write_json(path,p)
        bindings.append({'program':path.relative_to(repo).as_posix(),'sha256':digest(path),'numerical_signature':signature(p),
            'operative_text':f"Analyst-bound teacher slots {slots}, spatial assignments, normalized weights {weights}, reward ascent, window {p['window']}; no SMC or affinity derivative."})
    write_json(out/'bindings.json',{'programs':bindings,'teacher':(out/'teacher.json').relative_to(repo).as_posix(),
        'instruction_sha256':{'Analyst':hashlib.sha256(skill.encode()).hexdigest(),'Designer':hashlib.sha256(dskill.encode()).hexdigest()},
        'analyst_selected_slots':slots,'designer_slot_set_identical':True,'transfer':transfer})
    return programs
