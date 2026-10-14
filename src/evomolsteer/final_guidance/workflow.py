"""Real API terminal hypotheses, declarative rewards and exact compilation."""
import argparse
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.deepseek_api import call_json
from evomolsteer.agentic_steer.workflow import obj,arr
from .evidence import prepare,candidate_packet,trace_teacher
from .reward import HalfStateReward,active_updates


def failure(message):
    raise jsonschema.ValidationError(message,validator='decision_binding',validator_value=message,schema={})


CONTROL_SEMANTICS={'optimization':'reward_ascent','distance_error':'decreases_toward_teacher',
    'reward_value':'increases_toward_zero','teacher_weights':'uniform_normalized',
    'generated_atom_correspondence':'spatial_assignment_not_teacher_slot_indices',
    'hypothesis_implementation':'spatial_imitation_proxy_not_exact_feature_loss'}


def record_status(study,candidate,status):
    path=Path(study)/'ranked_finals.parquet';table=pd.read_parquet(path)
    found=table['candidate_id'].eq(candidate)
    if found.sum()!=1:raise ValueError('Processing status requires one registered final')
    table.loc[found,'processing_status']=status
    table.to_parquet(path,compression=None,index=False)


def validate_analyst(answer,packet):
    if answer['candidate_id']!=packet['candidate']['candidate_id']:failure('Wrong final candidate')
    hypotheses=answer['hypotheses'];regions={r['region_id']:r for r in packet['regions']};rows={r['evidence_id']:r for r in packet['one_to_rest']}
    if answer['decision']=='propose' and not hypotheses:failure('Propose requires an actual coordinate hypothesis')
    if answer['decision']=='defer' and hypotheses:failure('Deferred decision must not activate hypotheses')
    if len({h['id'] for h in hypotheses})!=len(hypotheses):failure('Duplicate hypothesis IDs')
    for h in hypotheses:
        region=regions[h['region_id']];row=rows[h['evidence_id']]
        if not region['local_usable']:failure('Region is empty or the whole molecule')
        if sorted(h['atom_slots'])!=region['atom_slots']:failure('Hypothesis slots must equal the registered terminal-region support')
        if row['region_id']!=h['region_id'] or row['observable']!=h['observable']:failure('Hypothesis observable and source row do not match')
        if row['candidate_value'] is None or row['candidate_minus_rest'] is None:failure('Unavailable coordinate contrast cannot support this hypothesis')
        suffix=int(h['evidence_id'].rsplit('/',1)[1])
        kind='occupancy' if suffix==0 else 'coordinate_offset' if suffix<=3 else 'axis_dispersion'
        if h['claim_kind']!=kind:failure('Claim kind does not describe the measured coordinate statistic')
        if suffix>=4 and len(h['atom_slots'])<2:failure('A one-point covariance is not a supported dispersion hypothesis')


def compile_programs(cfg,analyst,design,teacher,analyst_sha,designer_sha,teacher_sha):
    candidate=analyst['candidate_id'];hypotheses={h['id']:h for h in analyst['hypotheses']}
    if analyst['decision']!='propose' or design['decision']!='activate':return []
    if teacher['representation']!='current_world_A' or teacher['time']!=cfg['teacher_time']:raise ValueError('Forecast or wrong-clock teacher rejected')
    if design['candidate_id']!=candidate or teacher['candidate_id']!=candidate:raise ValueError('Candidate drift')
    if not design['hypothesis_ids'] or not set(design['hypothesis_ids'])<=set(hypotheses):raise ValueError('Unknown active hypothesis')
    slots=sorted({s for key in design['hypothesis_ids'] for s in hypotheses[key]['atom_slots']})
    if sorted(design['atom_slots'])!=slots:raise ValueError('Designer and executed slots differ')
    if design['reward_family']!='actual_half_state_pointcloud' or design['teacher_time']!=cfg['teacher_time'] or design['dose']!=cfg['dose']:
        raise ValueError('Declared formula, clock or parameters differ from frozen execution')
    if design['control_semantics']!=CONTROL_SEMANTICS:raise ValueError('Declared control direction or correspondence differs from executed loss')
    if not 0<=cfg['guidance_end']<=cfg['teacher_time']<cfg['terminal_time'] or cfg['terminal_time']!=1.:raise ValueError('Unsupported control or terminal clock')
    programs=[]
    for scope in ['local','full']:
        for start in cfg['guidance_starts']:
            if not 0<=start<cfg['guidance_end']:raise ValueError('Empty or unsupported guidance start')
            p={'schema_version':'final-half-pointcloud-program-1.0','program_id':f'{candidate}_{scope}_{start:g}',
                'candidate_id':candidate,'family':design['reward_family'],'scope':scope,
                'atom_slots':slots if scope=='local' else teacher['active_slots'],'teacher_time':cfg['teacher_time'],
                'teacher_representation':teacher['representation'],'teacher_sha256':teacher_sha,
                'window':[float(start),cfg['guidance_end']],'terminal_time':cfg['terminal_time'],'steps':cfg['steps'],
                'dose':cfg['dose'],'background_loss_weight':0 if scope=='local' else 1,
                'hypothesis_ids':design['hypothesis_ids'],'analyst_sha256':analyst_sha,'designer_sha256':designer_sha,
                'affinity_head_gradient':False,'smc':False,'graph_equality_gate':False,
                'expected_active_steps':active_updates(cfg['steps'],start,cfg['guidance_end']),
                'coordinate_evaluation':'linear_bridge_to_actual_teacher_clock',
                'formula':'negative_pseudo_Huber(mean squared matched point distances)',
                'comparison_role':'hypothesis_region' if scope=='local' else 'whole_cloud_control'}
            HalfStateReward(p,teacher)
            programs.append(p)
    return programs


def operative_text(program):
    return (f"Teacher {program['candidate_id']}, actual current time {program['teacher_time']:g}, "
        f"{program['scope']} point cloud, exact slots {program['atom_slots']}; "
        f"spatial assignment and negative pseudo-Huber loss with delta={program['dose']['pointcloud_delta_A']:g} A. "
        f"Guidance updates {program['expected_active_steps']}, window {program['window']}, "
        f"native RMS ratio={program['dose']['native_rms_ratio']:g}; remaining inference to {program['terminal_time']:g} "
        f"without guidance, SMC=False. Coordinates use the declared linear teacher-clock bridge. "
        'Gradient ascent reduces matched squared distance and increases the negative reward toward zero. '
        'Teacher slots are spatially assigned to generated atoms, not copied as generated atom indices. '
        'This is a spatial imitation proxy, not the exact occupancy or covariance observable.')


def analyze(repo,cfg,rank,caller=call_json):
    repo=Path(repo);study=repo/cfg['study'];packet=candidate_packet(repo,cfg,rank);candidate=packet['candidate']['candidate_id']
    out=study/'candidates'/candidate;root=repo/'skills'
    shared=(root/'final-region-guidance-workflow/references/contract.md').read_text(encoding='utf-8')
    analyst_skill=(root/'final-candidate-analyst/SKILL.md').read_text(encoding='utf-8')
    text={'type':'string','maxLength':1800};regions=[r['region_id'] for r in packet['regions']]
    schema=obj({'role':{'const':'Analyst'},'candidate_id':{'const':candidate},'decision':{'enum':['propose','defer']},
        'rationale':text,'hypotheses':arr(obj({'id':{'type':'string','maxLength':80},'region_id':{'enum':regions},
            'atom_slots':arr({'type':'integer','minimum':0,'maximum':len(packet['terminal_coords_A'])-1},1,len(packet['terminal_coords_A'])-1),
            'observable':{'enum':sorted({r['observable'] for r in packet['one_to_rest']})},
            'evidence_id':{'enum':[r['evidence_id'] for r in packet['one_to_rest']]},
            'claim_kind':{'enum':['occupancy','coordinate_offset','axis_dispersion']},
            'proposed_advantage':text,'competing_explanation':text,'falsifier':text}),0,3)})
    analyst=caller([{'role':'system','content':analyst_skill+'\n'+shared},
        {'role':'user','content':json.dumps({'task':'Analyze ONE actual final against every other final. Select spatial hypotheses and exact registered region slots. Candidate budget does not restrict the catalogue.',
            'packet':packet},ensure_ascii=False,separators=(',',':'))}],schema,out/'Analyst.json',input_store=study/'inputs',
        max_output_tokens=4500,response_validator=lambda a:validate_analyst(a,packet),max_format_repairs=2)
    validate_analyst(analyst,packet)
    if analyst['decision']=='defer':
        record_status(study,candidate,'analyst_deferred')
        write_json(out/'status.json',{'rank':rank,'decision':'deferred','no_synthetic_hypothesis':True});return []
    # Only now is the actual intermediate point cloud made available.
    teacher=trace_teacher(repo,cfg,packet)
    transfer=[]
    for h in analyst['hypotheses']:
        ids=h['atom_slots'];final=np.asarray(packet['terminal_coords_A'])[ids];current=np.asarray(teacher['coords_A'])[ids]
        transfer.append({'hypothesis_id':h['id'],'exact_slots':ids,
            'displacement_RMS_A':float(np.sqrt(np.square(final-current).sum(1).mean())),
            'terminal_Rg_A':float(np.sqrt(np.square(final-final.mean(0)).sum(1).mean())),
            'teacher_Rg_A':float(np.sqrt(np.square(current-current.mean(0)).sum(1).mean()))})
    write_json(out/'transfer.json',{'teacher_sha256':digest(out/'teacher.json'),'rows':transfer,'not_a_causal_advantage_test':True})
    designer_skill=(root/'half-state-reward-designer/SKILL.md').read_text(encoding='utf-8')
    approved=sorted({s for h in analyst['hypotheses'] for s in h['atom_slots']})
    dschema=obj({'role':{'const':'Designer'},'candidate_id':{'const':candidate},'decision':{'enum':['activate','defer']},
        'hypothesis_ids':arr({'enum':[h['id'] for h in analyst['hypotheses']]},0,3),
        'atom_slots':arr({'type':'integer','enum':approved},0,len(approved)),
        'reward_family':{'const':'actual_half_state_pointcloud'},'teacher_time':{'const':cfg['teacher_time']},
        'dose':{'const':cfg['dose']},'control_semantics':{'const':CONTROL_SEMANTICS},
        'scientific_rationale':{'type':'string','maxLength':8192},'falsifier':text})
    def validate_design(d):
        if d['decision']=='defer':
            if d['hypothesis_ids'] or d['atom_slots']:failure('Deferred design cannot activate slots')
            return
        try:compile_programs(cfg,analyst,d,teacher,digest(out/'Analyst.json'),'pending',digest(out/'teacher.json'))
        except ValueError as error:failure(str(error))
    design=caller([{'role':'system','content':designer_skill+'\n'+shared},
        {'role':'user','content':json.dumps({'analyst':analyst,'actual_teacher':teacher,'slot_transfer':transfer,
            'frozen_dose':cfg['dose'],'guidance_starts':cfg['guidance_starts'],'guidance_end':cfg['guidance_end'],
            'required_control_semantics':CONTROL_SEMANTICS,
            'task':'Declare the exact hypothesis/slot bindings. The compiler creates paired local/full controls and all requested starts without changing your decision.'},
            ensure_ascii=False,separators=(',',':'))}],dschema,out/'Designer.json',input_store=study/'inputs',max_output_tokens=3500,
        response_validator=validate_design,max_format_repairs=2)
    validate_design(design)
    programs=compile_programs(cfg,analyst,design,teacher,digest(out/'Analyst.json'),digest(out/'Designer.json'),digest(out/'teacher.json'))
    bindings=[]
    for program in programs:
        path=out/'programs'/(program['program_id']+'.json');write_json(path,program)
        bindings.append({'program':path.relative_to(repo).as_posix(),'sha256':digest(path),'operative_text':operative_text(program)})
    write_json(out/'bindings.json',{'programs':bindings,'teacher':(out/'teacher.json').relative_to(repo).as_posix(),
        'approved_hypothesis_slots':{h['id']:h['atom_slots'] for h in analyst['hypotheses']},
        'no_decision_override':True,'teacher_was_retrieved_after_analyst_freeze':True,
        'instruction_sha256':{'Analyst':hashlib.sha256(analyst_skill.encode()).hexdigest(),'Designer':hashlib.sha256(designer_skill.encode()).hexdigest(),
            'contract':hashlib.sha256(shared.encode()).hexdigest()}})
    record_status(study,candidate,'compiled' if programs else 'designer_deferred')
    write_json(out/'status.json',{'rank':rank,'candidate_id':candidate,'compiled_programs':len(programs),
        'analyst_decision':analyst['decision'],'designer_decision':design['decision']})
    return programs


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--repo',default='.')
    p.add_argument('--action',choices=['prepare','analyze'],required=True);a=p.parse_args();cfg=read_json(a.config)
    if a.action=='prepare':print({'candidates':len(prepare(a.repo,cfg))});return
    study=Path(a.repo)/cfg['study'];cursor=read_json(study/'cursor.json')
    for _ in range(cfg['analysis_limit']):
        rank=cursor['next_rank']
        if rank>cursor['total']:break
        programs=analyze(a.repo,cfg,rank);cursor['processed'].append({'rank':rank,'programs':len(programs)});cursor['next_rank']+=1
        write_json(study/'cursor.json',cursor);print({'rank':rank,'compiled_programs':len(programs)},flush=True)


if __name__=='__main__':main()
