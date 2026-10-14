"""Sequential policy interventions, replayable decisions and compact feedback."""
import copy
import gzip
import json
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.final_guidance.workflow import operative_text
from evomolsteer.final_guidance.reward import HalfStateReward
from .mining import match
from .agents import STRATEGIES


def prepare_controls(repo,master,label,n,batch,starts=None,endpoint_mix=0.):
    repo=Path(repo);cfg=copy.deepcopy(master);cfg.update(study=f"{master['campaign_study']}/controls_{label}",
        pilot_samples=n,pilot_initial_batch=batch,include_unguided=True,remote_output=f"{master['remote_campaign']}/controls_{label}")
    base=repo/master['base_study']/'candidates'/master['candidate_id'];out=repo/cfg['study']/'candidates'/master['candidate_id']
    teacher=read_json(base/'teacher.json');write_json(out/'teacher.json',teacher);bindings=[]
    if endpoint_mix:
        packet=read_json(repo/master['campaign_study']/'evidence.json')
        teacher.update(terminal_representation='terminal_world_A',terminal_coords_A=packet['terminal_coords_A'],
            terminal_source_sha256=packet['source_sha256']['final_fields'])
        write_json(out/'teacher.json',teacher)
        cfg['endpoint_mix']=endpoint_mix
    old=read_json(base/'bindings.json')
    for item in old['programs']:
        p=read_json(repo/item['program'])
        if starts is not None and p['window'][0] not in starts:continue
        p['program_id']=f"baseline_{p['scope']}_{p['window'][0]:g}";p['atom_weights']=[1/len(p['atom_slots'])]*len(p['atom_slots'])
        if endpoint_mix:p.update(family='actual_partial_two_clock',endpoint_mix=endpoint_mix,
            teacher_sha256=digest(out/'teacher.json'),coordinate_evaluation='shared-assignment_half_bridge_and_generator_endpoint')
        path=out/'programs'/(p['program_id']+'.json');write_json(path,p)
        description=operative_text(p)
        if endpoint_mix:description+=' Additional registered actual-final transfer loss, same points and shared correspondence, endpoint mix '+str(endpoint_mix)
        bindings.append({'program':path.relative_to(repo).as_posix(),'sha256':digest(path),'operative_text':description})
    write_json(out/'bindings.json',{'programs':bindings,'teacher':(out/'teacher.json').relative_to(repo).as_posix(),
        'baseline':'Frozen prior partial selection plus whole-cloud controls, no new agent selection'})
    path=repo/'configs/experiments/partial_region_search_20261014'/f'controls_{label}.json';write_json(path,cfg)
    return cfg,path


def prepare_round(repo,master,number,state):
    repo=Path(repo);label=f'R{number:02d}';out=repo/master['campaign_study']/label;strategy,instruction=STRATEGIES[number-1]
    core=repo/master['campaign_study']/'core_skills';core.mkdir(parents=True,exist_ok=True)
    for role,key in [('analyst','analyst_skill'),('designer','designer_skill')]:
        original=repo/master[key];saved=core/role/'SKILL.md'
        if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(original.read_text(encoding='utf-8'),encoding='utf-8')
    policy=(core/'analyst/SKILL.md').read_text(encoding='utf-8')+'\n\nCurrent research intervention: '+instruction+'\n'
    if number>=19:
        best=state['best_round']
        if best!='baseline':policy=(repo/master['campaign_study']/best/'skills/analyst/SKILL.md').read_text(encoding='utf-8')
        policy+='\nThis is a frozen-selection confirmation. Do not change the supplied frozen region or point importance.\n'
    (repo/master['analyst_skill']).write_text(policy,encoding='utf-8')
    for role,key in [('analyst','analyst_skill'),('designer','designer_skill')]:
        path=out/'skills'/role/'SKILL.md';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text((repo/master[key]).read_text(encoding='utf-8'),encoding='utf-8')
    cfg=copy.deepcopy(master);cfg.update(round_id=label,study=f"{master['campaign_study']}/{label}",
        analyst_skill=f"{master['campaign_study']}/{label}/skills/analyst/SKILL.md",
        designer_skill=f"{master['campaign_study']}/{label}/skills/designer/SKILL.md",
        pilot_samples=master['screen_samples'] if number<19 else master['confirmation_samples'],
        pilot_initial_batch=0 if number<20 else 1,include_unguided=False,allow_importance=13<=number<=18,
        remote_output=f"{master['remote_campaign']}/{label}",strategy=strategy)
    cfg['quantitative_decision']=number>=3
    cfg['endpoint_mix']=.5 if number==12 else state.get('best',{}).get('endpoint_mix',0.) if number>=13 else 0.
    cfg['require_untried_region']=13<=number<=18
    if number==4:cfg['required_selected_count']=len(read_json(repo/master['campaign_study']/'evidence.json')['terminal_coords_A'])-1
    feedback={'strategy':strategy,'scientific_intervention':instruction,'best_checkpoint':state.get('best'),
        'baseline':state.get('baseline'),'history':state.get('rounds',[]),
        'previous_small_pilot':'Three-point matching improved half-state loss but not final affinity; whole-cloud from zero improved mean in a small pilot, with wide uncertainty.',
        'selection_objective':'Highest qualified final prediction; separately report top-five mean, mean of all attempts and failures',
        'target_reference_pic50':8.704806,'same_random_seed':42}
    if number>=19 or number==12:
        if state['best_round']=='baseline':
            packet=read_json(repo/master['campaign_study']/'evidence.json');ids=state['best']['atom_slots']
            region=next(r['region_id'] for r in packet['regions'] if r['atom_slots']==ids)
            frozen={'region_ids':[region],'atom_slots':ids,'point_importance':[{'slot':s,'relative_weight':1.,'evidence_id':f'point/{s}'} for s in ids]}
        else:
            a=read_json(repo/master['campaign_study']/state['best_round']/'candidates'/master['candidate_id']/'Analyst.json')
            frozen={k:a[k] for k in ['region_ids','atom_slots','point_importance']}
        cfg['frozen_selection']=frozen;cfg['allow_importance']=True
        if number>=19:cfg['guidance_starts']=[state['best']['start']]
        feedback['frozen_selection']=frozen
    else:cfg['previous_selections']=state.get('seen_selections',[])
    feedback_path=out/'feedback_input.json';write_json(feedback_path,feedback)
    cfg_path=repo/'configs/experiments/partial_region_search_20261014'/f'{label}.json';write_json(cfg_path,cfg)
    return cfg,cfg_path,feedback


def condition_metrics(folder,top_k=5,teacher=None,terminal=None):
    folder=Path(folder);rows=read_json(folder/'records.json');audit=read_json(folder/'guidance_audit.json')
    score=np.asarray([r['pic50'] for r in rows],float);qualified=np.array([r['qualified_structure'] for r in rows],bool)
    finite=np.isfinite(score);good=score[qualified&finite];top=np.sort(good)[-top_k:]
    energy=[r['mmff_relief_per_heavy'] for r in rows if r['mmff_relief_per_heavy'] is not None]
    p=audit['program'];expected=[] if p is None else p['expected_active_steps']
    if audit['actual_active_steps']!=expected or audit['actual_steps']!=100 or not audit['no_smc'] or not audit['no_affinity_head_gradient']:
        raise ValueError('Control execution does not match the fixed inference contract')
    nonzero=sum(r['nonzero_molecules']>0 for r in audit['steps'])
    if nonzero!=len(expected):raise ValueError('Declared active updates did not all deliver a dose')
    result={'condition':folder.name,'n':len(rows),'top5_qualified_mean_pic50':float(top.mean()) if len(top)==top_k else None,
        'mean_all_pic50':float(score[finite].mean()),'best_qualified_pic50':float(good.max()) if len(good) else None,
        'n_qualified':int(qualified.sum()),'n_scored':int(finite.sum()),'n_above_reference':int((good>=8.704806).sum()),
        'median_mmff_relief_per_heavy':float(np.median(energy)) if energy else None,
        'start':None if p is None else p['window'][0],'atom_slots':[] if p is None else p['atom_slots'],
        'atom_weights':[] if p is None else p.get('atom_weights',[]),'actual_updates':len(expected),
        'cumulative_control_RMS_A':audit['mean_cumulative_control_rms_A'],'initial_sha256':audit['input_pairing']['initial_sha256'],
        'source_initial_slots':audit['input_pairing']['initial_slots']}
    result['endpoint_mix']=0. if p is None else p.get('endpoint_mix',0.)
    if p is not None and teacher is not None:
        geometry=read_json(folder/'geometry.json');half=np.asarray(geometry['teacher_clock']['coords_A'])
        final=np.asarray(geometry['terminal']['coords_A']);target=np.asarray(teacher['coords_A'])[p['atom_slots']]
        result['half_selected_match_RMS_A']=float(np.mean([match(x,target)[0] for x in half]))
        result['half_full_match_RMS_A']=float(np.mean([match(x,np.asarray(teacher['coords_A']))[0] for x in half]))
        if terminal is not None:result['final_selected_match_RMS_A']=float(np.mean([match(x,np.asarray(terminal)[p['atom_slots']])[0] for x in final]))
    return result


def compact_output(generated,output):
    generated=Path(generated);output=Path(output);payload={p.relative_to(generated).as_posix():read_json(p)
        for p in sorted(generated.rglob('*.json'))}
    raw=json.dumps(payload,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
    output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(gzip.compress(raw,mtime=0))
    return {'sha256':digest(output),'raw_bytes':len(raw),'compressed_bytes':output.stat().st_size,'files':len(payload)}


def update_state(state,label,metrics,master,commit):
    eligible=[m for m in metrics if m['top5_qualified_mean_pic50'] is not None]
    if not eligible:raise ValueError('No condition has five qualified final molecules')
    metric=master.get('performance_metric','top5_qualified_mean_pic50')
    best=max(eligible,key=lambda m:(m[metric],m['top5_qualified_mean_pic50'],m['mean_all_pic50']))
    reference=state['best'][metric];delta=best[metric]-reference
    improved=delta>master['decline_tolerance_pic50'];declined=delta<-master['decline_tolerance_pic50']
    state['decline_streak']=state.get('decline_streak',0)+1 if declined else 0
    if improved:state.update(best=best,best_round=label,best_commit=commit,decline_streak=0)
    rollback=state['decline_streak']>=master['rollback_declines']
    entry={'round':label,'commit':commit,'best_this_round':best,'delta_tail_vs_best':delta,'improved':improved,
        'performance_metric':metric,
        'declined':declined,'rollback_required':rollback,'restored_round':state['best_round'] if rollback else None,'conditions':metrics}
    state['rounds'].append(entry)
    if rollback:state['decline_streak']=0
    return entry
