"""Adaptive sequential experiments; local science/API and Git-pinned inference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest,write_table
from evomolsteer.partial_search.engine import Remote,infer,local,emit
from evomolsteer.final_guidance.reward import active_updates
from evomolsteer.nc_guidance.autoresearch import NATIVE_DEFAULTS
from .evidence_registry import build
from .regional_trajectories import build as trajectories
from .agents import ask
from .semantic_contract import bind_region,validate_program,identity,numerical_code_identity
from .control_response import summarize as response
from .evaluation import original_reference,score_records,paired,local_outcomes,energy


def commit(repo,m,message):
    paths=[m['campaign_study'],m['config_dir'],'src/evomolsteer/local_motif','src/evomolsteer/final_guidance/native.py',
        'src/evomolsteer/partial_search/engine.py','scripts/run_local_motif_research.py','skills/local-motif-analyst',
        'skills/local-motif-designer','tests/test_local_motif.py','.gitattributes']
    local(repo,['git','add','--',*paths])
    if local(repo,['git','diff','--cached','--name-only']):local(repo,['git','commit','-m',message])
    local(repo,['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'])
    sha=local(repo,['git','rev-parse','HEAD']);emit('commit_pushed',commit=sha,message=message);return sha


def base_program(repo,m,evidence,templates):
    old=read_json(repo/m['initial_program']);region=next(r for r in evidence['regions'] if r['region_id']==m['reference_candidate']+'__reference')
    p={'family':'local_motif_v2','scope':'local','program_id':'REFERENCE_V2','operator':'pointwise','representation':'bridge',
        'window':old['window'],'dose':old['dose'],'steps':m['steps'],'coordinate_updates_only':True,'smc':False,'affinity_head_gradient':False,
        'native_options':copy.deepcopy(NATIVE_DEFAULTS),'controller':'unit_rms','band_multiplier':1.,'assignment_element_cost':0.,
        'orientation_band':.05,'gradient_floor_multiplier':1.}
    return validate_program(bind_region(p,region,templates[region['template_id']]),templates[region['template_id']],m['learning_window'])


def prepare(repo,m):
    study=repo/m['campaign_study'];study.mkdir(parents=True,exist_ok=True)
    e,t=build(repo,m)
    from .hypothesis_ledger import load_history
    e['history']=load_history(repo,Path(m['prior_study']))
    write_json(study/'history_index_v2.json',e['history'])
    e['trajectory']=trajectories(repo,m,e['regions'],t)
    if m.get('enable_geometric_discovery'):
        from .geometric_regions import extend
        e,extra=extend(repo,m,e,t)
        if extra:
            more=trajectories(repo,m,extra,t,'geometric_trajectory_evidence_v1.json','geometric_curves_v1.parquet')
            e['trajectory']['fits'].update(more['fits'])
    write_json(study/'shared_evidence_manifest.json',{'evidence_sha256':digest(study/'evidence.json'),
        'trajectory_sha256':digest(study/'trajectory_evidence.json'),'input_code_version':local(repo,['git','rev-parse','HEAD']),
        'corrected_full_state_clone_controls':True,'tool_names':list(e['cards'][next(iter(e['cards']))])})
    ref=study/'original_steer_reference.json'
    if not ref.exists():write_json(ref,original_reference(repo,m))
    return e,t,read_json(ref)


def bind(repo,m,p,t,label,batch=0,n=24,seed=42,resume=None,include_unguided=False):
    study=m['campaign_study']+'/'+label;folder=repo/study/'candidates'/t['candidate_id'];folder.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(p);p['program_id']=label;write_json(folder/'teacher.json',t);p['teacher_sha256']=digest(folder/'teacher.json')
    program_path=folder/'programs'/(label+'.json');write_json(program_path,p)
    write_json(folder/'bindings.json',{'teacher':(folder/'teacher.json').relative_to(repo).as_posix(),
        'programs':[{'program':program_path.relative_to(repo).as_posix(),'sha256':digest(program_path),
            'operative_text':'Analyst-bound local coordinate observable, live FLOWR VJP; no affinity derivative, graph gate or resampling.'}]})
    cfg={'study':study,'round_id':label,'remote_output':m['remote_campaign']+'/'+label,'checkpoint':m['checkpoint'],
        'remote_dataset':m['remote_dataset'],'campaign':m['campaign'],'arm':m['arm'],'pilot_initial_batch':batch,'pilot_samples':n,
        'seed':seed,'steps':m['steps'],'teacher_time':t['time'],'terminal_time':1.,'include_unguided':include_unguided,
        'save_categorical_states':True,
        'only_program_ids':[label],'native_options':p['native_options']}
    if resume:cfg['resume']=resume
    path=repo/m['config_dir']/(label+'.json');write_json(path,cfg)
    write_json(repo/study/'resolved_config.json',{'program':p,'runtime':cfg,'local_support_count':len(p['atom_slots']),
        'teacher_count':len(t['coords_A']),'actual_operator':p['operator'],'learning_window':m['learning_window']})
    return cfg,path,p


def execute(repo,m,p,t,label,remote,batch=0,n=24,seed=42,resume=None,include_unguided=False):
    cfg,path,p=bind(repo,m,p,t,label,batch,n,seed,resume,include_unguided)
    folder=repo/m['generated_root']/label
    pending=repo/cfg['study']/'pending.json'
    if (folder/'COMPLETE.json').exists():
        if not pending.exists():raise ValueError('Completed output lacks pinned dispatch identity')
        return folder/label,read_json(pending)['commit'],cfg
    sha=commit(repo,m,'experiment(local-motif-v2): '+label+' freeze executable local intervention')
    write_json(pending,{'commit':sha,'numeric_identity':identity(p,t),'config_sha256':digest(path)})
    folder=infer(repo,m,cfg,path,remote,sha)
    return folder/label,sha,cfg


def observations(repo,m,folder,p,t,label,tau):
    study=repo/m['campaign_study']/label
    return {'metrics':score_records(read_json(folder/'records.json'),tau),'response':response(folder),
        'local_outcomes':local_outcomes(folder,p,t,study/'local_outcomes.parquet'),
        'energy':energy(folder,study/'energy.json')}


def feedback(repo,state,evidence):
    best=state['best'];history=[]
    for r in state['rounds']:
        item={k:r.get(k) for k in ['round','stage','factor','status','region','operator','metrics','local_outcomes','rollback','parent']}
        if r.get('trials'):item['suffix_evidence']=[{k:trial[k] for k in ['paired','response','local_outcomes']} for trial in r['trials']]
        if r.get('stage')=='suffix':
            item.pop('metrics',None)
            item['scope']='Mechanistic same-restored-state comparisons only. Do not compare absolute suffix affinity with from-zero pools.'
        history.append(item)
    return {'parent_outcomes':best['metrics'],'parent_response':best.get('response'),'history':history,
        'objective':'Frozen tail utility primary, affinity mean guard; target best 8.704806. Formation is a mechanism, not terminal success.',
        'discovery_gaps':['Static N/C exchanges were not realized by old coordinate guidance.','O fields have explanatory but not established intervention value.'],
        'previous_rollbacks':[r['round'] for r in state['rounds'] if r.get('rollback')]}


def parents(state):
    choices=[state['best']]
    # A measured failed branch can be a diagnostic parent without promotion.
    measured=[r for r in state['rounds'] if r.get('stage')=='from_zero' and r.get('program')]
    if measured and not measured[-1].get('rollback'):choices.append(measured[-1])
    choices.append(state['reference'])
    return {r['program']['program_id']:{'program':r['program'],'metrics':r.get('metrics'),'response':r.get('response'),
        'status':'active' if r is state['best'] else 'reference_or_diagnostic'} for r in choices}


def suffix_identity(repo,m,p,t,stream):
    from evomolsteer.storage.trajectory import TrajectoryPackage
    from evomolsteer.continuous.elite_path_credit import terminal_ancestors
    from evomolsteer.nc_guidance.window_features import unique_state_controls
    step=m['suffix_step'];batch=t['batch'];base=repo/m['raw_dataset']/'results'/m['campaign']/m['arm']/f'batch_{batch:03d}'
    path=repo/m['compact_dataset']/'results'/m['campaign']/m['arm']/f'batch_{batch:03d}'/'trajectory.h5'
    with TrajectoryPackage(path) as tr:
        anc=terminal_ancestors(tr.read('selected_indices').astype(int));focal=int(anc[step,t['terminal_slot']])
        xyz=tr.read('current_coords',step);a=tr.read('current_atomics',step);controls=unique_state_controls(xyz,a,focal)
        # Different observed contexts selected before new terminal outcomes.
        distance=np.square(xyz[controls].mean(1)-xyz[focal].mean(0)).sum(1)
        ranked=controls[np.argsort(distance,kind='stable')];slots=[focal,int(ranked[0]),int(ranked[-1])]
    return {'step':step,'slots':slots,'restart_sha256':digest(base/f'restart_step_{step:03d}.pt.gz'),
        'stream':stream,'selection':'Focal historical success ancestor plus nearest/farthest unique contemporaneous centroid; neither control labeled terminal failure.'}


def controls(repo,m,p,t,label,batch,n,remote):
    if batch==0 and n in (24,50):
        root=repo/'data/generated/partial_region_search_20261014'
        return {'unguided':repo/m['generated_root']/'S0_REFERENCE_REPLAY/unguided' if n==24 else root/'controls_batch0_n50/unguided',
            'historical_R13':root/('R13/R13_0.1' if n==24 else 'R19/R19_0.1')}
    out=repo/m['generated_root']/('controls_b%d_n%d'%(batch,n))
    if not (out/'COMPLETE.json').exists():
        name=out.name;cfg,path,_=bind(repo,m,p,t,name,batch,n,include_unguided=True)
        cfg['only_program_ids']=[];write_json(path,cfg)
        sha=commit(repo,m,'control(local-motif-v2): freeze missing matched unguided pool')
        out=infer(repo,m,cfg,path,remote,sha)
    return {'unguided':out/'unguided'}


def summary(repo,m,state):
    study=repo/m['campaign_study'];rows=[]
    for r in state['rounds']:
        x={k:r.get(k) for k in ['round','stage','factor','status','region','operator','parent','rollback','commit','new_attempts']}
        x.update(r.get('metrics',{}));rows.append(x)
    write_table(study/'summary/rounds.csv',rows)
    body='|轮次|阶段/改动|状态|均值|最高合格|尾部产出U|合格/尝试|回滚|\n|---|---|---|---:|---:|---:|---|---|\n'
    for r in rows:
        body+=f"|{r['round']}|{r['stage']}/{r.get('factor')}|{r['status']}|{r.get('mean_all_pic50')}|{r.get('best_qualified_pic50')}|{r.get('U_tau')}|{r.get('n_qualified')}/{r.get('n')}|{r.get('rollback')}|\n"
    text='# 局部motif auto-research v2：运行报告\n\n'+body
    text+=f"\n当前活动最佳：{state['best']['program']['program_id']}；已完成{len(rows)}/20个研究决定。目标8.704806。所有分数为模型预测。\n"
    text+='\nS0接口/API测试与冻结控制重放另计成本；后缀诊断不能代替从头生成。确认使用冻结程序，新池结果不参与调参。\n'
    (study/'summary/report.zh-CN.md').write_text(text,encoding='utf-8')


def run(repo,master_path,stop_after=20,prepare_only=False,functional_only=False):
    repo=Path(repo).resolve();m=read_json(master_path);study=repo/m['campaign_study'];sp=study/'state.json'
    if (study/'COMPLETE.json').exists():return read_json(sp)
    e,templates,original=prepare(repo,m);tau=original['tau'];base=base_program(repo,m,e,templates)
    if prepare_only:return {'regions':len(e['regions']),'tau':tau}
    if functional_only:return functional_tests(repo,m,e,templates,base)
    if not (study/'functional_checks.json').exists():functional_tests(repo,m,e,templates,base)
    remote=Remote()
    try:
        roots=[m['checkpoint'], '/root/private_data/MolSteer/flowr_root/flowr/models/fm_pocket.py',
            '/root/private_data/MolSteer/flowr_root/flowr/models/integrator.py']
        runtime=json.loads(remote.command("python - <<'PY'\nimport json,hashlib\nfrom pathlib import Path\npaths="+repr(roots)+"\nprint(json.dumps({p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest() for p in paths}))\nPY"))
        runtime_path=study/'upstream_runtime_identity.json'
        if runtime_path.exists() and read_json(runtime_path)!=runtime:raise ValueError('Upstream model/code changed during frozen campaign')
        write_json(runtime_path,runtime)
        if sp.exists():state=read_json(sp)
        else:
            t=templates[base['template_id']]
            folder,sha,cfg=execute(repo,m,base,t,'S0_REFERENCE_REPLAY',remote,include_unguided=True)
            base['program_id']='S0_REFERENCE_REPLAY'
            obs=observations(repo,m,folder,base,t,'S0_REFERENCE_REPLAY',tau)
            old=repo/m['initial_results'];oldrows=read_json(old/'records.json');newrows=read_json(folder/'records.json')
            ugold=repo/'data/generated/partial_region_search_20261014/controls_screen/unguided';ugnew=folder.parent/'unguided'
            compatibility={'max_reference_score_change':float(max(abs(a['pic50']-b['pic50']) for a,b in zip(oldrows,newrows))),
                'max_unguided_score_change':float(max(abs(a['pic50']-b['pic50']) for a,b in zip(read_json(ugold/'records.json'),read_json(ugnew/'records.json')))),
                'meaning':'Observed native branch check; reference normalization changed implementation, report any output discrepancy.'}
            write_json(study/'S0_REFERENCE_REPLAY/compatibility.json',compatibility)
            reference={'program':base,'result_folder':folder.relative_to(repo).as_posix(),'commit':sha,**obs}
            state={'next_round':1,'rounds':[],'reference':reference,'best':reference,'declines':0,'cache':{},'original_steer':{k:v for k,v in original.items() if k!='records'},
                'S0_generation_attempts':48,'new_attempts':48,'confirmed_best':None}
            write_json(sp,state)
        for number in range(state['next_round'],min(20,stop_after)+1):
            label=f'R{number:02d}';folder=study/label;stage='suffix' if number<=2 else 'from_zero' if number<19 else 'confirmation'
            parent_options=parents(state);frozen=copy.deepcopy(state['frozen_confirmation_program']) if number>=19 and 'frozen_confirmation_program' in state else None
            if number==19 and frozen is None:
                frozen=copy.deepcopy(state['best']['program']);state['frozen_confirmation_program']=frozen;write_json(sp,state)
            if frozen:
                parent_options={frozen['program_id']:{'program':frozen,'metrics':state['best']['metrics'],'status':'frozen_confirmation'}}
            if not (folder/'proposal.json').exists():
                write_json(folder/'proposal.json',{'stage':stage,'parents':parent_options,'feedback':feedback(repo,state,e),
                    'authorized_max_rounds':20,'objective_frozen_before_results':state['original_steer']})
            if (folder/'decision.json').exists() and (folder/'selected_program.json').exists():
                saved=read_json(folder/'decision.json');a=saved['Analyst'];d=saved['Designer'];status=saved['interface_status']
                p=read_json(folder/'selected_program.json')
                emit('immutable_decision_reused',round=label)
            else:
                emit('role_calls',round=label,stage=stage,incumbent=state['best']['program']['program_id'])
                p,a,d,status=ask(repo,m,folder,e,templates,parent_options,feedback(repo,state,e),stage,frozen)
                write_json(folder/'decision.json',{'Analyst':a,'Designer':d,'interface_status':status,
                    'decision_record':'Evidence, chosen hypothesis, executable mapping and falsifiers; no private reasoning transcript.'})
            entry={'round':label,'stage':stage,'status':status['status'],'factor':a['factor'],'region':a['region_id'],
                'parent':a['parent_program_id'],'new_attempts':0,'rollback':False}
            if p:
                t=templates[p['template_id']];p['program_id']=label;entry.update(program=p,operator=p['operator'])
                write_json(folder/'selected_program.json',p)
                if stage=='suffix':
                    trials=[]
                    for stream in range(2):
                        resume=suffix_identity(repo,m,p,t,stream);name=label+'_stream'+str(stream)
                        seed=m['seed']+10000*(number+1)+stream
                        result,sha,cfg=execute(repo,m,p,t,name,remote,t['batch'],3,seed,resume,True)
                        obs=observations(repo,m,result,p,t,name,tau);comp=paired(result,result.parent/'unguided',tau)
                        trials.append({'result_folder':result.relative_to(repo).as_posix(),'commit':sha,'paired':comp,**obs})
                    all_records=[r for trial in trials for r in read_json(repo/trial['result_folder']/'records.json')]
                    entry.update(status='completed',trials=trials,metrics=score_records(all_records,tau),new_attempts=12,
                        local_outcomes=trials[0]['local_outcomes'],commit=trials[-1]['commit'])
                else:
                    n=m['screen_samples'] if number<19 else m['confirmation_samples'];batch=0 if number<20 else m['validation_batch']
                    key=hashlib.sha256(json.dumps({'program':identity(p,t),'batch':batch,'n':n,'seed':m['seed'],
                        'implementation':numerical_code_identity(p),
                        'upstream_runtime':runtime,
                        'initial_sha256':digest(repo/m['raw_dataset']/'results'/m['campaign']/m['arm']/f'batch_{batch:03d}'/'initial_state.pt.gz'),
                        'checkpoint_sha256':read_json(repo/state['reference']['result_folder']/'guidance_audit.json')['input_pairing']['checkpoint_sha256']},sort_keys=True).encode()).hexdigest()
                    reused=key in state['cache']
                    if reused:
                        result=repo/state['cache'][key];sha=read_json(result.parent.parent/'COMPLETE.json').get('commit','cached') if (result.parent.parent/'COMPLETE.json').exists() else 'cached'
                    else:
                        result,sha,cfg=execute(repo,m,p,t,label,remote,batch,n,m['seed'])
                        state['cache'][key]=result.relative_to(repo).as_posix()
                    obs=observations(repo,m,result,p,t,label,tau)
                    ctr=controls(repo,m,p,t,label,batch,n,remote)
                    comps={k:paired(result,v,tau,allow_legacy_metadata=True) for k,v in ctr.items()}
                    best=state['best'];metric=obs['metrics'];mean_delta=metric['mean_all_pic50']-best['metrics']['mean_all_pic50'];gain=metric['U_tau']-best['metrics']['U_tau']
                    improves=gain>m['numeric_tolerance'] and mean_delta>=-m['mean_regression_tolerance']
                    decline=gain < -m['numeric_tolerance'] or mean_delta < -m['mean_regression_tolerance']
                    entry.update(status='reused' if reused else 'completed',result_folder=result.relative_to(repo).as_posix(),
                        commit=sha,paired=comps,new_attempts=0 if reused else n,delta_U_vs_active=gain,delta_mean_vs_active=mean_delta,**obs)
                    if number<19:
                        if improves:state['best']={**entry,'program':p};state['declines']=0
                        else:state['declines']=state['declines']+1 if decline else 0
                        if state['declines']>=m['rollback_declines']:
                            entry['rollback']=True;state['declines']=0
                            write_json(folder/'rollback.json',{'restored_program':state['best']['program'],
                                'reason':'Two consecutive paired-pool tail or guarded-mean declines; scientific defaults restored, correctness repairs retained.'})
                    elif number==20:
                        c=comps['unguided'];state['confirmed_best']=p if c['U95'][0]>0 and c['mean95'][0]>-m['mean_regression_tolerance'] else None
            state['rounds'].append(entry);state['new_attempts']+=entry['new_attempts'];state['next_round']=number+1
            state['actual_generated_attempts']=sum((x['conditions']*x['samples_per_condition'])
                for f in (repo/m['generated_root']).glob('*/COMPLETE.json') if (x:=read_json(f)).get('samples_per_condition'))
            write_json(folder/'results.json',entry);write_json(sp,state);write_json(repo/m['config_dir']/'active_best.json',state['best']['program']);summary(repo,m,state)
            commit(repo,m,'results(local-motif-v2): '+label+' '+entry['status']+(' rollback' if entry['rollback'] else ''))
            emit('round_complete',round=label,status=entry['status'],metrics=entry.get('metrics'),rollback=entry['rollback'],new_attempts=entry['new_attempts'])
            if (study/'STOP_AFTER_CURRENT').exists():return state
        if state['next_round']>20:
            write_json(study/'COMPLETE.json',{'research_decisions':len(state['rounds']),'new_generation_attempts':state['actual_generated_attempts'],
                'active_best':state['best']['program']['program_id'],'confirmed_best':state['confirmed_best'],
                'target_reached':any(r.get('metrics',{}).get('n_above_target',0)>0 for r in state['rounds'] if r['stage']!='suffix')})
            commit(repo,m,'results(local-motif-v2): complete twenty sequential decisions')
        return state
    finally:remote.close()


def functional_tests(repo,m,e,t,base):
    study=repo/m['campaign_study'];checks=[]
    # Real API fixtures, not simulated responses. Their generation budget is zero.
    candidates=[('conditional_NC',lambda r:r['candidate_id']=='F_b004_s027' and r['region_id'].endswith('__NC')),
                ('O_negative_intervention',lambda r:r['region_id'].endswith('__O') and 'O' in r['native_elements']),
                ('normalization_saturation',lambda r:r['region_id']==base['region_id']),
                ('trajectory_provenance',lambda r:r['region_id']==base['region_id'])]
    if m.get('functional_case_subset'):candidates=[x for x in candidates if x[0] in m['functional_case_subset']]
    else:candidates=candidates[:3]
    for name,predicate in candidates:
        region=next((r for r in e['regions'] if predicate(r)),e['regions'][0]);ee=copy.deepcopy(e)
        ee['regions']=[region];ee['catalog']=[c for c in e['catalog'] if c['region_id']==region['region_id']]
        pp=copy.deepcopy(base)
        fb={'parent_response':{'actual_enabled_terms':['pointwise'],'direct_orientation_force':False,'direct_type_force':False},
            'parent_outcomes':None,'history':[],'functional_case':name,
            'task':'Validate source distinctions and evidence-to-function handoff; no generation in this fixture.'}
        if name=='trajectory_provenance':
            actual=read_json(study/'state.json')['reference'];pp=copy.deepcopy(actual['program']);fb.update(parent_response=actual['response'],parent_outcomes=actual['metrics'],
                task='Query regional_trajectories and compare its historical Steer origin against the supplied measured current-controller response. Propose one actual numerical change; do not infer current-controller motion from teacher trajectories.')
        p,a,d,info=ask(repo,m,study/('functional_v'+str(m.get('interface_revision',1)))/name,ee,t,{pp['program_id']:{'program':pp}},fb,stage='from_zero')
        checks.append({'case':name,'decision':info,'Analyst_status':a['status'],'Designer_present':d is not None,
            'selected_region':a['region_id'],'new_generation_attempts':0,'source_refs_validated':True,
            'note':'Structured source/term response check, not a causal Skills ablation or performance result.'})
    if not all(c['Designer_present'] and c['decision']['status']=='execute' for c in checks):raise ValueError('Functional fixtures did not produce consistent executable bindings; inspect capability/interface gaps before generation')
    write_json(study/'functional_checks.json',{'real_api':True,'model':'deepseek-flash','checks':checks})
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--stop-after',type=int,default=20);p.add_argument('--prepare-only',action='store_true');p.add_argument('--functional-only',action='store_true')
    a=p.parse_args();result=run(a.repo,a.config,a.stop_after,a.prepare_only,a.functional_only)
    if isinstance(result,dict) and 'rounds' in result:
        emit('campaign_return',next_round=result['next_round'],active=result['best']['program']['program_id'],attempts=result.get('actual_generated_attempts'))
    else:print(result)
