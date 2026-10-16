"""Resumable 35-condition campaign; immutable decisions, paired pools and rollbacks."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest,write_table
from evomolsteer.partial_search.engine import Remote,infer,local,emit
from evomolsteer.local_motif.evaluation import original_reference,score_records,paired,energy
from evomolsteer.local_motif.control_response import summarize as response
from .bank import prepare
from .programs import base,actions,PHASES,numerical_identity,implementation_identity
from .agents import ask


def commit(repo,m,message):
    paths=[m['campaign_study'],m['config_dir'],'src/evomolsteer/ancestral_modes','src/evomolsteer/final_guidance/native.py',
        'scripts/run_ancestral_modes_research.py','scripts/analyze_ancestral_outcomes.py','skills/ancestral-mode-analyst','skills/ancestral-mode-designer',
        'tests/test_ancestral_modes.py','.gitattributes']
    local(repo,['git','add','--',*paths])
    if local(repo,['git','diff','--cached','--name-only']):local(repo,['git','commit','-m',message])
    for i in range(4):
        try:local(repo,['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main']);break
        except RuntimeError:
            if i==3:raise
            time.sleep(2**(i+1))
    sha=local(repo,['git','rev-parse','HEAD']);emit('commit_pushed',commit=sha,message=message);return sha


def payload(bank,p):
    ids={s['template_id'] for s in p['mode_specs']}
    return {'schema_version':bank['schema_version'],'time':p['teacher_time'],'candidate_id':'ancestral_mode_bank',
        'modes':{k:bank['modes'][k] for k in ids},'limits':bank['limits']}


def bind(repo,m,p,teacher,label,batch,n,include_unguided=False,only_unguided=False):
    study=m['campaign_study']+'/'+label;folder=repo/study/'candidates'/'registered';folder.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(p);p['program_id']=label;write_json(folder/'teacher.json',teacher);p['teacher_sha256']=digest(folder/'teacher.json')
    pf=folder/'programs'/(label+'.json');write_json(pf,p)
    write_json(folder/'bindings.json',{'teacher':(folder/'teacher.json').relative_to(repo).as_posix(),
        'programs':[{'program':pf.relative_to(repo).as_posix(),'sha256':digest(pf),
            'operative_text':'Analyst-bound local ancestral/mode observable. Explicit clocks, real coordinate derivative, no affinity derivative, graph gate or SMC.'}]})
    cfg={'study':study,'round_id':label,'remote_output':m['remote_campaign']+'/'+label,'checkpoint':m['checkpoint'],
        'remote_dataset':m['remote_dataset'],'campaign':m['campaign'],'arm':m['arm'],'pilot_initial_batch':batch,'pilot_samples':n,
        'seed':m['seed'],'steps':m['steps'],'teacher_time':p['teacher_time'],'terminal_time':1.,'include_unguided':include_unguided,
        'save_categorical_states':True,'only_program_ids':[] if only_unguided else [label],'native_options':p['native_options']}
    path=repo/m['config_dir']/(label+'.json');write_json(path,cfg)
    return cfg,path,p


def execute(repo,m,p,bank,label,remote,batch=0,n=50,include_unguided=False,only_unguided=False):
    cfg,path,p=bind(repo,m,p,payload(bank,p),label,batch,n,include_unguided,only_unguided)
    target=repo/m['generated_root']/label;pending=repo/cfg['study']/'pending.json'
    if (target/'COMPLETE.json').exists():
        if not pending.exists():raise ValueError('Completed output lacks pinned dispatch')
        return target,read_json(pending)['commit']
    sha=commit(repo,m,'experiment(ancestral-v3): '+label+' freeze local intervention')
    write_json(pending,{'commit':sha,'config_sha256':digest(path),'numeric_identity':numerical_identity(p,bank)})
    return infer(repo,m,cfg,path,remote,sha),sha


def score(repo,m,folder,label,tau):
    study=repo/m['campaign_study']/label
    return {'metrics':score_records(read_json(folder/'records.json'),tau),'response':response(folder),
        'energy':energy(folder,study/'energy.json')}


def cleanup_runtime(remote,m):
    # Remove only this campaign's extracted runtime; raw structures/checkpoints are elsewhere.
    code="from pathlib import Path; import shutil; root=Path("+repr(m['remote_campaign'])+").resolve(); p=(root/'runtime').resolve(); assert p.parent==root and p.name=='runtime'; shutil.rmtree(p) if p.exists() else None"
    remote.command('python -c '+__import__('shlex').quote(code))


def smoke(repo,m,bank,p,remote):
    study=repo/m['campaign_study'];done=study/'smoke.json'
    if done.exists():return read_json(done)
    label='S00_INTERFACE_PARITY';cfg,path,q=bind(repo,m,p,payload(bank,p),label,0,m['smoke_samples'])
    old=read_json(repo/m['initial_program']);old['program_id']='R15_LEGACY_SMOKE'
    teacher=read_json((repo/m['initial_program']).parent/'candidates'/old['candidate_id']/'teacher.json')
    root=repo/cfg['study']/'candidates'/'legacy';root.mkdir(parents=True,exist_ok=True)
    write_json(root/'teacher.json',teacher);old['teacher_sha256']=digest(root/'teacher.json');pf=root/'programs/R15_LEGACY_SMOKE.json';write_json(pf,old)
    write_json(root/'bindings.json',{'teacher':(root/'teacher.json').relative_to(repo).as_posix(),
        'programs':[{'program':pf.relative_to(repo).as_posix(),'sha256':digest(pf),'operative_text':'Frozen historical R15 numerical comparator'}]})
    cfg['only_program_ids']=[label,'R15_LEGACY_SMOKE'];write_json(path,cfg)
    sha=commit(repo,m,'test(ancestral-v3): freeze live R15 interface parity')
    target=infer(repo,m,cfg,path,remote,sha)
    a=read_json(target/label/'records.json');b=read_json(target/'R15_LEGACY_SMOKE'/'records.json')
    errors=[abs(x['pic50']-y['pic50']) for x,y in zip(a,b)]
    result={'commit':sha,'n_pairs':len(a),'max_abs_final_pic50_difference':max(errors),'mean_abs_difference':float(np.mean(errors)),
        'tolerance':m['smoke_score_tolerance'],'passed':max(errors)<=m['smoke_score_tolerance'],
        'scope':'Live paired interface regression, not proof of population-wide equality or affinity improvement.'}
    actual=read_json(target/label/'guidance_audit.json')['input_pairing'];prior=read_json(repo/m['reference_results']/'guidance_audit.json')['input_pairing']
    result['reference_model_and_native_match']={k:actual[k]==prior[k] for k in ('checkpoint_sha256','upstream_native_sha256','resolved_native_options')}
    result['passed']=result['passed'] and all(result['reference_model_and_native_match'].values())
    write_json(done,result);cleanup_runtime(remote,m)
    if not result['passed']:raise ValueError('Interface parity failed; inspect before research')
    return result


def control(repo,m,p,bank,batch,n,remote):
    label=f'UNGUIDED_b{batch:02d}_n{n}'
    folder=repo/m['generated_root']/label/'unguided'
    if not (folder/'COMPLETE.json').exists():execute(repo,m,p,bank,label,remote,batch,n,True,True)
    return folder


def write_report(repo,m,state):
    root=repo/m['campaign_study'];rows=[]
    for r in state['rounds']:
        row={k:r.get(k) for k in ('round','phase','stage','parent','status','rollback','commit','new_attempts')}
        row.update(r.get('metrics',{}));rows.append(row)
    write_table(root/'summary/conditions.csv',rows)
    text='# 祖先模式 auto-research v3：顺序实验报告\n\n'
    text+=f"已完成 {len(rows)}/{m['max_rounds']} 个条件；研究最佳 {state['best']['program']['program_id']}。原 Steer 最优 {state['original']['best']:.6f}。分数均为模型预测。\n\n"
    text+='|轮次|机制|均值|最高合格|Uτ|合格/尝试|回滚|\n|---|---|---:|---:|---:|---|---|\n'
    for r in rows:text+=f"|{r['round']}|{r['phase']}|{r.get('mean_all_pic50')}|{r.get('best_qualified_pic50')}|{r.get('U_tau')}|{r.get('n_qualified')}/{r.get('n')}|{r['rollback']}|\n"
    text+='\n开发池反复使用，区间未调整搜索偏差；最终两个新引导终态池完全冻结。历史 Steer 有谱系依赖且预算不同。未形成/未保持的特征不能解释为成功机制。\n'
    (root/'report.zh-CN.md').write_bytes(text.encode('utf-8'))


def functional(repo,m,bank,regions,state):
    folder=repo/m['campaign_study']/('functional_v'+str(m['functional_revision']));path=folder/'COMPLETE.json'
    if path.exists():return read_json(path)
    checks=[]
    for number,name,modules in [(1,'current_clock',[]),(3,'lookahead',[]),(16,'interface_roles_on',['C1']),(16,'interface_roles_off',[])]:
        choices=actions(repo,m,bank,regions,state,number)
        if not choices:raise ValueError('No functional actions for '+name)
        p,a,d=ask(repo,m,folder/name,bank,regions,state,choices,PHASES[number],modules)
        checks.append({'case':name,'real_api':True,'active_modules':d['active_modules'],'numeric_identity':numerical_identity(p,bank),
            'bound_regions':a['region_ids'],'delivery':d,'expected_response_realized':p['clock_policy']=='current' if number==1 else p['clock_policy']=='lookahead' if number==3 else True})
    result={'real_api':True,'model':m['api_model'],'checks':checks,'no_generation':True,
        'chemical_prompt_changed_numerical_choice':checks[2]['numeric_identity']!=checks[3]['numeric_identity'],
        'limit':'Delivery and executable response test; on/off single calls do not establish causal performance benefit.'}
    if not all(x['expected_response_realized'] for x in checks):raise ValueError('Functional clock response failed')
    write_json(path,result);return result


def suffix_probe(repo,m,bank,p,remote):
    study=repo/m['campaign_study'];done=study/'suffix_probe.json'
    if done.exists():return read_json(done)
    from evomolsteer.storage.trajectory import TrajectoryPackage
    from evomolsteer.continuous.elite_path_credit import terminal_ancestors
    from evomolsteer.local_motif.state_controls import full_state_controls
    mode=bank['modes'][p['mode_specs'][0]['template_id']];batch=mode['batch'];step=m['suffix_step']
    source=repo/m['compact_dataset']/'results'/m['campaign']/m['arm']/f'batch_{batch:03d}'/'trajectory.h5'
    with TrajectoryPackage(source) as tr:
        anc=terminal_ancestors(tr.read('selected_indices').astype(int));focal=int(anc[step,mode['terminal_slot']])
        x=tr.read('current_coords',step);ids=full_state_controls(tr,step,x,tr.read('current_atomics',step),focal).tolist()
        chosen=[focal]+ids[:7];unique_count=len(chosen)
        while len(chosen)<8:chosen.append(chosen[len(chosen)%unique_count])
    restart=repo/m['raw_dataset']/'results'/m['campaign']/m['arm']/f'batch_{batch:03d}'/f'restart_step_{step:03d}.pt.gz'
    q=copy.deepcopy(p);q['clock_policy']='current';label='S01_NATIVE_SUFFIX';cfg,path,q=bind(repo,m,q,payload(bank,q),label,batch,8,True)
    cfg['resume']={'step':step,'slots':chosen,'restart_sha256':digest(restart),'stream':0,
        'selection':'Historical success ancestor and unique measured contemporaries; no unobserved terminal failure labels'}
    write_json(path,cfg);sha=commit(repo,m,'test(ancestral-v3): freeze actual-state native suffix with no SMC')
    result=infer(repo,m,cfg,path,remote,sha)
    comparison=paired(result/label,result/'unguided',original_reference(repo,m)['tau'])
    write_json(done,{'commit':sha,'comparison':comparison,'n_pairs':8,'unique_complete_starts':unique_count,
        'scope':'Restored full-state mechanistic diagnostic using new paired suffix random streams, not a from-zero benefit or historical RNG replay.'})
    cleanup_runtime(remote,m);return read_json(done)


def run(repo,path,stop_after=35,prepare_only=False,functional_only=False,smoke_only=False):
    repo=Path(repo).resolve();m=read_json(path);study=repo/m['campaign_study'];sp=study/'state.json'
    if (study/'COMPLETE.json').exists():return read_json(sp)
    bank,regions=prepare(repo,m);p=base(repo,m,bank,regions)
    if not sp.exists():
        original=original_reference(repo,m);ref=repo/m['reference_results'];tau=original['tau']
        obs=score(repo,m,ref,'REFERENCE_REUSED',tau)
        state={'next_round':1,'rounds':[],'best':{'program':p,'result_folder':ref.relative_to(repo).as_posix(),**obs},
            'reference':{'program':copy.deepcopy(p),'result_folder':ref.relative_to(repo).as_posix(),**copy.deepcopy(obs)},
            'original':{k:v for k,v in original.items() if k!='records'},'declines':0,'cache':{},'confirmed':False,
            'dose_reference_steps':read_json(repo/m['dose_reference_results']/'guidance_audit.json')['steps']}
        prior=repo/m['prior_study']/'scientific_readout.zh-CN.md'
        state['prior_campaign_evidence']={'source_sha256':digest(prior),'scientific_readout':prior.read_text(encoding='utf-8'),
            'scope':'Previous heterogeneous intervention evidence, not independent replications or current-controller measurements.'}
        write_json(sp,state)
    else:state=read_json(sp)
    if prepare_only:return {'modes':len(bank['modes']),'regions':len(regions),'next_round':state['next_round']}
    functional(repo,m,bank,regions,state)
    if functional_only:return {'functional':'passed'}
    remote=Remote()
    try:
        smoke(repo,m,bank,p,remote)
        if smoke_only:return {'smoke':'passed'}
        suffix_probe(repo,m,bank,p,remote)
        for number in range(state['next_round'],min(m['max_rounds'],stop_after)+1):
            label=f'R{number:02d}_v3';folder=study/label;phase=PHASES[number];stage='confirmation' if number>=34 else 'discovery'
            if number==34 and 'frozen' not in state:state['frozen']=copy.deepcopy(state['best']['program']);write_json(sp,state)
            if stage=='confirmation':q=copy.deepcopy(state['frozen']);parent=state['best']['program']['program_id'];delivery={'frozen':True,'no_redesign':True}
            elif (folder/'selected_program.json').exists():
                q=read_json(folder/'selected_program.json');d=read_json(folder/'decision.json');parent=d['action']['parent_id'];delivery=read_json(folder/'delivery.json')
            else:
                choices=actions(repo,m,bank,regions,state,number)
                if not choices:
                    choices=actions(repo,m,bank,regions,state,31);phase=phase+'_unsupported_then_adaptive'
                if not choices:raise ValueError('No distinct legal scientific actions remain')
                write_json(study/'regions.json',regions)
                if number==33:
                    # One matched deterministic selection opportunity, not an equal-budget full search attribution.
                    action=sorted(choices,key=lambda a:(len(a['program']['mode_specs'][0]['atom_slots']),a['action_id']))[0]
                    q=copy.deepcopy(action['program']);parent=action['parent_id'];delivery={'deterministic':True,'rule':'smallest support then stable action ID','full_search_agent_causality_established':False}
                    write_json(folder/'decision.json',{'action':{k:v for k,v in action.items() if k!='program'},'selector':delivery})
                    write_json(folder/'selected_program.json',q);write_json(folder/'delivery.json',delivery)
                else:q,action,delivery=ask(repo,m,folder,bank,regions,state,choices,phase.split('_unsupported')[0]);parent=action['parent_id']
            q['program_id']=label;nid=numerical_identity(q,bank);write_json(folder/'selected_program.json',q)
            pools=[0] if stage=='discovery' else m['confirmation_pools'][number-34]
            outputs=[];allrows=[];paired_summaries=[];commits=[];new=0
            for batch in pools:
                name=label if stage=='discovery' else label+f'_b{batch:02d}'
                key=json.dumps([nid,batch,m['screen_samples'],m['seed'],implementation_identity()])
                if key in state['cache']:result=repo/state['cache'][key];sha='reused'
                else:
                    target,sha=execute(repo,m,q,bank,name,remote,batch,m['screen_samples']);result=target/name;new+=m['screen_samples']
                    state['cache'][key]=result.relative_to(repo).as_posix();write_json(sp,state)
                ug=control(repo,m,q,bank,batch,m['screen_samples'],remote)
                paired_summaries.append(paired(result,ug,state['original']['tau']))
                allrows+=read_json(result/'records.json');outputs.append(result);commits.append(sha)
                cleanup_runtime(remote,m)
            metrics=score_records(allrows,state['original']['tau']);energies=[energy(o,folder/(o.parent.name+'_energy.json')) for o in outputs]
            from .outcomes import analyze
            outcomes=[analyze(o,bank,folder/('outcomes_'+o.parent.name)) for o in outputs]
            obs_response=response(outputs[0]);best=state['best'];gain=metrics['U_tau']-best['metrics']['U_tau'];mean=metrics['mean_all_pic50']-best['metrics']['mean_all_pic50']
            improves=gain>m['numeric_tolerance'] and mean>=-m['mean_regression_tolerance'];declines=gain < -m['numeric_tolerance'] or mean < -m['mean_regression_tolerance']
            entry={'round':label,'stage':stage,'phase':phase,'parent':parent,'program':q,'numeric_identity':nid,
                'metrics':metrics,'response':obs_response,'energy_summaries':[{k:v for k,v in e.items() if k!='rows'} for e in energies],
                'local_outcomes':[{k:v for k,v in o.items() if k not in ('full_window_error_fits','sources')} for o in outcomes],
                'paired_unguided':paired_summaries,'result_folders':[o.relative_to(repo).as_posix() for o in outputs],
                'result_folder':outputs[0].relative_to(repo).as_posix(),'status':'completed' if new else 'reused',
                'new_attempts':new,'commit':commits[-1],'rollback':False,'delta_U_vs_incumbent':gain,'delta_mean_vs_incumbent':mean,'delivery':delivery}
            if stage=='discovery':
                previous_tail=state.get('tail_candidate',state['reference'])
                if (metrics['best_qualified_pic50'] or -1)>(previous_tail['metrics']['best_qualified_pic50'] or -1):state['tail_candidate']=copy.deepcopy(entry)
                if improves:state['best']=copy.deepcopy(entry);state['declines']=0
                else:state['declines']=state['declines']+1 if declines else 0
                if state['declines']>=m['rollback_declines']:
                    entry['rollback']=True;state['declines']=0
                    write_json(folder/'rollback.json',{'restored_program':state['best']['program'],'reason':'Two consecutive tail/guarded-mean declines; restore incumbent science, preserve correctness fixes.'})
            else:
                # Pool aggregate uses actual paired outcomes, no pooling of bootstrap intervals.
                ugrows=[]
                for batch in pools:ugrows+=read_json(repo/m['generated_root']/f'UNGUIDED_b{batch:02d}_n{m["screen_samples"]}'/'unguided/records.json')
                rng=np.random.default_rng(m['seed']);ax=np.array([r['pic50'] for r in allrows]);bx=np.array([r['pic50'] for r in ugrows])
                qa=np.array([r['qualified_structure'] for r in allrows]);qb=np.array([r['qualified_structure'] for r in ugrows]);tau=state['original']['tau']
                du=np.where(qa,np.maximum(ax-tau,0),0)-np.where(qb,np.maximum(bx-tau,0),0)
                # Independent generation batches are bootstrap units; only four means, hence broad limits.
                dm=(ax-bx).reshape(len(pools),-1).mean(1);dum=du.reshape(len(pools),-1).mean(1);ids=rng.integers(0,len(pools),(5000,len(pools)))
                interval=np.quantile(dm[ids].mean(1),[.025,.975]);ui=np.quantile(dum[ids].mean(1),[.025,.975])
                entry['confirmation']={'batch_mean95':interval.tolist(),'batch_U95':ui.tolist(),'mean_difference':float(dm.mean()),'U_difference':float(dum.mean()),
                    'passes':bool(interval[0]>-m['mean_regression_tolerance'] and ui[0]>0),'scope':'New guided-terminal pools within historical teacher-prior knowledge, not target/prior holdout.'}
            state['rounds'].append(entry);state['next_round']=number+1;write_json(folder/'results.json',entry);write_json(sp,state)
            write_json(repo/m['config_dir']/'research_best.json',state['best']['program']);write_report(repo,m,state)
            commit(repo,m,'results(ancestral-v3): '+label+' '+phase+(' rollback' if entry['rollback'] else ''))
            emit('round_complete',round=label,phase=phase,metrics=metrics,rollback=entry['rollback'],new_attempts=new)
            if (study/'STOP_AFTER_CURRENT').exists():return state
        if state['next_round']>m['max_rounds']:
            state['confirmed']=all(r.get('confirmation',{}).get('passes',False) for r in state['rounds'][-2:])
            completed={'conditions':len(state['rounds']),'research_best':state['best']['program']['program_id'],'confirmed':state['confirmed'],
                'target_reached':any(r['metrics']['n_above_target'] for r in state['rounds']),'original_steer_best':state['original']['best']}
            write_json(sp,state);write_json(study/'COMPLETE.json',completed)
            write_json(study/'execution_recommendation.json',{'historical_baseline_changed':False,'research_candidate':state['best']['program'],
                'confirmed_against_native':state['confirmed'],'note':'Target/quality/compute comparisons remain separately reported.'})
            commit(repo,m,'results(ancestral-v3): complete thirty-five sequential conditions')
        return state
    finally:remote.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--stop-after',type=int,default=35)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--functional-only',action='store_true');p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();state=run(a.repo,a.config,a.stop_after,a.prepare_only,a.functional_only,a.smoke_only)
    emit('campaign_return',next_round=state.get('next_round'),status=state.get('functional',state.get('smoke','returned')))
