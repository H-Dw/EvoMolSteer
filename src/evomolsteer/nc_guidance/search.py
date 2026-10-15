"""Sequential 20-round local N/C experiments; preserve R13 and all raw decisions."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.partial_search.engine import Remote,infer,local,emit
from evomolsteer.partial_search.campaign import compact_output,condition_metrics
from evomolsteer.final_guidance.reward import active_updates
from .workflow import analyze,numerical_signature


PLANS=[
('local_geometry','Use a local N/C-evidence patch but geometry alone, to measure the effect of replacing the broad R13 cloud.'),
('add_decoder_NC','Freeze the first local patch; add a modest decoder N/C target via coordinate gradients.'),
('stronger_decoder_NC','Freeze the same patch; test a larger type-gradient share, not more points.'),
('r13_counterfactual','Use measured best-legal N/C placement on the R13 teacher, with local final geometry transfer.'),
('quality_native','Test the highest-scoring qualified native teacher with candidate-specific N/C evidence.'),
('directional_anchor','Add supported directional donor geometry to a local N/C proposal; never infer hydrogen-bond energy.'),
('carbon_counterexamples','Constrain at least two measured C-preferred sites along with N, instead of rewarding N everywhere.'),
('disconnected_N_anchors','Test multiple favorable N sites in a possibly disconnected region, retaining only required local context.'),
('conditional_CN','Hold the best local patch and compare C/N-conditional decoder odds with full-vocabulary likelihood.'),
('site_contrast_weighting','Hold the patch; test bounded relative site importance from finite layout contrast magnitude.'),
('r13_context_calibration','Restore the historical R13 proper-subset geometry without any type term; verify its native reproduction before interpreting added N/C terms.'),
('r13_context_native','Restore the exact historical R13 proper-subset geometry; apply the original local N/C labels on only the two compared sites.'),
('r13_context_joint_swap','Keep the same R13 geometry and dose; swap only the measured N/C targets at the two compared sites.'),
('r13_context_stronger_NC','Keep R13 geometry and the paired swapped sites; increase the local type-gradient share.'),
('r13_context_chemical_role','Keep the two-site swap and R13 context; add a selected-N neutral/aromatic decoder preference.'),
('later_start','Freeze the R13 context and paired N/C swap from the controlled comparison; test a later start and complete the native suffix.'),
('earlier_start','Freeze the R13 context and paired N/C swap from the controlled comparison; test guidance from the initial state.'),
('local_categorical_gradient','Keep R13 geometry and the paired local swap; additionally ascend the type reward only on the two selected decoder logits before native categorical flow. No affinity gradient or discrete override.'),
('confirmation','Freeze the training winner on more matched inputs; do not reinterpret confirmation as selection.'),
('heldout_initial_pool','Freeze the training winner on a new initial pool; do not tune using its outcome.')]


def commit(repo,cfg,message):
    local(repo,['git','add','--',cfg['campaign_study'],cfg['config_dir'],'skills/local-nc-analyst','skills/local-nc-designer'])
    if local(repo,['git','diff','--cached','--name-only']):local(repo,['git','commit','-m',message])
    sha=local(repo,['git','rev-parse','HEAD'])
    for attempt in range(3):
        try:local(repo,['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main']);break
        except RuntimeError:
            if attempt==2:raise
    emit('commit_pushed',commit=sha,message=message);return sha


def default_contract(master):
    return {'window':[.1,master['guidance_end']],'weights':{'half':.3,'endpoint':.4,'type':.3,'orientation':0.},
        'dose':copy.deepcopy(master['dose']),'type_mode':'full_ce','gradient_balance':'unit_rms','assignment_endpoint_fraction':1.,
        'assignment_type_cost':0.,'type_ramp':False,'anchor_distance_A':3.,'anchor_width_A':.4}


def prepare_round(repo,master,number,state):
    label=f'R{number:02d}';strategy,instruction=PLANS[number-1];study=repo/master['campaign_study'];path=repo/master['config_dir']/(label+'.json')
    if path.exists():return read_json(path),path
    cfg=copy.deepcopy(master);cfg.update(round_id=label,study=master['campaign_study']+'/'+label,strategy=strategy,intervention=instruction,
        remote_output=master['remote_campaign']+'/'+label,pilot_samples=master['screen_samples'] if number<19 else master['confirmation_samples'],
        pilot_initial_batch=0 if number<20 else master['validation_initial_batch'],include_unguided=False,only_program_ids=[label])
    base=copy.deepcopy(state.get('best_typed_contract') or state.get('best_nc_contract') or default_contract(master))
    cfg['contrast_weighting']=state.get('best_typed_contrast_weighting',False)
    cfg['execution_contract']=base
    if number==1:cfg.update(candidate_node=master['r13_node_id'],template_kind='native');base.update(weights={'half':1.,'endpoint':0.,'type':0.,'orientation':0.},assignment_endpoint_fraction=0.)
    if number in (2,3):
        cfg['frozen_patch']=state['rounds'][0]['patch_id'];cfg['execution_contract']=copy.deepcopy(state['rounds'][0]['contract']);base=cfg['execution_contract']
        alpha=.15 if number==2 else .4;base['weights']={'half':1-alpha,'endpoint':0.,'type':alpha,'orientation':0.}
    if number==4:cfg.update(candidate_node=master['r13_node_id'],template_kind='counterfactual',require_compensation=True);cfg['execution_contract']=default_contract(master)
    if number==5:
        packet=read_json(study/'evidence.json');candidates=[c for c in packet['candidate_summaries'] if c['node_id']!=master['r13_node_id']]
        cfg.update(candidate_node=max(candidates,key=lambda c:c['native_score'])['node_id'],template_kind='native');cfg['execution_contract']=default_contract(master)
    if number>=6 and state.get('best_nc_patch'):cfg['frozen_patch']=state['best_nc_patch']
    if number==6:
        patch=next(p for p in read_json(study/'evidence.json')['patches'] if p['patch_id']==cfg.get('frozen_patch')) if cfg.get('frozen_patch') else None
        if patch is None or not patch['anchors']:cfg.pop('frozen_patch',None)
        base['weights']['orientation']=.15
    if number in (7,8,15):cfg.pop('frozen_patch',None)
    if number==7:cfg['minimum_C']=2
    if number==8:cfg['minimum_N']=2
    if number==9:base['type_mode']='cn_conditional'
    if number==10:cfg['contrast_weighting']=True
    if number==11:base['assignment_type_cost']=.2
    if 11<=number<=18:
        cfg.pop('frozen_patch',None);cfg.pop('exclude_patches',None)
        cfg.update(r13_context_only=True,template_kind='native' if number<=12 else 'counterfactual',
            candidate_node=master['r13_node_id'],max_local_points=20,contrast_weighting=False)
        cfg['execution_contract']=default_contract(master);base=cfg['execution_contract']
        alpha=0. if number==11 else .4 if number==14 else .15
        base.update(weights={'half':1-alpha,'endpoint':0.,'type':alpha,'orientation':0.},assignment_endpoint_fraction=0.)
        if number==15:base['weights']['role']=.15
    if number==16:base['window'][0]=.3
    if number==17:base['window'][0]=0.
    if number==18:base.update(atomic_logit_strength=8.,logit_max_shift=4.)
    if number>=19:
        if state['best_round']=='R13_reference':
            # No local proposal improved R13: confirm the strongest local candidate explicitly, without promoting it.
            cfg['confirmation_of_nonwinning_local']=True
        cfg['frozen_patch']=state['best_nc_patch'];cfg['execution_contract']=copy.deepcopy(state['best_nc_contract'])
        cfg['contrast_weighting']=state.get('best_nc_contrast_weighting',False)
        bestpatch=next(p for p in read_json(study/'evidence.json')['patches'] if p['patch_id']==cfg['frozen_patch'])
        cfg['allow_r13_context']=bool(bestpatch.get('r13_context'))
    core=study/('core_skills' if master.get('skill_revision',1)==1 else f"core_skills_v{master['skill_revision']}")
    for role,key in [('analyst','analyst_skill'),('designer','designer_skill')]:
        saved=core/role/'SKILL.md'
        if not saved.exists():saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes((repo/master[key]).read_bytes())
        text=saved.read_text(encoding='utf-8')+('\nCurrent research intervention: '+instruction+'\n')
        dest=study/label/'skills'/role/'SKILL.md';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(text.encode())
        cfg[key]=dest.relative_to(repo).as_posix()
    write_json(path,cfg);write_json(study/label/'feedback.json',{'objective':'Max qualified final pIC50; report means, failures and local N/C response',
        'incumbent':state['best'],'history':state['rounds'],'intervention':instruction,'target_pic50':8.704806})
    return cfg,path


def atomic_indices(folder,geometry,vocab):
    terminal=geometry['terminal']
    if 'atomic_indices' in terminal:return np.asarray(terminal['atomic_indices'],int)
    import gzip
    from rdkit import Chem
    xyz=np.asarray(terminal['coords_A']);result=np.full(xyz.shape[:2],-1,int)
    for i in range(len(xyz)):
        path=folder/f'{i:03d}.sdf.gz'
        if not path.exists():continue
        with gzip.open(path,'rt') as f:mol=Chem.MolFromMolBlock(f.read().split('$$$$')[0],sanitize=False,removeHs=False)
        if mol is None or mol.GetNumAtoms()!=xyz.shape[1]:continue
        coords=mol.GetConformer().GetPositions();cost=np.square(xyz[i,:,None]-coords[None]).sum(-1);rr,cc=linear_sum_assignment(cost)
        if np.sqrt(cost[rr,cc].max())>.01:raise ValueError('SDF has changed coordinates; cannot infer original slot types')
        result[i,rr]=[vocab.get(mol.GetAtomWithIdx(int(j)).GetSymbol(),-1) for j in cc]
    return result


def metrics(folder,master,program,teacher,repo=Path('.')):
    # Generic final metrics also verify the actual control window, nonzero steps and native suffix.
    result=condition_metrics(folder);geometry=read_json(folder/'geometry.json');xyz=np.asarray(geometry['terminal']['coords_A'])
    ids=atomic_indices(folder,geometry,teacher['atom_vocabulary']);target=np.asarray(teacher['terminal_coords_A'])[program['atom_slots']]
    types=[];distances=[];n_roles=[]
    local_map={s:i for i,s in enumerate(program['atom_slots'])};ti=[local_map[s] for s in program['typed_slots']]
    desired=np.asarray([teacher['atom_vocabulary'][e] for e in program['typed_elements']])
    for sample,(x,a) in enumerate(zip(xyz,ids)):
        cost=np.square(x[:,None]-target[None]).sum(-1);rr,cc=linear_sum_assignment(cost);assignment=rr[np.argsort(cc)]
        distances.append(float(np.sqrt(cost[assignment,np.arange(len(target))].mean())))
        if (a>=0).all():types.append(float((a[assignment[ti]]==desired).mean()))
        import gzip
        from rdkit import Chem
        sdf=folder/f'{sample:03d}.sdf.gz'
        if sdf.exists():
            with gzip.open(sdf,'rt') as stream:mol=Chem.MolFromMolBlock(stream.read().split('$$$$')[0],sanitize=True,removeHs=False)
            if mol is not None and mol.GetNumAtoms()==len(x):
                coords=mol.GetConformer().GetPositions();rr,cc=linear_sum_assignment(np.square(x[:,None]-coords[None]).sum(-1))
                original_to_sdf=dict(zip(rr,cc));flags=[]
                for local,element in zip(ti,program['typed_elements']):
                    if element!='N':continue
                    atom=mol.GetAtomWithIdx(int(original_to_sdf[assignment[local]]))
                    flags.append(atom.GetSymbol()=='N' and atom.GetIsAromatic() and atom.GetFormalCharge()==0 and atom.GetDegree()==2 and atom.GetTotalNumHs()==0)
                if flags:n_roles.append(float(np.mean(flags)))
    audit=read_json(folder/'guidance_audit.json');rows=audit['steps']
    if audit['program'] and audit['program']['family']=='local_nc_layout':
        templates=read_json(Path(repo)/master['campaign_study']/'templates.json')
        actual_teacher=templates[audit['program']['template_id']]
        if numerical_signature(audit['program'],actual_teacher)!=numerical_signature(program,teacher):raise ValueError('Executed numeric decision differs')
        if max(r.get('direct_endpoint_support_count',0) for r in rows)>program['max_local_points']:raise ValueError('Support exceeds declared local cap')
        if max(r.get('direct_unmatched_gradient_max',0) for r in rows)!=0:raise ValueError('Direct loss escaped selected local support')
    result.update(local_NC_identity_accuracy=float(np.mean(types)) if types else None,identity_coverage=len(types),
        aromatic_N_role_fraction=float(np.mean(n_roles)) if n_roles else None,chemical_role_coverage=len(n_roles),
        final_local_geometry_rms_A=float(np.mean(distances)),max_type_gradient_rms=max(r.get('type_gradient_rms',0.) for r in rows),
        max_type_mixture_input_rms=max(r.get('type_mixture_input_rms',0.) for r in rows))
    return result


def baseline_controls(repo,master,batch,n,remote):
    if batch==0:
        if n==24:return {'R13_reference':repo/'data/generated/partial_region_search_20261014/R13/R13_0.1',
            'unguided':repo/'data/generated/partial_region_search_20261014/controls_screen/unguided'}
        return {'R13_reference':repo/'data/generated/partial_region_search_20261014/R19/R19_0.1',
            'unguided':repo/'data/generated/partial_region_search_20261014/controls_batch0_n50/unguided'}
    label=f'controls_b{batch}_n{n}';path=repo/master['config_dir']/(label+'.json');cfg=copy.deepcopy(master)
    cfg.update(study=master['campaign_study']+'/'+label,pilot_initial_batch=batch,pilot_samples=n,include_unguided=True,
        only_program_ids=['R13_0.1'],remote_output=master['remote_campaign']+'/'+label)
    old=repo/'docs/experiments/partial_region_search_20261014/R13/candidates/F_b011_s030';dst=repo/cfg['study']/'candidates/F_b011_s030'
    for name in ('teacher.json','Analyst.json','Designer.json'):
        dest=dst/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((old/name).read_bytes())
    b=read_json(old/'bindings.json');entry=next(e for e in b['programs'] if e['program'].endswith('/R13_0.1.json'))
    p=read_json(repo/entry['program']);dest=dst/'programs/R13_0.1.json';write_json(dest,p)
    write_json(dst/'bindings.json',{'programs':[{**entry,'program':dest.relative_to(repo).as_posix(),'sha256':digest(dest)}],
        'teacher':(dst/'teacher.json').relative_to(repo).as_posix()})
    write_json(path,cfg);sha=commit(repo,master,'experiment(nc-layout): freeze heldout R13 and unguided controls')
    local_folder=repo/master['generated_root']/label
    if not (local_folder/'COMPLETE.json').exists():local_folder=infer(repo,master,cfg,path,remote,sha)
    return {'R13_reference':local_folder/'R13_0.1','unguided':local_folder/'unguided'}


def run(repo,config,stop_after=20):
    repo=Path(repo).resolve();master=read_json(config);study=repo/master['campaign_study'];sp=study/'state.json'
    if (study/'COMPLETE.json').exists():return read_json(sp)
    state=read_json(sp) if sp.exists() else None;remote=Remote()
    try:
        if state is None:
            baseline=baseline_controls(repo,master,0,master['screen_samples'],remote)
            state={'next_round':1,'best_round':'R13_reference','best':condition_metrics(baseline['R13_reference']),
                'rounds':[],'cache':{},'decline_streak':0,'best_nc_score':-1e30,'rollback_commits':[]}
            write_json(sp,state)
        # Reindex any older metadata-sensitive keys without regenerating numeric duplicates.
        templates=read_json(study/'templates.json');cache={}
        for source in state['cache'].values():
            folder=repo/source;a=read_json(folder/'guidance_audit.json');p=a['program']
            batch=next((r.get('initial_batch',0) for r in state['rounds'] if r['round']==folder.name),0)
            key=f"{batch}:{len(read_json(folder/'records.json'))}:"+numerical_signature(p,templates[p['template_id']]);cache[key]=source
        state['cache']=cache
        for number in range(state['next_round'],min(stop_after,master['max_rounds'])+1):
            cfg,path=prepare_round(repo,master,number,state);label=cfg['round_id'];folder=repo/cfg['study'];feedback=read_json(folder/'feedback.json')
            emit('real_agent_call',round=label,strategy=cfg['strategy'])
            if (folder/'Designer.json').exists() and read_json(folder/'Designer.json')['decision']=='defer':p=None
            elif (folder/'selection.json').exists() and list((folder/'candidates').glob('*/bindings.json')):
                bp=next((folder/'candidates').glob('*/bindings.json'));b=read_json(bp);p=read_json(repo/b['programs'][0]['program'])
            else:p=analyze(repo,cfg,feedback)
            if p is None:
                patch=read_json(folder/'selection.json')
                entry={'round':label,'strategy':cfg['strategy'],'patch_id':patch['patch_id'],'status':'designer_deferred',
                    'contract':cfg['execution_contract'],'metrics':None,'confirmation':False,'rollback':False,
                    'additional_generation_attempts':0,'incumbent_after':state['best_round']}
                state['rounds'].append(entry);state['next_round']=number+1;write_json(folder/'results.json',entry);write_json(sp,state)
                commit(repo,master,f'results(nc-layout): {label} honor Designer defer; no inference')
                emit('round_deferred',round=label,patch=patch['patch_id']);continue
            bp=next((folder/'candidates').glob('*/bindings.json'));teacher=read_json(repo/read_json(bp)['teacher'])
            key=f"{cfg['pilot_initial_batch']}:{cfg['pilot_samples']}:"+numerical_signature(p,teacher)
            sha=commit(repo,master,f"experiment(nc-layout): {label} {cfg['strategy']} local Analyst-bound reward")
            if key in state['cache']:result_folder=repo/state['cache'][key];reused=True
            else:
                generated=repo/master['generated_root']/label
                if (generated/'COMPLETE.json').exists():
                    recovery=read_json(folder/'recovery.json') if (folder/'recovery.json').exists() else read_json(folder/'pending_inference.json')
                    sha=recovery['inference_commit'];emit('completed_inference_recovered',round=label,commit=sha)
                else:
                    write_json(folder/'pending_inference.json',{'inference_commit':sha,'program_sha256':digest(repo/read_json(bp)['programs'][0]['program'])})
                    generated=infer(repo,master,cfg,path,remote,sha)
                result_folder=generated/label;reused=False
                compact_output(generated,folder/'transport.json.gz');state['cache'][key]=result_folder.relative_to(repo).as_posix()
            m=metrics(result_folder,master,p,teacher,repo);controls=baseline_controls(repo,master,cfg['pilot_initial_batch'],cfg['pilot_samples'],remote)
            control_scores={name:condition_metrics(f) for name,f in controls.items()}
            matched={name:metrics(f,master,p,teacher,repo) for name,f in controls.items()}
            if any(v['initial_sha256']!=m['initial_sha256'] or v['source_initial_slots']!=m['source_initial_slots'] for v in control_scores.values()):raise ValueError('Unpaired input control')
            patch=read_json(folder/'selection.json');score=m['best_qualified_pic50'] if m['best_qualified_pic50'] is not None else -1e30;rollback=False;confirmation=number>=19
            if not confirmation:
                delta=score-state['best']['best_qualified_pic50'];state['decline_streak']=state['decline_streak']+1 if delta<-master['decline_tolerance_pic50'] else 0
                if score>state['best_nc_score']:
                    state.update(best_nc_score=score,best_nc_patch=p['patch_id'],best_nc_round=label,
                        best_nc_contract=cfg['execution_contract'],best_nc_contrast_weighting=cfg.get('contrast_weighting',False))
                if p['weights']['type']>0 and score>state.get('best_typed_score',-1e31):
                    state.update(best_typed_score=score,best_typed_contract=cfg['execution_contract'],
                        best_typed_contrast_weighting=cfg.get('contrast_weighting',False))
                if delta>master['decline_tolerance_pic50']:
                    state.update(best=m,best_round=label,best_config=path.relative_to(repo).as_posix(),decline_streak=0)
                rollback=state['decline_streak']>=2
                if rollback:state['decline_streak']=0
            entry={'round':label,'strategy':cfg['strategy'],'patch_id':p['patch_id'],'template_id':p['template_id'],
                'contract':cfg['execution_contract'],'metrics':m,'controls':control_scores,'local_response_controls':matched,'initial_batch':cfg['pilot_initial_batch'],
                'reused_numeric_program':reused,'confirmation':confirmation,'inference_commit':sha,'rollback':rollback,
                'incumbent_after':state['best_round'],'selected_points':len(p['atom_slots']),'typed_sites':len(p['typed_slots'])}
            state['rounds'].append(entry);state['next_round']=number+1;write_json(folder/'results.json',entry);write_json(sp,state)
            active=read_json(repo/state['best_config']) if state.get('best_config') else read_json(repo/'configs/experiments/partial_region_search_20261014/active_best.json')
            active.update(nc_search_incumbent=state['best_round'],independent_validation_passed=False)
            write_json(repo/master['config_dir']/'active_best.json',active)
            if rollback:
                write_json(folder/'rollback.json',{'restored_checkpoint':state['best_round'],'reason':'Two consecutive highest-qualified final-score declines beyond .01; mathematical fixes retained'})
            sha=commit(repo,master,f"{'rollback' if rollback else 'results'}(nc-layout): {label} terminal feedback; incumbent {state['best_round']}")
            if rollback:state['rollback_commits'].append(sha);write_json(sp,state)
            emit('round_complete',round=label,patch=p['patch_id'],points=len(p['atom_slots']),score=score,
                mean=m['mean_all_pic50'],local_NC=m['local_NC_identity_accuracy'],incumbent=state['best_round'],rollback=rollback)
        if state['next_round']>master['max_rounds']:
            write_json(study/'COMPLETE.json',{'rounds':len(state['rounds']),'best_round':state['best_round'],'best':state['best'],
                'best_local_round':state['best_nc_round'],'final_confirmation':state['rounds'][-1],
                'target_reached':any(r.get('metrics') and r['metrics']['n_above_reference']>0 for r in state['rounds'])})
            commit(repo,master,'results(nc-layout): complete 20-round local NC search')
        return state
    finally:remote.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--stop-after',type=int,default=20);a=p.parse_args()
    run(a.repo,a.config,a.stop_after)


if __name__=='__main__':main()
