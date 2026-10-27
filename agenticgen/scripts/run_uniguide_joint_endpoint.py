"""Background 2x2 pilot: coordinate inpainting x terminal output policy.

Two existing chemical-only native-output groups are reused; six groups are new.
"""
import argparse
import copy
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import types

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agenticgen.uniguide.io import read,write,frozen,sha,value_sha
from agenticgen.uniguide.joint_endpoint import (POLICY,ClockAudit,EndpointControl,compile_endpoint,
    finish_output,execute_variant,FactorialBudget)
from run_uniguide_chem_inpainting import now,verify_sources,CRITICAL

ARMS={'joint_native':dict(joint=True,short=False),
      'chemical_raw999':dict(joint=False,short=True),
      'joint_raw999':dict(joint=True,short=True)}


def prepare(root,source,prior):
    if not (prior/'FINALIZED.json').exists():raise ValueError('Previous chemical-only pilot must be finalized')
    old=read(prior/'STUDY.json');verify_sources(source,old)
    repo=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no']).strip():raise ValueError('Dirty execution checkout')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    import hashlib
    for name in CRITICAL:
        original_blob=subprocess.check_output(['git','-C',str(repo),'show',old['code_commit']+':agenticgen/src/agenticgen/'+name])
        if hashlib.sha256(original_blob).hexdigest()!=sha(repo/'agenticgen/src/agenticgen'/name):
            raise ValueError('Original chemical-only scientific engine changed: '+name)
    config=copy.deepcopy(read(prior/'config.json'));config.pop('chemical_only_pilot',None)
    config.update(root=str(root),joint_endpoint_pilot=dict(policy=POLICY,arms=ARMS,new_attempt_cap=300,
        gradient_guidance_retained=True,original_coordinate_windows=[10,25,26,50,51,75,76,90],
        short_grid='100 intervals, 101 points from 0 to 0.999; last loop forward at approximately 0.98901',
        short_output='last loop prediction before final integration/inpainting; no extra terminal forward or output overlay',
        endpoint_rescore_time=.9999))
    protected={}
    for t in old['controls']:
        for n in ['aggregate.json','records.json','COMPLETE.json','QUALITY_COMPLETE.json','quality.json','trace.h5','molecules_all_built.sdf']:
            protected[f'groups/{t}/{n}']=sha(prior/'groups'/t/n)
    protected['STUDY.json']=sha(prior/'STUDY.json');protected['FINALIZED.json']=sha(prior/'FINALIZED.json')
    spec=dict(policy=POLICY,code_commit=commit,config_sha256=value_sha(config),source_root=str(source),prior_root=str(prior),
        controls=old['controls'],teacher_hashes=old['teacher_hashes'],source_hashes=old['source_hashes'],prior_hashes=protected,
        new_arms=ARMS,reused_arm='chemical_native',new_attempt_cap=300,reused_attempts=100,
        population_per_group=50,seeds=[42,100045,200048,300051,400054],slots=list(range(10)),
        conditioning_scope='same nine PLIP direct atoms per teacher',gradient_guidance_retained=True,
        interventions='hard coordinate replacement on core only; fixed original Agent scales; no valence repair',
        source_campaign_remains_paused=True)
    root.mkdir(parents=True,exist_ok=True)
    frozen(root/'STUDY.json',spec);frozen(root/'config.json',config)
    teachers=read(source/'teachers.json');frozen(root/'teachers.json',teachers)
    from agenticgen.uniguide.native_tools import load_pt
    clock_reference={}
    for teacher in teachers:
        checkpoint=load_pt(source/'R0'/teacher['target']/'checkpoint_100.pt.gz')
        clock_reference[teacher['target']]=dict(integrated_state_times=[float(t[0]) for t in checkpoint['times']],
            native_final_forward_times=[float((t-1e-4)[0]) for t in checkpoint['times']],
            source='saved checkpoint_100 times plus pinned native _update_times(-1e-4); effective network times additionally logged during new runs')
    frozen(root/'OLD_TIME_AUDIT.json',clock_reference)
    for name in ['initials','random_stream_reference']:
        p=root/name
        if not p.exists():p.symlink_to(source/name,target_is_directory=True)
        if not p.is_symlink() or p.resolve()!=(source/name).resolve():raise ValueError('Wrong initial/RNG reference')
    return config,teachers,spec


def verify(source,prior,spec):
    verify_sources(source,spec)
    if any(sha(prior/p)!=h for p,h in spec['prior_hashes'].items()):raise ValueError('Previous pilot changed')


def run(root,source,prior):
    import fcntl
    import torch
    from agenticgen.uniguide import sampler
    from agenticgen.uniguide.prepare import load_model
    from agenticgen.uniguide.chem_inpainting import teacher_features
    from agenticgen.uniguide.quality_queue import enqueue
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.executor.lock').open('a+') as own,(source/'.executor.lock').open('r+') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'FINALIZED.json').exists():return
        if (root/'FAILED.json').exists():raise ValueError('Preserve failed run; no automatic restart')
        config,teachers,spec=prepare(root,source,prior);verify(source,prior,spec)
        write(root/'STARTED.json',dict(time_shanghai=now(),pid=os.getpid(),code_commit=spec['code_commit']))
        scientific_failures=[]
        for teacher in teachers:
            target=teacher['target'];model=load_model(config,teacher)
            features,meta=teacher_features(model,teacher);frozen(root/'features'/(target+'.json'),meta)
            original_generate=model._generate_selective;original_times=model._get_times
            source_text=(root/'provenance/native/instrumented_selective.py').read_text()
            for arm,params in ARMS.items():
                folder=root/'groups'/target/arm
                armconfig=dict(config,active_arm=dict(name=arm,**params))
                if (folder/'COMPLETE.json').exists():enqueue(armconfig,folder,teacher);continue
                if (folder/'FAILED.json').exists():raise ValueError('Do not repeat a failed saved group')
                folder.mkdir(parents=True,exist_ok=True);frozen(folder/'config.json',armconfig)
                model.integrator._guard_warned=False
                clock=ClockAudit()
                def observed_times(this,times,lig_mask,inpaint_mask):
                    packed=original_times(times,lig_mask,inpaint_mask);clock.observe(times,packed);return packed
                model._get_times=types.MethodType(observed_times,model)
                compiled,rendered=compile_endpoint(original_generate,source_text,params['short'])
                (folder/'effective_sampler.py').write_text(rendered)
                native=types.MethodType(compiled,model)
                def generate(*args,**kwargs):
                    return finish_output(native(*args,**kwargs),teacher,features,folder,
                                         params['joint'],params['short'],clock,model)
                model._generate_selective=generate
                old_control=sampler.Control
                baseline_trace=(prior/'groups'/target/'trace.h5' if not params['short'] else
                    root/'groups'/target/'chemical_raw999/trace.h5' if params['joint'] else None)
                sampler.Control=functools.partial(EndpointControl,chemical_features=features,joint=params['joint'],clock=clock,baseline_trace=baseline_trace)
                executor=execute_variant(sampler,lambda ignored:FactorialBudget(root),
                    'last loop uncovered prediction on 0..0.999; no terminal overlay' if params['short'] else
                    'native terminal prediction with chemical and coordinate core overlay; raw output retained separately')
                try:
                    executor(model,armconfig,teacher,spec['controls'][target]['aggregate']['schedule'],folder,'formal',50,
                        resume_path=None if params['short'] else source/'R0'/target/'checkpoint_010.pt.gz')
                except FloatingPointError as error:
                    scientific_failures.append(dict(target=target,arm=arm,error=str(error),charged_attempts=50))
                finally:
                    sampler.Control=old_control;model._generate_selective=original_generate;model._get_times=original_times
                    write(folder/'NETWORK_TIMES.json',clock.events)
                if (folder/'COMPLETE.json').exists():enqueue(armconfig,folder,teacher)
                verify(source,prior,spec)
                write(root/'progress.json',dict(time_shanghai=now(),last_target=target,last_arm=arm,
                    completed=[str(p.parent.relative_to(root)) for p in (root/'groups').glob('*/*/COMPLETE.json')],
                    scientific_failures=scientific_failures))
            del model;torch.cuda.empty_cache()
        verify(source,prior,spec)
        frozen(root/'SAMPLING_FINISHED.json',dict(time_shanghai=now(),budget_charged=sum(r['count'] for r in read(root/'attempts.json').values()),
            completed_groups=len(list((root/'groups').glob('*/*/COMPLETE.json'))),scientific_failures=scientific_failures,
            source_unchanged=True))
        report(root)


def report(root):
    import csv
    spec=read(root/'STUDY.json');prior=Path(spec['prior_root']);rows=[];details={}
    for target in spec['controls']:
        for arm in ['chemical_native',*ARMS]:
            folder=prior/'groups'/target if arm=='chemical_native' else root/'groups'/target/arm
            key=target+'/'+arm
            if not (folder/'COMPLETE.json').exists():
                details[key]=dict(status='failed' if (folder/'FAILED.json').exists() else 'incomplete',folder=str(folder));continue
            aggregate=read(folder/'aggregate.json')
            output=read(folder/'OUTPUT_POLICY.json') if arm!='chemical_native' else dict(raw_C=None,final_C_is_imposed=False,output_source='previous native chemical-only pilot')
            quality=read(folder/'QUALITY_COMPLETE.json') if (folder/'QUALITY_COMPLETE.json').exists() else None
            details[key]=dict(status='complete',aggregate=aggregate,output=output,quality=quality,folder=str(folder))
            rows.append(dict(target=target,arm=arm,attempts=50,mean_pic50=aggregate['mean_pic50'],C_A=aggregate['C'],
                C_is_imposed=output.get('final_C_is_imposed',False),raw_before_terminal_overlay_C=output.get('raw_C'),
                gap=aggregate['gap'],built=aggregate['built'],unique_smiles=aggregate['unique_smiles'],
                checked_attempts=quality['checked_attempts'] if quality else 0,
                posebusters_pass=quality['posebusters_pass'] if quality else None))
    write(root/'RESULTS.json',dict(time_shanghai=now(),policy=POLICY,groups=details,rows=rows,
        interpretation='For joint_native C is forced by coordinate replacement; assess raw C and raw999 outputs, validity and affinity. Output-policy comparison combines grid endpoint, last-loop source and omission of final overlays.',
        endpoint_rescore_time=.9999))
    if rows:
        with (root/'summary.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# 联合inpainting与终态输出策略实验','',f'更新时间：{now()}','',
          '| 教师 | 组 | 平均pIC50 | C（Å） | C是否强制 | 覆盖前C | 构建/50 | 唯一SMILES | 抽样PB通过 |',
          '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['target']} | {r['arm']} | {r['mean_pic50']:.6f} | {r['C_A']:.6f} | {r['C_is_imposed']} | {r['raw_before_terminal_overlay_C']} | {r['built']} | {r['unique_smiles']} | {r['posebusters_pass']}/10 |")
    text += ['', 'joint_native的终态核心坐标被覆盖，C接近0是直接约束结果，不能当作独立优化证据。raw999组输出最后循环未覆盖预测，不做额外终态前向或覆盖；100积分区间下最后网络调用约在0.98901，积分状态到0.999。网络有效原子时间记录于各组NETWORK_TIMES.json。所有组的endpoint affinity重评分时间仍为0.9999，以保持指标口径。',
             '原UG05与原巡检保持暂停。每组固定抽查10/50，未抽样40个不补查。']
    (root/'report.zh-CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','report'])
    p.add_argument('--root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--prior-root',required=True)
    a=p.parse_args();root,source,prior=map(lambda x:Path(x).resolve(),[a.root,a.source_root,a.prior_root])
    if len({root,source,prior})!=3 or any(root.is_relative_to(p) or p.is_relative_to(root) for p in (source,prior)):raise ValueError('Separate sibling result roots required')
    try:
        if a.mode=='prepare':prepare(root,source,prior)
        elif a.mode=='run':run(root,source,prior)
        else:report(root)
    except BaseException as error:
        if a.mode=='run' and root.exists():write(root/'FAILED.json',dict(time_shanghai=now(),error_type=type(error).__name__,error=str(error),no_automatic_retry=True))
        raise


if __name__=='__main__':main()
