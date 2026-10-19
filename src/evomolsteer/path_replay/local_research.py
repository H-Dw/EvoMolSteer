"""Small, independent local-feature experiments on unchanged historical paths.

Local Analyst/Designer decisions are frozen before dispatch. The remote worker
only executes those programs and uses the existing exact replay input adapter.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json,exclusive_lock
from .common import identity,validate,condition_program,groups_for,read_batch,batch_path,binding,fail_batch


def load(repo,path):
    repo=Path(repo);m=read_json(path);base=repo/m['base_config'];cfg=read_json(base);validate(cfg)
    if digest(base)!=m['base_config_sha256'] or m['base_protocol_sha256']!=cfg['protocol_sha256']:
        raise ValueError('Research base protocol changed')
    if m['research_sha256']!=identity({k:v for k,v in m.items() if k!='research_sha256'}):
        raise ValueError('Research manifest changed')
    if m['schema_version']!='local-feature-path-research-1' or m['batch_size']!=50 or m['saved_tensor_policy']!='saved_tensors_cpu_exact':
        raise ValueError('Expected exact B50 activation-offload protocol')
    if m['remote_work']==m['base_work'] or m['base_work']!=cfg['remote_work']:
        raise ValueError('Research must not overwrite historical results')
    scope=m.get('comparison_scope','strict_historical_replay')
    if scope not in ('strict_historical_replay','current_runtime_paired_diagnostic'):
        raise ValueError('Unknown comparison scope')
    if scope=='current_runtime_paired_diagnostic':
        evidence=m['diagnostic_evidence']
        if digest(repo/evidence['path'])!=evidence['sha256']:raise ValueError('Runtime diagnostic evidence changed')
        arrays=read_json(repo/evidence['path'])['native_vs_offload_zero_state_arrays']
        if set(arrays)!={'terminal_coords_A','half_coords_A','mask','atomics','bonds','charges'} or not all(v['equal'] is True for v in arrays.values()):
            raise ValueError('Current runtime repeatability has not been established')
        if any(c.get('reuse_from') for c in m['conditions']):raise ValueError('Cross-device diagnostic must recompute every control')
        if not {str(g['index']) for g in groups_for(cfg,m['conditions'][0])}<=set(m['native_zero_repeat_sources']):
            raise ValueError('Every diagnostic group needs its own repeatability source')
    ids=[c['id'] for c in m['conditions']]
    if len(ids)!=len(set(ids)) or m['conditions'][0]['kind']!='unguided':raise ValueError('Unique conditions, zero replay first')
    for c in m['conditions']:
        groups_for(cfg,c)
        program,teacher=condition_program(repo,c)
        if program is not None:
            from evomolsteer.ancestral_modes.programs import validate as validate_program
            validate_program(program,teacher)
            frozen=c.get('role')=='frozen_current_reference'
            if frozen:
                original=next(v for v in cfg['conditions'] if v['id']==c['frozen_source_condition'])
                for key in ('kind','program','reference','program_sha256','reference_sha256','window'):
                    if c.get(key)!=original.get(key):raise ValueError('Current reference changed frozen execution: '+key)
            if not c.get('reuse_from') and not frozen and 'local_feature_binding' not in program:
                raise ValueError('New local reward lacks an executable feature binding')
    return cfg,m,base


def receipts(cfg,m,work):
    return {c['id']:[read_batch(work,cfg,c,g) for g in groups_for(cfg,c)] for c in m['conditions']}


def status(cfg,m,work):
    rows=[]
    for c in m['conditions']:
        r=[read_batch(work,cfg,c,g) for g in groups_for(cfg,c)]
        rows.append({'id':c['id'],'n':c['n'],**{s:50*sum(v is not None and v['status']==s for v in r)
                    for s in ('complete','failed','skipped_condition_error')},'pending':50*sum(v is None for v in r)})
    return rows


def reuse(cfg,m,c,work):
    source=next(v for v in cfg['conditions'] if v['id']==c['reuse_from'])
    for key in ('kind','program','reference','program_sha256','reference_sha256','numeric_program_sha256'):
        if c.get(key)!=source.get(key):raise ValueError('Reuse changed scientific execution: '+key)
    for g in groups_for(cfg,c):
        r=read_batch(m['base_work'],cfg,source,g)
        if not r or r['status']!='complete':raise ValueError('Reusable baseline batch is incomplete')
        if read_batch(work,cfg,c,g) is not None:continue
        old=batch_path(m['base_work'],source,g);dest=batch_path(work,c,g);dest.parent.mkdir(parents=True,exist_ok=True)
        for name,sha in r['artifacts'].items():
            target=dest.parent/name
            if not target.resolve().is_relative_to(dest.parent.resolve()):raise ValueError('Unsafe reused artifact')
            shutil.copy2(old.parent/name,target)
            if digest(target)!=sha:raise ValueError('Reuse checksum differs')
        adopted=copy.deepcopy(r);adopted.update(condition=c['id'],binding=binding(cfg,c,g),
            reused_from={'receipt':str(old),'sha256':digest(old),'source_binding':r['binding']})
        for row in adopted['records']:row['condition']=c['id']
        atomic_json(dest,adopted)


def reuse_current(repo,cfg,m,c,work):
    """Reuse only a checked, identical control from this GPU/runtime family."""
    source=c['reuse_current_from'];other_cfg,other,_=load(repo,Path(repo)/source['manifest'])
    oldc=next(v for v in other['conditions'] if v['id']==source['condition'])
    if other_cfg['protocol_sha256']!=cfg['protocol_sha256'] or other.get('comparison_scope')!=m.get('comparison_scope'):
        raise ValueError('Current-control protocol/scope differs')
    for key in ('kind','window','program_sha256','reference_sha256'):
        if c.get(key)!=oldc.get(key):raise ValueError('Current-control scientific identity differs: '+key)
    if other['saved_tensor_policy']!=m['saved_tensor_policy']:raise ValueError('Current-control activation policy differs')
    source_work=Path(other['remote_work'])
    import torch
    audit=read_json(source_work/'methods'/oldc['id']/'source_audit.json')
    if (audit['device_name']!=torch.cuda.get_device_name(0) or audit['torch']!=torch.__version__ or audit['hip']!=torch.version.hip):
        raise ValueError('Current-control device/library identity differs')
    # Ignore pure orchestration/analysis additions; require identical numerical
    # control, replay, model adapter and integrator instrumentation source bytes.
    for name in ('path_replay/worker.py','path_replay/native.py','path_replay/tape.py','ancestral_modes/reward.py',
                 'ancestral_modes/bank.py','final_guidance/native.py','final_guidance/reward.py'):
        if digest(source_work/'runtime/src/evomolsteer'/name)!=digest(Path(repo)/'src/evomolsteer'/name):
            raise ValueError('Current-control numerical implementation differs: '+name)
    for g in groups_for(cfg,c):
        if g not in groups_for(cfg,oldc):raise ValueError('Current-control group not present in source')
        r=read_batch(source_work,cfg,oldc,g)
        if not r or r['status']!='complete':raise ValueError('Current control is incomplete')
        if read_batch(work,cfg,c,g) is not None:continue
        old=batch_path(source_work,oldc,g);dest=batch_path(work,c,g);dest.parent.mkdir(parents=True,exist_ok=True)
        for name,sha in r['artifacts'].items():
            target=dest.parent/name
            if not target.resolve().is_relative_to(dest.parent.resolve()):raise ValueError('Unsafe current-control artifact')
            shutil.copy2(old.parent/name,target)
            if digest(target)!=sha:raise ValueError('Current-control checksum changed')
        adopted=copy.deepcopy(r);adopted.update(condition=c['id'],binding=binding(cfg,c,g),
            reused_current_from={'receipt':str(old),'sha256':digest(old),'binding':r['binding'],'checked_device':audit['device_name']})
        for row in adopted['records']:row['condition']=c['id']
        atomic_json(dest,adopted)
    (Path(work)/'methods'/c['id']).mkdir(parents=True,exist_ok=True)
    atomic_json(Path(work)/'methods'/c['id']/'source_audit.json',dict(audit,condition=c,reused_current_from=source))


def control_summary(active,batch_size):
    import numpy as np
    values={}
    for key in ('injected_rms_A_mean','direct_support_count_mean','direct_support_fraction_mean',
                'direct_injected_rms_A_mean','outside_injected_rms_A_mean','current_gradient_rms_mean',
                'atom_clip_fraction','cumulative_clip_fraction','gradient_native_observed_cosine_mean',
                'assignment_switch_fraction','point_rms_A'):
        vals=[s[key] for s in active if key in s]
        if vals:values[key]=float(np.mean(vals))
    nonzero=sum(s['nonzero_molecules'] for s in active)
    values.update(active_steps=len(active),nonzero_updates=nonzero,
        active_seed_update_fraction=nonzero/(batch_size*len(active)) if active else None)
    for key in ('direct_injected_rms_A_mean','outside_injected_rms_A_mean'):
        if active and all(key in s for s in active):
            values[key.replace('_mean','_nonzero_mean')]=sum(s[key]*batch_size for s in active)/nonzero if nonzero else None
    return values


def analyze(repo,cfg,m,work,export_docs=False):
    from .analysis import distribution,paired_cluster,write_csv
    datasets={};performance=[];pairs=[];responses=[];local_rows=[];controlled_rows=[]
    local_conditions=[c for c in m['conditions'] if c['kind']!='unguided' and not c.get('reuse_from') and c.get('role')!='frozen_current_reference']
    feature_program,feature_teacher=condition_program(repo,local_conditions[0]) if local_conditions else (None,None)
    for c in m['conditions']:
        records=[]
        for g in groups_for(cfg,c):
            r=read_batch(work,cfg,c,g)
            if r:records.extend(r['records'])
            if r and r['status']=='complete' and feature_program is not None:
                import numpy as np
                from .local_features import measure
                with np.load(batch_path(work,c,g).parent/'states.npz') as states:
                    measured=measure(states['half_coords_A'],states['terminal_coords_A'],states['mask'],feature_teacher,feature_program)
                for rec,v in zip(r['records'],measured):
                    local_rows.append({'condition':c['id'],'group':g['index'],'id':rec['id'],'pic50':rec['pic50'],**v})
            if r and r['status']=='complete' and c['kind']!='unguided':
                import gzip,numpy as np
                with gzip.open(batch_path(work,c,g).parent/'guidance.json.gz','rt') as f:audit=json.load(f)
                active=[s for s in audit['steps'] if s['active']]
                if c in local_conditions and active and active[-1].get('actual_local_assignments') is not None:
                    program,teacher=condition_program(repo,c)
                    with np.load(batch_path(work,c,g).parent/'states.npz') as states:
                        tracked=measure(states['half_coords_A'],states['terminal_coords_A'],states['mask'],teacher,program,
                            assignments=active[-1]['actual_local_assignments'])
                    for i,(rec,v) in enumerate(zip(r['records'],tracked)):
                        controlled_rows.append({'condition':c['id'],'group':g['index'],'id':rec['id'],'pic50':rec['pic50'],
                            'operator':program['operator'],'last_control_forward_time':active[-1]['step']/program['steps'],
                            'last_forward_point_gap_A':active[-1]['target_error_per_seed_A'][i],
                            'last_forward_reward_mean':active[-1]['reward_mean'],**v})
                responses.append({'condition':c['id'],'group':g['index'],**control_summary(active,len(r['records']))})
        datasets[c['id']]=records
        original_best=max(a['historical_pic50'] for a in cfg['attempts'])
        performance.append({'condition':c['id'],**distribution(records,c['n'],cfg['tail_threshold']),
            'above_original_global_best_count':sum(r.get('pic50') is not None and r['pic50']>=original_best for r in records),
            'above_original_global_best_valid_count':sum(r.get('pic50') is not None and r['pic50']>=original_best and r.get('valid_connected',False) for r in records)})
        ids={k for g in groups_for(cfg,c) for k in g['ids']}
        historic=[dict(a,pic50=a['historical_pic50'],valid_connected=a['historical_valid_connected']) for a in cfg['attempts'] if a['id'] in ids]
        pairs.append({'condition':c['id'],'reference':'original_steer',**paired_cluster(records,historic,1000,42)})
    for name,rows in datasets.items():
        for ref in (m['conditions'][0]['id'],m.get('parent_condition_id')):
            if ref and name!=ref and ref in datasets:
                pairs.append({'condition':name,'reference':ref,**paired_cluster(rows,datasets[ref],1000,42)})
    out=Path(work)/'analysis';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'performance.csv',performance);write_csv(out/'paired.csv',pairs);write_csv(out/'response.csv',responses)
    write_csv(out/'local_formation_retention.csv',local_rows)
    write_csv(out/'active_support_formation_retention.csv',controlled_rows)
    result={'research_sha256':m['research_sha256'],'status':status(cfg,m,work),
            'analysis_code_sha256':{'analysis':digest(__file__),'local_features':digest(Path(__file__).with_name('local_features.py'))},
            'performance':performance,'paired':pairs,'responses':responses,
            'comparison_scope':m.get('comparison_scope','strict_historical_replay'),
            'interpretation':'Historical initial tensors and primitive random inputs are aligned. A current-runtime diagnostic is NOT a passed exact historical replay. Fresh same-runtime controls are mandatory in that scope. Small selected pilots are not unbiased N1000 or cross-target results.'}
    atomic_json(out/'summary.json',result)
    render_report(repo,cfg,m,work,result,controlled_rows,export_docs)
    return result


def render_report(repo,cfg,m,work,result,controlled_rows,export_docs=False):
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    lines=['# 局部特征推理对照：'+m['name'],'',
        '本报告按已验证完成记录生成；缺失/失败不当作零分，不把复用对照重复计数。',
        '**比较范围：** '+result['comparison_scope']+'。','']
    if result['comparison_scope']=='current_runtime_paired_diagnostic':
        lines+=['当前GPU与历史GPU存在数值差异，原1e-5严格回放验证未通过。本实验独立使用当前运行环境的配对对照，不能写成严格历史复现通过。','']
    lines+=['使用冻结的历史初态和逐步原始随机输入，B50、100积分步、无SMC；引导窗口后继续至终态。',
            '本次小池按历史优质路径选择，不代表整体1000例或随机起点泛化。','',
            '|条件|评分/计划|均值|中位数|P90|P95|P99|Best|有效Best|高分数|有效数|应变中位数|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['performance']:
        cells=[r['condition'],f"{r['n_scored']}/{r['n_planned']}"]+[fmt(r[k]) for k in ('mean','median','p90','p95','p99','best','best_valid')]
        cells += [str(r['high_score_count']),str(r['valid_count']),fmt(r['strain_median'])]
        lines.append('|'+ '|'.join(cells)+'|')
    lines+=['',f"高分阈值为冻结协议的{cfg['tail_threshold']:.6f} pIC50；应变使用MMFF缓解能量/重原子，仅在可计算结构中统计。",
            f"历史原Steer全部1000条的最高分为{max(a['historical_pic50'] for a in cfg['attempts']):.6f}；与小池Best的样本量不同。",'',
            '## 实际区域与控制响应','',
            '|条件|算子|教师槽位|时钟|窗口|实际平均支持数|平均注入RMS Å|支持内RMS Å|支持外RMS Å|',
            '|---|---|---|---|---|---:|---:|---:|---:|']
    for c in m['conditions']:
        if c['kind']=='unguided':continue
        p,_=condition_program(repo,c);r=next((r for r in result['responses'] if r['condition']==c['id']),{})
        lines.append('|'+ '|'.join([c['id'],p['operator'],str(p['mode_specs'][0]['atom_slots']),p['clock_policy'],str(p['window'])]
            +[fmt(r.get(k)) for k in ('direct_support_count_mean','injected_rms_A_mean','direct_injected_rms_A_mean','outside_injected_rms_A_mean')])+'|')
    lines+=['','教师支持限制奖励参考区域；生成器的坐标VJP仍可间接改变其它输入点。不能把未激活的方向、口袋或化学元数据写成生效奖励。','',
            '## 配对差值','',
            '|条件|对照|配对数|原来源批次数|均值差|95%区间|改善比例|',
            '|---|---|---:|---:|---:|---|---:|']
    for r in result['paired']:
        if not r['n_paired']:continue
        ci='['+fmt(r.get('mean_delta_ci_low'))+', '+fmt(r.get('mean_delta_ci_high'))+']'
        lines.append('|'+ '|'.join([r['condition'],r['reference'],str(r['n_paired']),str(r['source_batches']),fmt(r['mean_delta']),ci,fmt(r['fraction_improved'])])+'|')
    lines+=['','区间按原来源批次聚类；历史原Steer比较受跨GPU数值差异影响，当前对照比较另列。','',
            '## 形成、保持及价值的解释','',
            '`local_formation_retention.csv`是统一的半程观察性匹配；`active_support_formation_retention.csv`追踪最后控制forward实际匹配的槽位，二者不能混用。',
            '半程使用真实半程教师，终态使用真实终态教师。终态重匹配仅检测空间模式存在，不能伪装成固定槽位保持；槽位移动也不能自动判定区域丢失。',
            '最后forward点差是在更新前记录的端点比较；pair/pocket奖励的实际目标由对应算子决定。','']
    for c in m['conditions']:
        rows=[r for r in controlled_rows if r['condition']==c['id']]
        if not rows:continue
        values={key:sum(r[key] for r in rows)/len(rows) for key in ('last_forward_point_gap_A','terminal_tracked_point_error_A','terminal_rematched_point_error_A')}
        lines.append(c['id']+'：'+', '.join(k+'='+fmt(v)+' Å' for k,v in values.items())+'。')
    lines+=['','只形成几何模式而未提高最终pIC50，不能宣称奖励有效。少量开发池极值提高仍需要独立输入池复核。','',
            '原始评分、失败、结构与逐步控制记录留在对应结果目录。分析实现指纹：`'+json.dumps(result['analysis_code_sha256'],sort_keys=True)+'`。','']
    text='\n'.join(lines)
    (Path(work)/'analysis/report.zh-CN.md').write_text(text,encoding='utf-8')
    if export_docs:
        destination=Path(repo)/'docs/experiments'/m['name'];destination.mkdir(parents=True,exist_ok=True)
        (destination/'report.zh-CN.md').write_text(text,encoding='utf-8')


def run(repo,path,work):
    from .extension import attach_verified_tapes
    from .queue import zero_gate
    from .tape import verify_source
    repo=Path(repo).resolve();work=Path(work).resolve();cfg,m,base=load(repo,path)
    if work!=Path(m['remote_work']).resolve():raise ValueError('Workspace differs from manifest')
    work.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(work/'.queue.lock'):
        state={'status':'verifying','pid':os.getpid(),'research_sha256':m['research_sha256'],'started_unix':time.time()}
        atomic_json(work/'queue_status.json',state)
        try:
            verify_source(cfg)
            if digest(cfg['checkpoint'])!=cfg['checkpoint_sha256']:raise ValueError('Model checkpoint changed')
            gate=zero_gate(m['base_work'],cfg,publish=False)
            attach_verified_tapes(cfg,m,work);atomic_json(work/'historical_zero_gate.json',gate)
            for c in m['conditions']:
                state.update(status='running',current=c['id'],conditions=status(cfg,m,work));atomic_json(work/'queue_status.json',state)
                if c.get('reuse_current_from'):
                    reuse_current(repo,cfg,m,c,work)
                elif c.get('reuse_from'):
                    reuse(cfg,m,c,work)
                else:
                    for g in groups_for(cfg,c):
                        if read_batch(work,cfg,c,g) is not None:continue
                        args=[sys.executable,'-u','-m','evomolsteer.path_replay.worker','--repo',str(repo),
                              '--config',str(base),'--work',str(work),'--condition',c['id'],'--group',str(g['index']),
                              '--research-manifest',str(path)]
                        log=work/'logs'/f"{c['id']}_g{g['index']:03d}.log";log.parent.mkdir(parents=True,exist_ok=True)
                        if shutil.disk_usage(work).free<cfg['minimum_free_bytes']:raise RuntimeError('Low disk reserve')
                        try:
                            with log.open('ab',buffering=0) as f:
                                child=subprocess.Popen(args,cwd=repo,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                                (work/'worker.pid').write_text(str(child.pid))
                                try:code=child.wait(timeout=m.get('batch_timeout_seconds',7200))
                                except BaseException:
                                    from evomolsteer.elite_alignment.queue import stop_child
                                    stop_child(child);raise
                            if code and read_batch(work,cfg,c,g) is None:fail_batch(work,cfg,c,g,'Worker exit '+str(code))
                        except Exception:fail_batch(work,cfg,c,g,traceback.format_exc())
                if c['kind']=='unguided':
                    for g in groups_for(cfg,c):
                        r=read_batch(work,cfg,c,g)
                        complete=r is not None and r['status']=='complete'
                        q=read_json(batch_path(work,c,g).parent/'equivalence.json') if complete else {}
                        if m.get('comparison_scope')=='current_runtime_paired_diagnostic' and complete:
                            import numpy as np
                            source=Path(m['native_zero_repeat_sources'][str(g['index'])])
                            equality={}
                            with np.load(source) as old,np.load(batch_path(work,c,g).parent/'states.npz') as new:
                                for key in old.files:equality[key]=bool(np.array_equal(old[key],new[key]))
                            atomic_json(work/'current_runtime_repeatability.json',{'passed':all(equality.values()),'arrays':equality,
                                'source':str(source),'source_sha256':digest(source),'historical_equivalence_passed':q.get('passed',False)})
                            if not all(equality.values()):raise ValueError('Current-runtime baseline is not repeatable; diagnostics blocked')
                        elif not complete or not q.get('passed'):
                            raise ValueError('Current runtime zero-guidance exact replay failed; guidance blocked')
                analyze(repo,cfg,m,work)
            rows=status(cfg,m,work);state.update(status='complete_with_errors' if any(r['failed'] or r['skipped_condition_error'] for r in rows) else 'complete',
                current=None,conditions=rows,finished_unix=time.time());atomic_json(work/'queue_status.json',state)
        except BaseException:
            state.update(status='blocked',error=traceback.format_exc(),conditions=status(cfg,m,work));atomic_json(work/'queue_status.json',state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','status','analyze','collect'])
    for name in ('repo','manifest','work'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();cfg,m,_=load(a.repo,a.manifest)
    if a.action=='run':run(a.repo,a.manifest.resolve(),a.work)
    elif a.action=='analyze':print(json.dumps(analyze(a.repo,cfg,m,a.work,export_docs=not bool(os.environ.get('EVOMOLSTEER_RUNTIME_COMMIT'))),indent=2))
    elif a.action=='status':
        extra={}
        for key in ('queue_status','worker_state','worker_error'):
            f=a.work/(key+'.json')
            if f.exists():extra[key]=read_json(f)
        print(json.dumps({'counts':status(cfg,m,a.work),**extra},indent=2))
    else:
        from .collection import collect
        collect(cfg,a.work,'-')


if __name__=='__main__':main()
