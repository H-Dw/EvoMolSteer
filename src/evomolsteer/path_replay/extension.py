"""Independent R11 N1000 extension of an immutable B50 historical protocol.

The base config, receipts, tapes and scientific parameters are never rewritten.
Only complete, checksummed base batches can be reused; failed batches are retried
in this new workspace. Saved tensors can move to CPU without changing the B50
forward, the reward or its coordinate derivatives.
"""
import argparse
import copy
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import time
import traceback

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json,exclusive_lock
from .common import validate,identity,groups_for,read_batch,batch_path,binding,condition_program


def prepare(repo,base_config,output):
    repo=Path(repo).resolve();base=Path(base_config).resolve();cfg=read_json(base);validate(cfg)
    c=copy.deepcopy(next(c for c in cfg['conditions'] if c['id']=='NC_R11'));c['n']=1000
    manifest={'schema_version':'historical-path-extension-1','name':'path_replay_r11_extension_20261019',
        'base_config':base.relative_to(repo).as_posix(),'base_config_sha256':digest(base),
        'base_protocol_sha256':cfg['protocol_sha256'],'base_work':cfg['remote_work'],
        'remote_work':str(PurePosixPath(cfg['remote_work']).parent/'evomolsteer_path_replay_r11_extension_20261019'),
        'condition':c,'saved_tensor_policy':'saved_tensors_cpu_exact','batch_size':50,'n':1000,
        'batch_timeout_seconds':7200,'max_worker_restarts':1,'gpu_memory_threshold_percent':10.,
        'earliest_start_utc':'2026-10-18T23:54:36Z',
        'reuse':'only complete base batches with matching frozen program, teacher, model, input and noise fingerprints'}
    manifest['extension_sha256']=identity(manifest);validate_extension(cfg,manifest)
    atomic_json(output,manifest);return manifest


def validate_extension(cfg,ext):
    validate(cfg)
    if ext.get('extension_sha256')!=identity({k:v for k,v in ext.items() if k!='extension_sha256'}):
        raise ValueError('Extension manifest changed')
    if ext['schema_version']!='historical-path-extension-1' or ext['base_protocol_sha256']!=cfg['protocol_sha256']:
        raise ValueError('Extension base protocol differs')
    expected=copy.deepcopy(next(c for c in cfg['conditions'] if c['id']=='NC_R11'));expected['n']=1000
    if ext['condition']!=expected or ext['n']!=1000 or ext['batch_size']!=50:
        raise ValueError('Extension changed the frozen R11 condition or B50 budget')
    if ext['saved_tensor_policy']!='saved_tensors_cpu_exact': raise ValueError('Expected lossless activation offload')
    if ext['gpu_memory_threshold_percent']!=10.: raise ValueError('VRAM threshold differs from authorization')
    if ext['base_work']!=cfg['remote_work'] or Path(ext['remote_work'])==Path(ext['base_work']):
        raise ValueError('Extension must not overwrite its base workspace')
    if len(groups_for(cfg,ext['condition']))!=20: raise ValueError('Extension must use all 20 original B50 groups')


def load(repo,manifest_path):
    repo=Path(repo);ext=read_json(manifest_path);base=repo/ext['base_config']
    if digest(base)!=ext['base_config_sha256']: raise ValueError('Base config file differs')
    cfg=read_json(base);validate_extension(cfg,ext);return cfg,ext,base


def attach_verified_tapes(cfg,ext,work):
    source=Path(ext['base_work'])/'tapes';record=read_json(source/'verification.json')
    if not record.get('passed') or record['protocol_sha256']!=cfg['protocol_sha256']:
        raise ValueError('Base random tapes are unverified or belong to a different protocol')
    if len(record['artifacts'])!=20: raise ValueError('Expected all 20 original random tapes')
    for name,sha in record['artifacts'].items():
        if digest(source/name)!=sha: raise ValueError('Base random tape changed: '+name)
    target=Path(work)/'tapes'
    if target.exists() or target.is_symlink():
        if not target.is_symlink() or target.resolve()!=source.resolve(): raise ValueError('Unexpected extension tape path')
    else: target.symlink_to(source,target_is_directory=True)
    return record


def reuse_completed(cfg,ext,work):
    original=next(c for c in cfg['conditions'] if c['id']=='NC_R11');c=ext['condition'];audit=[]
    for g in groups_for(cfg,original):
        r=read_batch(ext['base_work'],cfg,original,g)
        row={'group':g['index'],'source_status':r['status'] if r else 'missing','reused':False}
        if r is not None and r['status']=='complete':
            old=batch_path(ext['base_work'],original,g);new=batch_path(work,c,g)
            current=read_batch(work,cfg,c,g)
            if current is None:
                new.parent.mkdir(parents=True,exist_ok=True)
                for key,sha in r['artifacts'].items():
                    dest=new.parent/key
                    if not dest.resolve().is_relative_to(new.parent.resolve()): raise ValueError('Unsafe artifact path')
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old.parent/key,dest)
                    if digest(dest)!=sha: raise ValueError('Reused artifact checksum differs')
                adopted=copy.deepcopy(r);adopted.update(binding=binding(cfg,c,g),
                    reused_from={'receipt':str(old),'sha256':digest(old),'source_binding':r['binding']},
                    extension_sha256=ext['extension_sha256'])
                atomic_json(new,adopted)
            row.update(reused=True,receipt_sha256=digest(old))
        elif r is not None:
            row.update(source_error=r.get('error'),source_receipt_sha256=digest(batch_path(ext['base_work'],original,g)))
        audit.append(row)
    result={'base_protocol_sha256':cfg['protocol_sha256'],'extension_sha256':ext['extension_sha256'],
            'base_groups':audit,'reused_n':50*sum(r['reused'] for r in audit),'new_required_n':1000-50*sum(r['reused'] for r in audit)}
    atomic_json(Path(work)/'reuse_audit.json',result);return result


def status(cfg,ext,work):
    receipts=[read_batch(work,cfg,ext['condition'],g) for g in groups_for(cfg,ext['condition'])]
    return {'condition':'NC_R11','n':1000,'batch_size':50,
        **{s:50*sum(r is not None and r['status']==s for r in receipts) for s in ('complete','failed','skipped_condition_error')},
        'pending':50*sum(r is None for r in receipts),'extension_sha256':ext['extension_sha256']}


def analyze(repo,cfg,ext,work,bootstrap=True):
    from .analysis import distribution,paired_cluster,write_csv
    from .common import groups_for
    work=Path(work);out=work/'analysis';out.mkdir(parents=True,exist_ok=True)
    attempts={a['id']:a for a in cfg['attempts']};datasets={};planned={}
    for c in cfg['conditions']:
        override=c['id']=='NC_R11';source=ext['condition'] if override else next((x for x in cfg['conditions'] if x['id']==c.get('alias_of')),c)
        records=[]
        for g in groups_for(cfg,source):
            r=read_batch(work if override else ext['base_work'],cfg,source,g)
            if r:
                records += [dict(attempts[v['id']],**{k:x for k,x in v.items() if k not in attempts[v['id']]}) for v in r['records']]
        datasets[c['id']]=records;planned[c['id']]=1000 if override else c['n']
    datasets['original_steer']=[dict(a,pic50=a['historical_pic50'],valid_connected=a['historical_valid_connected']) for a in cfg['attempts']]
    planned['original_steer']=1000
    native=Path(cfg['remote_dataset'])/'results/main1000_w050/unguided'
    if not native.exists():native=Path(repo)/'data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/unguided'
    native_rows=[r for p in sorted(native.glob('batch_*/final_records.json')) for r in read_json(p)]
    if native_rows:
        datasets['historical_native_unpaired']=[{'id':f"native_b{r['batch']:03d}_s{r['slot']:03d}",'batch':r['batch'],
            'pic50':r['pic50_on_rescore'],'valid_connected':bool(r.get('build_success') and r.get('connected'))} for r in native_rows]
        planned['historical_native_unpaired']=len(native_rows)
    performance=[];paired=[]
    for name,records in datasets.items():
        n=planned[name];cohorts=[('all1000' if n==1000 else 'assigned_top20',n)]
        if name!='historical_native_unpaired':cohorts += [('top05',50),('top10',100)]+([('top20',200)] if n==1000 else [])
        for label,size in cohorts:
            subset=records if label in ('all1000','assigned_top20') else [r for r in records if r['rank']<=size]
            performance.append({'condition':name,'cohort':label,**distribution(subset,size,cfg['tail_threshold'])})
            if name=='NC_R11':
                for ref in ('original_steer','unguided','M2','ancestor_R02_v3','motif_R15_B50'):
                    controls=datasets[ref]
                    if label not in ('all1000','assigned_top20'):controls=[r for r in controls if r['rank']<=size]
                    paired.append({'condition':name,'reference':ref,'cohort':label,'reference_planned':planned[ref],
                        'full_cohort_comparator':planned[ref]>=size,
                        **paired_cluster(subset,controls,cfg['bootstrap_replicates'] if bootstrap else 0,cfg['bootstrap_seed'])})
    write_csv(out/'performance.csv',performance);write_csv(out/'paired.csv',paired)
    result={'extension_sha256':ext['extension_sha256'],'base_protocol_sha256':cfg['protocol_sha256'],
        'performance':performance,'paired':paired,'status':status(cfg,ext,work),
        'interpretation':'Retrospectively conditioned historical paths. Unguided and M2 have only top200; all1000 comparisons to them cover their intersection only.'}
    atomic_json(out/'summary.json',result)
    lines=['# N/C R11 历史路径回放扩展','',
        'R11沿用原1000路径和20个B50分组。原Steer前5/10/20%在统计时提取。',
        'No guidance与M2仅覆盖前200条：整体1000的配对比较只能取交集，不能当作1000例对照。',
        '原生无引导另列为非配对历史对照；历史路径回放不能解释为随机起点泛化。','',
        '|条件|范围|评分/计划|均值|P90|P95|P99|Best|有效Best|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.6f}'
    for r in performance:lines.append('|'+ '|'.join([r['condition'],r['cohort'],f"{r['n_scored']}/{r['n_planned']}"]+[fmt(r[k]) for k in ('mean','p90','p95','p99','best','best_valid')])+'|')
    (out/'report.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result


def run(repo,manifest_path,work):
    from .queue import zero_gate,run_condition
    from .tape import verify_source
    from .resources import resource_gate
    from datetime import datetime,timezone
    repo=Path(repo).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    cfg,ext,base=load(repo,manifest_path);c=ext['condition']
    if work!=Path(ext['remote_work']).resolve(): raise ValueError('Workspace differs from extension manifest')
    earliest=datetime.fromisoformat(ext['earliest_start_utc'].replace('Z','+00:00')).timestamp()
    if time.time()<earliest: raise RuntimeError('Before authorized start time')
    with exclusive_lock(work/'.queue.lock'):
        state={'status':'verifying','pid':os.getpid(),'started_unix':time.time(),'extension_sha256':ext['extension_sha256']}
        atomic_json(work/'queue_status.json',state)
        try:
            validate(cfg);condition_program(repo,c);verify_source(cfg)
            if digest(cfg['checkpoint'])!=cfg['checkpoint_sha256']: raise ValueError('Model checkpoint changed')
            gate=zero_gate(ext['base_work'],cfg);atomic_json(work/'inherited_zero_replay_gate.json',gate)
            attach_verified_tapes(cfg,ext,work);reuse=reuse_completed(cfg,ext,work)
            atomic_json(work/'extension.json',ext)
            resources=resource_gate(work,ext['gpu_memory_threshold_percent']);atomic_json(work/'prelaunch_resources.json',resources)
            if not resources['passed']:
                state.update(status='waiting_for_resources',resources=resources,**status(cfg,ext,work))
                atomic_json(work/'queue_status.json',state);return state
            state.update(status='running',**status(cfg,ext,work));atomic_json(work/'queue_status.json',state)
            execution=dict(cfg,batch_timeout_seconds=ext['batch_timeout_seconds'],max_worker_restarts=ext['max_worker_restarts'])
            run_condition(repo,base,execution,c,work,Path(manifest_path).resolve())
            analyze(repo,cfg,ext,work)
            state.update(status='complete',finished_unix=time.time(),**status(cfg,ext,work))
            if state['failed'] or state['skipped_condition_error']:state['status']='complete_with_errors'
            atomic_json(work/'queue_status.json',state);return state
        except BaseException:
            state.update(status='blocked',error=traceback.format_exc(),**status(cfg,ext,work));atomic_json(work/'queue_status.json',state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','status','analyze','collect'])
    p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--base-config',type=Path);p.add_argument('--work',type=Path);p.add_argument('--output');a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare(a.repo,a.base_config,a.manifest),indent=2));return
    cfg,ext,base=load(a.repo,a.manifest);work=a.work or Path(ext['remote_work'])
    if a.action=='run':result=run(a.repo,a.manifest,work)
    elif a.action=='analyze':
        result=analyze(a.repo,cfg,ext,work);result={'status':result['status'],'analysis':str(work/'analysis')}
    elif a.action=='collect':
        from .collection import collect
        if a.output is None:p.error('collect requires --output')
        collect(cfg,work,a.output);return
    else:result={'queue':read_json(work/'queue_status.json') if (work/'queue_status.json').exists() else None,'counts':status(cfg,ext,work),
        'worker':read_json(work/'worker_state.json') if (work/'worker_state.json').exists() else None}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
