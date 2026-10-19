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


def analyze(repo,cfg,m,work):
    from .analysis import distribution,paired_cluster,write_csv
    datasets={};performance=[];pairs=[];responses=[];local_rows=[]
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
                values={}
                for key in ('injected_rms_A_mean','direct_support_count_mean','direct_support_fraction_mean',
                            'direct_injected_rms_A_mean','outside_injected_rms_A_mean','gradient_norm_mean'):
                    vals=[s[key] for s in active if key in s]
                    if vals:values[key]=float(np.mean(vals))
                responses.append({'condition':c['id'],'group':g['index'],'active_steps':len(active),
                    'nonzero_updates':sum(s['nonzero_molecules'] for s in active),**values})
        datasets[c['id']]=records
        performance.append({'condition':c['id'],**distribution(records,c['n'],cfg['tail_threshold'])})
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
    result={'research_sha256':m['research_sha256'],'status':status(cfg,m,work),
            'performance':performance,'paired':pairs,'responses':responses,
            'comparison_scope':m.get('comparison_scope','strict_historical_replay'),
            'interpretation':'Historical initial tensors and primitive random inputs are aligned. A current-runtime diagnostic is NOT a passed exact historical replay. Fresh same-runtime controls are mandatory in that scope. Small selected pilots are not unbiased N1000 or cross-target results.'}
    atomic_json(out/'summary.json',result);return result


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
                if c.get('reuse_from'):
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
    elif a.action=='analyze':print(json.dumps(analyze(a.repo,cfg,m,a.work),indent=2))
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
