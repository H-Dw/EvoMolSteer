"""Sequential, resumable B50 queue with shared replay gates and error isolation."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from evomolsteer.io import read_json,digest
from evomolsteer.storage.transactions import atomic_json,exclusive_lock
from evomolsteer.elite_alignment.queue import stop_child
from .common import validate,condition_program,groups_for,read_batch,fail_batch,batch_path
from .tape import build_tapes


def pending(work,cfg,c): return [g for g in groups_for(cfg,c) if read_batch(work,cfg,c,g) is None]


def summary(work,cfg):
    rows=[]
    for c in cfg['conditions']:
        source=next((x for x in cfg['conditions'] if x['id']==c.get('alias_of')),c)
        batches=[read_batch(work,cfg,source,g) for g in groups_for(cfg,source)]
        row={'id':c['id'],'n':c['n'],'alias_of':c.get('alias_of'),'batch_size':50,
             **{s:50*sum(r is not None and r['status']==s for r in batches) for s in ('complete','failed','skipped_condition_error')},
             'pending':50*sum(r is None for r in batches)}
        rows.append(row)
    return rows


def run_condition(repo,config_path,cfg,c,work):
    state=work/'worker_state.json'
    for restart in range(cfg['max_worker_restarts']+1):
        if not pending(work,cfg,c): return
        if shutil.disk_usage(work).free<cfg['minimum_free_bytes']: raise RuntimeError('Storage below 256 MiB reserve')
        if state.exists(): state.unlink()
        args=[sys.executable,'-u','-m','evomolsteer.path_replay.worker','--repo',str(repo),'--config',str(config_path),
              '--work',str(work),'--condition',c['id']]
        logpath=work/'logs'/(c['id']+'.worker.log');logpath.parent.mkdir(parents=True,exist_ok=True)
        active=None;started=time.monotonic();batch_started=started;timed_out=False
        with logpath.open('a',encoding='utf-8') as log:
            log.write(f'\nSTART restart={restart} unix={time.time()}\n');log.flush()
            child=subprocess.Popen(args,cwd=repo,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!='nt')
            try:
                while child.poll() is None:
                    try: info=read_json(state) if state.exists() else {}
                    except (OSError,json.JSONDecodeError): info={}
                    if info.get('condition')==c['id'] and info.get('group')!=active:
                        active=info.get('group');batch_started=time.monotonic()
                    limit=cfg['batch_timeout_seconds'] if active is not None else cfg['startup_timeout_seconds']
                    if time.monotonic()-(batch_started if active is not None else started)>limit:
                        timed_out=True;stop_child(child);break
                    if shutil.disk_usage(work).free<cfg['minimum_free_bytes']:
                        stop_child(child);raise RuntimeError('Storage below 256 MiB reserve; queue stopped without reducing B50')
                    time.sleep(cfg['poll_seconds'])
                code=child.wait()
            except BaseException: stop_child(child);raise
        left=pending(work,cfg,c)
        if not left: return
        error=f'Worker exited {code}; timeout={timed_out}; restart={restart}; group={active}'
        failed=next((g for g in left if g['index']==active),None)
        if failed is not None: fail_batch(work,cfg,c,failed,error,phase='worker_crash')
        if c['kind']=='unguided': raise RuntimeError('Zero-guidance replay failed; all guided conditions are blocked pending repair')
        if restart==cfg['max_worker_restarts'] or active is None:
            for g in pending(work,cfg,c): fail_batch(work,cfg,c,g,error,status='skipped_condition_error',phase='condition_worker')


def zero_gate(work,cfg):
    c=next(c for c in cfg['conditions'] if c['id']=='unguided');reports=[]
    for g in groups_for(cfg,c):
        r=read_batch(work,cfg,c,g)
        if r is None or r['status']!='complete': raise ValueError('All 200 zero-guidance paths must complete before guidance')
        q=read_json(batch_path(work,c,g).parent/'equivalence.json')
        if not q['passed']: raise ValueError('Historical replay equivalence gate failed')
        reports.append(q)
    out={'passed':True,'n':200,'protocol_sha256':cfg['protocol_sha256'],
         'max_intermediate_coordinate_error_A':max(r['max_coordinate_error_A'] for r in reports),
         'max_terminal_coordinate_error_A':max(r['terminal_max_coordinate_error_A'] for r in reports),
         'max_pic50_error':max(r['terminal_max_pic50_error'] for r in reports),
         'discrete_states_equal':True,'bitwise_coordinates':all(r['max_coordinate_error_A']==0 and r['terminal_max_coordinate_error_A']==0 for r in reports)}
    atomic_json(work/'zero_replay_gate.json',out);return out


def verify(repo,cfg,work):
    validate(cfg)
    if digest(cfg['checkpoint'])!=cfg['checkpoint_sha256']: raise ValueError('Shared FLOWR checkpoint differs')
    build_tapes(cfg,work)
    from .native import r15_equivalence
    result=r15_equivalence(repo,cfg)
    result['protocol_sha256']=cfg['protocol_sha256'];atomic_json(Path(work)/'r15_equivalence.json',result)
    return result


def run(repo,config_path,work,retry_failed=False):
    repo=Path(repo).resolve();config_path=Path(config_path).resolve();work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    cfg=read_json(config_path);validate(cfg)
    with exclusive_lock(work/'.queue.lock'):
        status={'status':'verifying','pid':os.getpid(),'protocol_sha256':cfg['protocol_sha256'],'started_unix':time.time()}
        atomic_json(work/'queue_status.json',status)
        try:
            verify(repo,cfg,work)
            if retry_failed:
                for c in cfg['conditions']:
                    if c.get('alias_of'): continue
                    for g in groups_for(cfg,c):
                        r=read_batch(work,cfg,c,g)
                        if r and r['status']!='complete':
                            path=batch_path(work,c,g).parent
                            target=work/'failed_history'/c['id']/f"group_{g['index']:03d}_{time.time_ns()}"
                            target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(path),str(target))
            aliases={}
            for c in cfg['conditions']:
                status.update(status='running',current=c['id'],conditions=summary(work,cfg),updated_unix=time.time())
                atomic_json(work/'queue_status.json',status)
                if c.get('alias_of'):
                    aliases[c['id']]={'source':c['alias_of'],'reason':'Identical operative program, teacher, B50 membership, initial states and primitive random tape',
                                      'numeric_program_sha256':c['numeric_program_sha256']}
                    atomic_json(work/'aliases.json',aliases);continue
                if c['kind']!='unguided': zero_gate(work,cfg)
                try: condition_program(repo,c)
                except Exception:
                    if c['kind']=='unguided': raise
                    for g in pending(work,cfg,c): fail_batch(work,cfg,c,g,traceback.format_exc(),status='skipped_condition_error',phase='frozen_reward')
                    continue
                try: run_condition(repo,config_path,cfg,c,work)
                except (KeyboardInterrupt,SystemExit): raise
                except Exception:
                    if c['kind']=='unguided' or shutil.disk_usage(work).free<cfg['minimum_free_bytes']: raise
                    for g in pending(work,cfg,c): fail_batch(work,cfg,c,g,traceback.format_exc(),status='skipped_condition_error',phase='condition_supervisor')
                if c['kind']=='unguided': zero_gate(work,cfg)
                from .analysis import analyze
                analyze(repo,cfg,work,bootstrap=False)
            from .analysis import analyze
            analyze(repo,cfg,work,bootstrap=True)
            status.update(current=None,conditions=summary(work,cfg),finished_unix=time.time())
            status['status']='complete_with_errors' if any(r['failed'] or r['skipped_condition_error'] for r in status['conditions']) else 'complete'
            atomic_json(work/'queue_status.json',status);return status
        except BaseException:
            status.update(status='blocked_shared_validation',error=traceback.format_exc(),conditions=summary(work,cfg),updated_unix=time.time())
            atomic_json(work/'queue_status.json',status);raise
