"""Durable CPU quality jobs; never wait for them in the GPU sampling process."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .io import read,write,frozen,sha,value_sha

WORKERS=2


def binding(folder,teacher):
    folder=Path(folder)
    return {'records':sha(folder/'records.json'),'sdf':sha(folder/'molecules_all_built.sdf'),
            'protein':sha(teacher['protein']),'teacher':value_sha(teacher)}


def _alive(pid):
    try:
        p=Path('/proc')/str(pid)
        return (p/'stat').read_text().split()[2]!='Z' and b'agenticgen.uniguide.quality_queue' in (p/'cmdline').read_bytes().split(b'\0')
    except (OSError,IndexError):return False


def ensure_worker(config):
    """Launch once, detached from the sampling command and its executor lock."""
    queue=Path(config['root'])/'quality_queue';queue.mkdir(parents=True,exist_ok=True)
    with (queue/'.launch.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        receipt=queue/'worker.json'
        if receipt.exists() and _alive(read(receipt)['pid']):return
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='')
        runtime={'root':config['root'],'plip_dependencies':config['plip_dependencies']}
        frozen(queue/'runtime.json',runtime)
        log=queue/f'worker_{time.time_ns()}.log'
        with log.open('xb') as stream:
            process=subprocess.Popen([sys.executable,'-m','agenticgen.uniguide.quality_queue',
                '--runtime',str(queue/'runtime.json')],stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,env=env,close_fds=True,start_new_session=True)
        write(receipt,{'pid':process.pid,'log':str(log),'started_unix':time.time(),'workers':WORKERS,'device':'cpu'})


def enqueue(config,folder,teacher):
    """Persist a full-population job and return only deterministic inline data.

    Advice sees the same basic metrics regardless of worker timing. Slow PLIP /
    PoseBusters outputs are separate diagnostics, never pending-as-pass flags.
    """
    from .quality import assess_light
    folder=Path(folder);root=Path(config['root'])
    if not (folder/'COMPLETE.json').exists():raise ValueError('Quality job requires completed inference')
    basic=assess_light(folder,teacher)
    request={'folder':str(folder),'teacher':teacher,'binding':binding(folder,teacher),
             'scope':'all_attempts','population':basic['attempts']}
    key=value_sha(str(folder.relative_to(root)))
    frozen(root/'quality_queue/requests'/f'{key}.json',request)
    try:ensure_worker(config)
    except Exception as error:
        write(root/'quality_queue/LAUNCH_FAILED.json',{'error':str(error),'unix':time.time()})
        print({'quality_worker_launch_failed':str(error),'inference_continues':True},flush=True)
    return basic


def snapshot(root,groups):
    rows=[]
    for group in groups:
        folder=Path(root)/group
        if (folder/'QUALITY_COMPLETE.json').exists():
            rows.append({'id':group,'status':'complete','receipt':str(folder/'QUALITY_COMPLETE.json'),
                         'sha256':sha(folder/'QUALITY_COMPLETE.json')})
        elif (folder/'QUALITY_FAILED.json').exists():
            rows.append({'id':group,'status':'failed','error':read(folder/'QUALITY_FAILED.json')})
        else:rows.append({'id':group,'status':'pending','receipt':str(folder/'QUALITY_COMPLETE.json')})
    return {'mode':'all_50_async_cpu','blocks_inference':False,
            'complete':sum(r['status']=='complete' for r in rows),
            'failed':sum(r['status']=='failed' for r in rows),
            'pending':sum(r['status']=='pending' for r in rows),'groups':rows}


def run_request(request_path):
    from .quality import assess
    request=read(request_path);folder=Path(request['folder'])
    with (folder/'.quality.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 'busy'
        try:
            if binding(folder,request['teacher'])!=request['binding']:
                raise ValueError('Quality job source changed after enqueue')
            write(folder/'QUALITY_RUNNING.json',{'pid':os.getpid(),'started_unix':time.time(),
                                               'scope':'all_attempts','population':request['population']})
            receipt=assess(folder,request['teacher'])
            if receipt['attempts']!=request['population']:raise ValueError('Quality denominator changed')
            return 'complete'
        except Exception as error:
            write(folder/'QUALITY_FAILED.json',{'error':str(error),'type':type(error).__name__,
                  'traceback':traceback.format_exc(),'unix':time.time(),'request':str(request_path),
                  'does_not_imply_molecular_failure':True,'inference_continues':True})
            return 'failed'


def worker(runtime,idle_seconds=900):
    sys.path.insert(0,runtime['plip_dependencies'])
    root=Path(runtime['root']);queue=root/'quality_queue'
    with (queue/'.worker.lock').open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        idle_since=time.monotonic();futures={}
        # Spawn avoids inheriting any CUDA/DTK context or PLIP global state.
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=multiprocessing.get_context('spawn')) as pool:
            while True:
                for future,path in list(futures.items()):
                    if future.done():
                        print({'request':str(path),'result':future.result()},flush=True)
                        del futures[future]
                pending=[]
                for path in sorted((queue/'requests').glob('*.json'),key=lambda p:(p.stat().st_mtime_ns,p.name)):
                    folder=Path(read(path)['folder'])
                    if not (folder/'QUALITY_COMPLETE.json').exists() and not (folder/'QUALITY_FAILED.json').exists():pending.append(path)
                for path in pending:
                    if len(futures)>=WORKERS:break
                    if path not in futures.values():futures[pool.submit(run_request,str(path))]=path
                write(queue/'status.json',{'pid':os.getpid(),'unix':time.time(),'active':[str(p) for p in futures.values()],
                    'pending':len(pending),'workers':WORKERS,'device':'cpu','blocks_inference':False})
                if pending or futures:idle_since=time.monotonic()
                elif time.monotonic()-idle_since>=idle_seconds:return
                time.sleep(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',required=True)
    worker(read(parser.parse_args().runtime))
