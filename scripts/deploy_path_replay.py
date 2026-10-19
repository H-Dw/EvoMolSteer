"""Deploy pushed code through the local VPN and detach the historical replay queue."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time

import paramiko

from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.path_replay.common import validate
from deploy_elite_initial_alignment import LIBRARIES,PYTHON,REMOTE_REPO,DEPENDENCIES


def connect():
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(os.environ.get('MOLSTEER_SSH_HOST','ksai.scnet.cn'),
              port=int(os.environ.get('MOLSTEER_SSH_PORT','50496')),
              username=os.environ.get('MOLSTEER_SSH_USER','root'),password=os.environ['MOLSTEER_SCNET_PASSWORD'],
              timeout=25,auth_timeout=25,look_for_keys=False,allow_agent=False)
    c.get_transport().set_keepalive(30);return c


def command(c,text):
    _,out,err=c.exec_command(text,timeout=300)
    value=out.read().decode();error=err.read().decode();code=out.channel.recv_exit_status()
    if code: raise RuntimeError(f'Remote exit {code}: {error[-4000:]} {value[-1000:]}')
    return value


def environment(runtime,sha):
    return {'LD_LIBRARY_PATH':LIBRARIES,'PYTHONPATH':runtime+'/src:'+DEPENDENCIES,
            'EVOMOLSTEER_RUNTIME_COMMIT':sha,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['deploy','status','collect'])
    p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--config',type=Path,required=True)
    p.add_argument('--proxy-port',type=int,default=7897);p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--no-start',action='store_true',help='Stage the pushed runtime without launching a queue (diagnostic repairs).')
    a=p.parse_args();repo=a.repo.resolve();cfg=read_json(a.config);validate(cfg)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    rel=a.config.resolve().relative_to(repo).as_posix();work=cfg['remote_work'];runtime=work+'/runtime'
    args=[PYTHON,'-u','-m','evomolsteer.path_replay','run','--repo',runtime,'--config',runtime+'/'+rel,'--work',work]
    env=environment(runtime,sha);prefix='env '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '
    c=connect()
    try:
        if a.action=='status':
            cmd=prefix+' '.join(shlex.quote(v) for v in [*args[:4],'status',*args[5:]])
            print(command(c,cmd),flush=True);return
        if a.action=='collect':
            from evomolsteer.path_replay.collection import unpack_snapshot
            cmd=prefix+' '.join(shlex.quote(v) for v in [*args[:4],'collect',*args[5:],'--output','-'])
            target=repo/'data/archives'/cfg['name']/f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.tar.gz'
            target.parent.mkdir(parents=True,exist_ok=True)
            _,out,err=c.exec_command(cmd)
            with target.with_suffix('.partial').open('wb') as f:
                while True:
                    data=out.read(1024*1024)
                    if not data: break
                    f.write(data)
            error=err.read().decode()
            if out.channel.recv_exit_status(): raise RuntimeError(error)
            target.with_suffix('.partial').replace(target)
            receipt=unpack_snapshot(target,repo/'data/generated'/cfg['name'])
            receipt.update(archive=str(target),unix=time.time())
            atomic_json(repo/'docs/experiments'/cfg['name']/'collection_receipt.json',receipt)
            print(json.dumps(receipt,indent=2),flush=True);return
        from remote_session import bridge
        def handler(channel,origin,server):
            threading.Thread(target=bridge,args=(channel,'127.0.0.1',a.proxy_port),daemon=True).start()
        port=c.get_transport().request_port_forward('127.0.0.1',0,handler=handler)
        pull='\n'.join(['set -eu','cd '+shlex.quote(REMOTE_REPO),
            f'git -c http.proxy=http://127.0.0.1:{port} -c http.version=HTTP/1.1 fetch origin main',
            'git merge --ff-only origin/main',f'test "$(git rev-parse HEAD)" = {sha}','git rev-parse HEAD'])
        print('remote_commit',command(c,pull).splitlines()[-1],flush=True)
        paths={'src',rel}
        for condition in cfg['conditions']:
            if condition['kind']!='unguided': paths.update((condition['program'],condition['reference']))
        archive=subprocess.check_output(['git','archive','--format=tar',sha,*sorted(paths)],cwd=repo)
        archive_sha=hashlib.sha256(archive).hexdigest()
        code=f"""from pathlib import Path
import os
p=Path({work!r});p.mkdir(parents=True,exist_ok=True)
for name in ('queue.pid',):
 q=p/name
 if q.exists():
  pid=int(q.read_text());cmd=Path('/proc')/str(pid)/'cmdline'
  assert not (cmd.exists() and b'evomolsteer.path_replay' in cmd.read_bytes()),'Replay queue already active'
assert not Path('/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_elite_initial_alignment_20261017').exists(),'Old workspace must be backed up and retired first'
"""
        command(c,'python -c '+shlex.quote(code))
        with c.open_sftp() as s: s.putfo(io.BytesIO(archive),work+'/runtime.tar')
        code=f"""from pathlib import Path
import tarfile,hashlib,subprocess,json
p=Path({work!r});archive=p/'runtime.tar'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={archive_sha!r}
runtime=p/'runtime';runtime.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for m in t.getmembers(): assert (runtime/m.name).resolve().is_relative_to(runtime.resolve()) and (m.isfile() or m.isdir())
 t.extractall(runtime,filter='data')
archive.unlink()
flowr=p/'flowr_runtime';flowr.mkdir(exist_ok=True)
old=json.loads((p/'deployment.json').read_text()) if (p/'deployment.json').exists() else {{}}
reuse=old.get('flowr_commit')=={cfg['flowr_commit']!r} and all((flowr/name).is_file() and hashlib.sha256((flowr/name).read_bytes()).hexdigest()==sha for name,sha in {cfg['flowr_files']!r}.items())
if not reuse:
 proc=subprocess.Popen(['git','-C',{cfg['flowr_repository']!r},'archive','--format=tar',{cfg['flowr_commit']!r}],stdout=subprocess.PIPE)
 with tarfile.open(fileobj=proc.stdout,mode='r|') as t:
  for m in t:
   assert (flowr/m.name).resolve().is_relative_to(flowr.resolve()) and (m.isfile() or m.isdir())
   t.extract(m,flowr,filter='data')
 assert proc.wait()==0
for name,sha in {cfg['flowr_files']!r}.items(): assert hashlib.sha256((flowr/name).read_bytes()).hexdigest()==sha,name
(p/'deployment.json').write_text(json.dumps({{'commit':{sha!r},'archive_sha256':{archive_sha!r},'protocol_sha256':{cfg['protocol_sha256']!r},'flowr_commit':{cfg['flowr_commit']!r},'unchanged_flowr_runtime_reused':reuse}},indent=2))
"""
        command(c,'python -c '+shlex.quote(code))
        if a.no_start:
            print(json.dumps({'staged':True,'commit':sha,'runtime':runtime,'started':False}),flush=True);return
        if a.retry_failed: args.append('--retry-failed')
        code=f"""from pathlib import Path
import subprocess,os,json
p=Path({work!r});env=os.environ.copy();env.update({env!r})
with (p/'queue.log').open('ab',buffering=0) as log:
 child=subprocess.Popen({args!r},cwd={runtime!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(p/'queue.pid').write_text(str(child.pid))
print(json.dumps({{'pid':child.pid,'work':str(p),'commit':{sha!r},'protocol_sha256':{cfg['protocol_sha256']!r}}}))
"""
        receipt=json.loads(command(c,'python -c '+shlex.quote(code)));receipt['unix']=time.time()
        atomic_json(repo/'docs/experiments'/cfg['name']/'launch_receipt.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True)
    finally: c.close()


if __name__=='__main__': main()
