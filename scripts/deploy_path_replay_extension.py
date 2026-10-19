"""Deploy the independent R11 extension from pushed code, then detach its queue."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

from evomolsteer.io import read_json
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.path_replay.extension import load
from deploy_path_replay import connect,command,environment
from deploy_elite_initial_alignment import PYTHON,REMOTE_REPO


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['deploy','status','collect'])
    p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--proxy-port',type=int,default=7897);p.add_argument('--no-start',action='store_true');a=p.parse_args()
    repo=a.repo.resolve();cfg,ext,base=load(repo,a.manifest);rel=a.manifest.resolve().relative_to(repo).as_posix()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    work=ext['remote_work'];runtime=work+'/runtime';env=environment(runtime,sha)
    args=[PYTHON,'-u','-m','evomolsteer.path_replay.extension',a.action if a.action!='deploy' else 'run',
        '--repo',runtime,'--manifest',runtime+'/'+rel,'--work',work]
    prefix='env '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '
    client=connect()
    try:
        if a.action=='status':print(command(client,prefix+shlex.join(args)),flush=True);return
        if a.action=='collect':
            from evomolsteer.path_replay.collection import unpack_snapshot
            target=repo/'data/archives'/ext['name']/f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.tar.gz';target.parent.mkdir(parents=True,exist_ok=True)
            _,out,err=client.exec_command(prefix+shlex.join(args+['--output','-']))
            partial=target.with_suffix('.partial')
            with partial.open('wb') as f:
                while True:
                    block=out.read(1024*1024)
                    if not block:break
                    f.write(block)
            error=err.read().decode()
            if out.channel.recv_exit_status():raise RuntimeError(error)
            partial.replace(target);receipt=unpack_snapshot(target,repo/'data/generated'/ext['name'])
            receipt.update(archive=str(target),unix=time.time());atomic_json(repo/'docs/experiments'/ext['name']/'collection_receipt.json',receipt)
            print(json.dumps(receipt,indent=2),flush=True);return
        # An existing live queue owns this immutable runtime; never overwrite it.
        code=f"""from pathlib import Path
import json
p=Path({work!r});p.mkdir(parents=True,exist_ok=True)
f=p/'queue.pid'
if f.exists():
 q=Path('/proc')/f.read_text().strip()/'cmdline'
 assert not (q.exists() and b'evomolsteer.path_replay' in q.read_bytes()),'R11 extension already active'
"""
        command(client,'python -c '+shlex.quote(code))
        from remote_session import bridge
        def handler(channel,origin,server):threading.Thread(target=bridge,args=(channel,'127.0.0.1',a.proxy_port),daemon=True).start()
        port=client.get_transport().request_port_forward('127.0.0.1',0,handler=handler)
        pull='\n'.join(['set -eu','cd '+shlex.quote(REMOTE_REPO),
            f'git -c http.proxy=http://127.0.0.1:{port} -c http.version=HTTP/1.1 fetch origin main',
            'git merge --ff-only origin/main',f'test "$(git rev-parse HEAD)" = {sha}','git rev-parse HEAD'])
        print('remote_commit',command(client,pull).splitlines()[-1],flush=True)
        paths=['src',rel,ext['base_config'],ext['condition']['program'],ext['condition']['reference']]
        archive=subprocess.check_output(['git','archive','--format=tar',sha,*paths],cwd=repo)
        archive_sha=hashlib.sha256(archive).hexdigest()
        with client.open_sftp() as s:s.putfo(io.BytesIO(archive),work+'/runtime.tar')
        code=f"""from pathlib import Path
import tarfile,hashlib,json
p=Path({work!r});a=p/'runtime.tar';assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
runtime=p/'runtime';runtime.mkdir(exist_ok=True)
with tarfile.open(a) as t:
 for m in t.getmembers():assert (runtime/m.name).resolve().is_relative_to(runtime.resolve()) and (m.isfile() or m.isdir())
 t.extractall(runtime,filter='data')
a.unlink()
(p/'deployment.json').write_text(json.dumps({{'commit':{sha!r},'archive_sha256':{archive_sha!r},'extension_sha256':{ext['extension_sha256']!r},'base_protocol_sha256':{cfg['protocol_sha256']!r},'base_work':{ext['base_work']!r}}},indent=2))
"""
        command(client,'python -c '+shlex.quote(code))
        if a.no_start:print(json.dumps({'staged':True,'commit':sha,'work':work,'started':False}));return
        code=f"""from pathlib import Path
import subprocess,os,json
p=Path({work!r});env=os.environ.copy();env.update({env!r})
with (p/'queue.log').open('ab',buffering=0) as log:
 child=subprocess.Popen({args!r},cwd={runtime!r},env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(p/'queue.pid').write_text(str(child.pid))
print(json.dumps({{'pid':child.pid,'work':str(p),'commit':{sha!r},'extension_sha256':{ext['extension_sha256']!r}}}))
"""
        receipt=json.loads(command(client,'python -c '+shlex.quote(code)));receipt['unix']=time.time()
        atomic_json(repo/'docs/experiments'/ext['name']/'launch_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    finally:client.close()


if __name__=='__main__':main()
