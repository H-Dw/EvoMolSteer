"""Deploy a committed local-feature decision; remote runs inference, never LLMs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

from evomolsteer.path_replay.local_research import load
from evomolsteer.storage.transactions import atomic_json
from deploy_path_replay import connect,command,environment
from deploy_elite_initial_alignment import PYTHON,REMOTE_REPO


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['deploy','status','collect'])
    p.add_argument('--repo',type=Path,default=Path('.'));p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--proxy-port',type=int,default=7897);a=p.parse_args()
    repo=a.repo.resolve();cfg,m,base=load(repo,a.manifest);rel=a.manifest.resolve().relative_to(repo).as_posix()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();work=m['remote_work'];runtime=work+'/runtime'
    env=environment(runtime,sha);prefix='env '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '
    args=[PYTHON,'-u','-m','evomolsteer.path_replay.local_research','run' if a.action=='deploy' else a.action,
          '--repo',runtime,'--manifest',runtime+'/'+rel,'--work',work]
    client=connect()
    try:
        if a.action=='status':print(command(client,prefix+shlex.join(args)));return
        local=repo/'data/generated'/m['name'];doc=repo/'docs/experiments'/m['name']
        if a.action=='collect':
            from evomolsteer.path_replay.collection import unpack_snapshot
            archive=repo/'data/archives'/m['name']/f'snapshot_{time.strftime("%Y%m%d_%H%M%S")}.tar.gz';archive.parent.mkdir(parents=True,exist_ok=True)
            _,out,err=client.exec_command(prefix+shlex.join(args))
            with archive.with_suffix('.partial').open('wb') as f:
                while True:
                    block=out.read(1024*1024)
                    if not block:break
                    f.write(block)
            error=err.read().decode()
            if out.channel.recv_exit_status():raise RuntimeError(error)
            archive.with_suffix('.partial').replace(archive)
            receipt=unpack_snapshot(archive,local);receipt.update(archive=str(archive),unix=time.time())
            atomic_json(doc/'collection_receipt.json',receipt);print(json.dumps(receipt,indent=2));return
        code=f'''from pathlib import Path
import json
p=Path({work!r});p.mkdir(parents=True,exist_ok=True)
for name in ('queue.pid','worker.pid'):
 f=p/name
 if f.exists():
  q=Path('/proc')/f.read_text().strip()/'cmdline'
  assert not (q.exists() and b'evomolsteer.path_replay' in q.read_bytes()),'Research runtime already active'
'''
        command(client,'python -c '+shlex.quote(code))
        from remote_session import bridge
        def handler(channel,origin,server):threading.Thread(target=bridge,args=(channel,'127.0.0.1',a.proxy_port),daemon=True).start()
        port=client.get_transport().request_port_forward('127.0.0.1',0,handler=handler)
        pull='\n'.join(['set -eu','cd '+shlex.quote(REMOTE_REPO),
            f'git -c http.proxy=http://127.0.0.1:{port} -c http.version=HTTP/1.1 fetch origin main',
            'git merge --ff-only origin/main',f'test "$(git rev-parse HEAD)" = {sha}'])
        command(client,pull)
        paths=['src',rel,m['base_config']]
        if m.get('diagnostic_evidence'):paths.append(m['diagnostic_evidence']['path'])
        for c in m['conditions']:
            for key in ('program','reference'):
                if key in c:paths.append(c[key])
        todo=[c['reuse_current_from']['manifest'] for c in m['conditions'] if c.get('reuse_current_from')]
        seen=set()
        while todo:
            relative=todo.pop()
            if relative in seen:continue
            seen.add(relative);paths.append(relative)
            _,source,_=load(repo,repo/relative);paths.append(source['base_config'])
            if source.get('diagnostic_evidence'):paths.append(source['diagnostic_evidence']['path'])
            for c in source['conditions']:
                for key in ('program','reference'):
                    if key in c:paths.append(c[key])
                if c.get('reuse_current_from'):todo.append(c['reuse_current_from']['manifest'])
        archive=subprocess.check_output(['git','archive','--format=tar',sha,*sorted(set(paths))],cwd=repo)
        archive_sha=hashlib.sha256(archive).hexdigest()
        with client.open_sftp() as s:s.putfo(io.BytesIO(archive),work+'/runtime.tar')
        code=f'''from pathlib import Path
import tarfile,hashlib,json,subprocess,os
p=Path({work!r});a=p/'runtime.tar';assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
runtime=p/'runtime';runtime.mkdir(exist_ok=True)
with tarfile.open(a) as t:
 for item in t.getmembers():assert (runtime/item.name).resolve().is_relative_to(runtime.resolve()) and (item.isfile() or item.isdir())
 t.extractall(runtime,filter='data')
a.unlink()
(p/'deployment.json').write_text(json.dumps({{'commit':{sha!r},'archive_sha256':{archive_sha!r},'research_sha256':{m['research_sha256']!r}}},indent=2))
env=os.environ.copy();env.update({env!r})
with (p/'queue.log').open('ab',buffering=0) as f:
 child=subprocess.Popen({args!r},cwd=str(runtime),env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(p/'queue.pid').write_text(str(child.pid));print(json.dumps({{'pid':child.pid,'commit':{sha!r},'work':str(p)}}))
'''
        receipt=json.loads(command(client,'python -c '+shlex.quote(code)));receipt['unix']=time.time()
        atomic_json(doc/'launch_receipt.json',receipt);print(json.dumps(receipt,indent=2))
    finally:client.close()


if __name__=='__main__':main()
