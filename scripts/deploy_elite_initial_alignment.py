"""Pull a pushed commit, freeze an offline runtime and detach the remote queue.

Password comes only from MOLSTEER_SCNET_PASSWORD. No credential is persisted.
Git fetch uses a temporary SSH reverse tunnel to the specified local VPN proxy;
the tunnel is closed after launch and is not needed by the scientific queue.
"""
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

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json

LIBRARIES = '/opt/miniforge3/envs/molsteer-flowr-dtk/lib:/opt/hyhal/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib'
PYTHON = '/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
REMOTE_REPO = '/root/private_data/MolSteer/EvoMolSteer'
DEPENDENCIES = '/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.', type=Path); p.add_argument('--config', required=True, type=Path)
    p.add_argument('--proxy-port', type=int, default=7897)
    p.add_argument('--host', default='ksai.scnet.cn'); p.add_argument('--port', type=int, default=50496)
    a = p.parse_args(); repo = a.repo.resolve(); cfg = read_json(a.config)
    password = os.environ['MOLSTEER_SCNET_PASSWORD']
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    rel = a.config.resolve().relative_to(repo).as_posix()
    paths = {'src', 'scripts/run_elite_initial_alignment_queue.py', rel}
    for c in cfg['conditions']:
        if c['kind'] != 'unguided': paths.update((c['program'], c['reference']))
    archive = subprocess.check_output(['git','archive','--format=tar',sha,*sorted(paths)], cwd=repo)
    archive_sha = hashlib.sha256(archive).hexdigest()
    work = cfg['remote_work']; runtime = work+'/runtime'
    expected_root = cfg['flowr_root']+'/experiments/evomolsteer_'
    if not work.startswith(expected_root) or '..' in work: raise ValueError('Unexpected remote workspace')
    c = paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(a.host, port=a.port, username='root', password=password, timeout=20, auth_timeout=25,
              look_for_keys=False, allow_agent=False)
    c.get_transport().set_keepalive(30)
    def command(text):
        _, out, err = c.exec_command(text, timeout=120)
        result = out.read().decode(); errors = err.read().decode(); code = out.channel.recv_exit_status()
        if code: raise RuntimeError(f'Remote command exited {code}: {errors[-1600:]} {result[-800:]}')
        return result
    def handler(channel, origin, server):
        def forward():
            import select
            upstream = socket.create_connection(('127.0.0.1', a.proxy_port), timeout=20)
            try:
                while True:
                    ready,_,_ = select.select([channel,upstream],[],[],30)
                    for source in ready:
                        data=source.recv(65536)
                        if not data:return
                        (upstream if source is channel else channel).sendall(data)
            finally:
                channel.close();upstream.close()
        threading.Thread(target=forward,daemon=True).start()
    try:
        proxy = c.get_transport().request_port_forward('127.0.0.1',0,handler=handler)
        # Network is needed only before the detached runtime exists.
        pull = '\n'.join(['set -eu', 'cd '+shlex.quote(REMOTE_REPO),
            f'git -c http.proxy=http://127.0.0.1:{proxy} -c http.version=HTTP/1.1 fetch origin main',
            'git merge --ff-only origin/main',f'test "$(git rev-parse HEAD)" = {sha}', 'git rev-parse HEAD'])
        print('remote_commit', command(pull).splitlines()[-1],flush=True)
        check = f"from pathlib import Path;import os;p=Path({work!r});p.mkdir(parents=True,exist_ok=True);q=p/'queue.pid';pid=int(q.read_text()) if q.exists() else 0;cmd=Path('/proc')/str(pid)/'cmdline';assert not (pid and cmd.exists() and b'run_elite_initial_alignment_queue' in cmd.read_bytes()),'Queue already running'"
        command('python -c '+shlex.quote(check))
        sftp = c.open_sftp()
        def mkdir(path):
            try:sftp.stat(path)
            except FileNotFoundError:
                mkdir(path.rsplit('/',1)[0]);sftp.mkdir(path)
        sftp.putfo(io.BytesIO(archive),work+'/runtime.tar')
        local_work = repo/'data/generated'/cfg['name']
        for initial in cfg['initials'].values():
            local = local_work/initial['file']; remote = work+'/'+initial['file']
            if digest(local)!=initial['sha256']:raise ValueError('Local initial changed')
            mkdir(remote.rsplit('/',1)[0]);sftp.put(str(local),remote)
        extract = f"""from pathlib import Path
import tarfile,hashlib,json
p=Path({work!r});a=p/'runtime.tar';assert hashlib.sha256(a.read_bytes()).hexdigest()=={archive_sha!r}
r=p/'runtime';r.mkdir(exist_ok=True)
with tarfile.open(a) as t:
 for m in t.getmembers():
  assert (r/m.name).resolve().is_relative_to(r.resolve()) and (m.isfile() or m.isdir())
 t.extractall(r,filter='data')
a.unlink()
(p/'deployment.json').write_text(json.dumps({{'commit':{sha!r},'archive_sha256':{archive_sha!r},'protocol_sha256':{cfg['protocol_sha256']!r}}},indent=2))
"""
        command('python -c '+shlex.quote(extract))
        # Dry-run does not import the GPU runtime; validates all staged artifacts.
        env = {'LD_LIBRARY_PATH': LIBRARIES, 'PYTHONPATH': runtime+'/src:'+DEPENDENCIES,
               'EVOMOLSTEER_RUNTIME_COMMIT': sha, 'PYTHONUNBUFFERED':'1', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'}
        args = [PYTHON,'-u',runtime+'/scripts/run_elite_initial_alignment_queue.py','--repo',runtime,
                '--config',runtime+'/'+rel,'--work',work,'--resume']
        dry = 'env '+' '.join(k+'='+shlex.quote(v) for k,v in env.items())+' '+' '.join(shlex.quote(x) for x in args+['--dry-run'])
        print(command(dry),flush=True)
        launch = f"""import os,subprocess,json
from pathlib import Path
p=Path({work!r});env=os.environ.copy();env.update({env!r})
with (p/'queue.log').open('ab',buffering=0) as log:
 child=subprocess.Popen({args!r},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
(p/'queue.pid').write_text(str(child.pid))
print(json.dumps({{'pid':child.pid,'work':str(p),'commit':{sha!r}}}))
"""
        receipt = json.loads(command('python -c '+shlex.quote(launch)))
        receipt.update(protocol_sha256=cfg['protocol_sha256'], deployed_unix=time.time(), host=a.host, port=a.port)
        atomic_json(repo/'docs/experiments'/cfg['name']/'launch_receipt.json',receipt)
        print(json.dumps(receipt,indent=2),flush=True)
        sftp.close()
    finally:
        c.close()


if __name__ == '__main__': main()
