"""Back up and retire exactly the superseded alignment workspace.

The archive is streamed to the local machine, completely checked against a
remote SHA256 inventory, and only then is the explicitly named directory
removed. Credentials are read from the process environment and never saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile
import time

import paramiko

OLD = '/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_elite_initial_alignment_20261017'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--retire', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    c = paramiko.SSHClient(); c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('ksai.scnet.cn', port=50496, username='root',
              password=os.environ['MOLSTEER_SCNET_PASSWORD'], look_for_keys=False,
              allow_agent=False, timeout=25, auth_timeout=25)
    c.get_transport().set_keepalive(20)

    def command(code):
        _, out, err = c.exec_command('python -c '+shlex.quote(code))
        value=out.read().decode(); errors=err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(errors)
        return value

    try:
        # Stop only processes whose exact command line refers to this workspace.
        stop = f"""from pathlib import Path
import os,signal,time,json
p=Path({OLD!r}); assert p.resolve()==p and not p.is_symlink()
assert p.parent==Path('/root/private_data/MolSteer/flowr_root/experiments')
matches=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit() or int(q.name)==os.getpid(): continue
 try: parts=(q/'cmdline').read_bytes().decode(errors='replace').split('\\0');cmd=' '.join(parts)
 except OSError: continue
 if {OLD!r} in cmd and ('evomolsteer.elite_alignment.worker' in parts or any(v.endswith('/run_elite_initial_alignment_queue.py') for v in parts)):
  matches.append(int(q.name))
for pid in matches:
 try: os.kill(pid,signal.SIGTERM)
 except ProcessLookupError: pass
time.sleep(2 if matches else 0)
for pid in matches:
 try: os.kill(pid,signal.SIGKILL)
 except ProcessLookupError: pass
print(json.dumps({{'stopped_pids':matches,'path':str(p),'exists':p.exists()}}))
"""
        print(command(stop), flush=True)
        inventory = f"""from pathlib import Path
import hashlib,json
p=Path({OLD!r}); rows={{}}
for q in sorted(p.rglob('*')):
 assert not q.is_symlink(),'Symlink needs explicit archival handling: '+str(q)
 if not q.is_file(): continue
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 rows[q.relative_to(p).as_posix()]={{'size':q.stat().st_size,'sha256':h.hexdigest()}}
print(json.dumps(rows,sort_keys=True))
"""
        manifest=json.loads(command(inventory))
        if not manifest: raise RuntimeError('Old workspace is absent or empty; nothing retired')
        (a.output/'inventory.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print('inventory_files',len(manifest),flush=True)
        partial=a.output/'complete_old_workspace.tar.gz.partial'
        _, out, err=c.exec_command('tar -C '+shlex.quote(str(Path(OLD).parent).replace('\\','/'))+
                                  ' -czf - '+shlex.quote(OLD.rsplit('/',1)[1]))
        h=hashlib.sha256()
        with partial.open('wb') as f:
            while True:
                b=out.read(1024*1024)
                if not b: break
                f.write(b);h.update(b)
        errors=err.read().decode()
        if out.channel.recv_exit_status(): raise RuntimeError(errors)
        checked={}; prefix=OLD.rsplit('/',1)[1]+'/'
        with tarfile.open(partial,'r:gz') as archive:
            for member in archive:
                if member.isdir(): continue
                if not member.isfile() or not member.name.startswith(prefix):
                    raise RuntimeError('Unexpected archive member: '+member.name)
                key=member.name[len(prefix):]; f=archive.extractfile(member); mh=hashlib.sha256()
                for b in iter(lambda:f.read(1024*1024),b''): mh.update(b)
                checked[key]={'size':member.size,'sha256':mh.hexdigest()}
        if checked != manifest: raise RuntimeError('Complete archive differs from remote inventory')
        final=partial.with_suffix(''); partial.replace(final)
        receipt={'source':OLD,'archive':str(final.resolve()),'archive_sha256':h.hexdigest(),
                 'files':len(checked),'bytes':sum(x['size'] for x in checked.values()),
                 'verified':True,'retired':False,'unix':time.time()}
        if a.retire:
            # Re-hash immediately before deletion; concurrent changes abort retirement.
            if json.loads(command(inventory)) != manifest:
                raise RuntimeError('Old workspace changed after backup; refusing retirement')
            code=f"""from pathlib import Path
import shutil
p=Path({OLD!r});assert p.resolve()==p and not p.is_symlink()
assert p.parent==Path('/root/private_data/MolSteer/flowr_root/experiments')
shutil.rmtree(p)
print('RETIRED_VERIFIED_OLD_WORKSPACE')
"""
            print(command(code),flush=True);receipt['retired']=True
        (a.output/'backup_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt,indent=2),flush=True)
    finally: c.close()


if __name__=='__main__': main()
