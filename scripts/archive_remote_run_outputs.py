"""Archive explicit old-run binary outputs locally, verify, then optionally prune.

Authentication is read from temporary process environment, never saved. Scientific
inputs, teachers, reports, source, original Steer and replay dependencies survive.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tarfile
import time

from deploy_path_replay import connect, command

ROOT = '/root/private_data/MolSteer/flowr_root/experiments'
PROTECTED = {'ck2_clk3_lineage_20261003', 'evomolsteer_online_20261004',
             'evomolsteer_path_replay_b50_20261018'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    if any('/' in n or '\\' in n or n in PROTECTED or not n.startswith('evomolsteer_') for n in a.experiment):
        raise ValueError('Explicit derived experiment names only; protected inputs cannot be pruned')
    a.output.mkdir(parents=True, exist_ok=True)
    c = connect()
    try:
        code = f'''from pathlib import Path
import json,hashlib
root=Path({ROOT!r}).resolve(); rows=[]
for name in {a.experiment!r}:
 base=(root/name).resolve();assert base.parent==root and base.is_dir()
 for f in sorted(base.rglob('*')):
  if not f.is_file() or f.is_symlink():continue
  rel=f.relative_to(base)
  if any(x.lower() in ('inputs','teachers','teacher','reference','references','runtime','flowr_runtime','code','runtime_deps','tapes','seed_structures') for x in rel.parts[:-1]):continue
  if any(x in f.name.lower() for x in ('teacher','reference','initial_state','checkpoint')):continue
  if not f.name.endswith(('.sdf','.sdf.gz','.npz','.h5','.pt','.pt.gz','.tar.gz')):continue
  assert f.resolve().is_relative_to(base)
  rows.append({{'path':f.relative_to(root).as_posix(),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}})
active=[]
for d in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(d/'cmdline').read_bytes().replace(b'\\x00',b' ').decode(errors='replace')
  if any(name in cmd for name in {a.experiment!r}) and ('worker' in cmd or 'queue' in cmd) and 'python -c' not in cmd:active.append(d.name)
 except (OSError,PermissionError):pass
assert not active, 'Active run owns requested outputs: '+str(active)
print(json.dumps({{'root':str(root),'files':rows}}))
'''
        inventory = json.loads(command(c, 'python -c ' + shlex.quote(code)))
        (a.output/'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        paths = [r['path'] for r in inventory['files']]
        if not paths:
            print(json.dumps({'files':0,'deleted':False})); return
        archive = a.output/'old_outputs.tar.gz'
        # Stream the archive; no second archive consumes the remote quota.
        code = f'''import tarfile,sys
from pathlib import Path
root=Path({ROOT!r})
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in {paths!r}:t.add(root/name,arcname=name,recursive=False)
'''
        _, out, err = c.exec_command('python -c ' + shlex.quote(code))
        with archive.with_suffix('.partial').open('wb') as f:
            while True:
                block = out.read(1024*1024)
                if not block:break
                f.write(block)
        error=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(error)
        archive.with_suffix('.partial').replace(archive)
        expected={r['path']:r for r in inventory['files']}
        with tarfile.open(archive) as t:
            if set(t.getnames())!=set(expected):raise ValueError('Archive membership differs')
            for member in t:
                data=t.extractfile(member).read()
                if len(data)!=expected[member.name]['bytes'] or hashlib.sha256(data).hexdigest()!=expected[member.name]['sha256']:
                    raise ValueError('Archive checksum differs: '+member.name)
        deleted=[]
        if a.apply:
            code=f'''from pathlib import Path
import hashlib,json
root=Path({ROOT!r}).resolve();rows={inventory['files']!r}
for r in rows:
 f=root/r['path'];assert f.resolve().is_relative_to(root) and not f.is_symlink()
 assert f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
for r in rows:(root/r['path']).unlink()
print(json.dumps([r['path'] for r in rows]))
'''
            deleted=json.loads(command(c,'python -c '+shlex.quote(code)))
        receipt={'unix':time.time(),'files':len(paths),'bytes_archived':sum(r['bytes'] for r in inventory['files']),
                 'archive':str(archive.resolve()),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                 'deleted':deleted,'protection':'Original Steer, models, teachers, source, reports, replay tapes and shared dependencies retained.'}
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in receipt.items() if k!='deleted'},indent=2))
    finally:c.close()


if __name__=='__main__':main()
