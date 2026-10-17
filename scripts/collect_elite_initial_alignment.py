"""Collect an immutable receipt-based snapshot without stopping the remote queue."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import time

import paramiko

from evomolsteer.io import read_json, digest, safe_extract
from evomolsteer.storage.transactions import atomic_json
from evomolsteer.elite_alignment.common import validate_protocol, read_result
from evomolsteer.elite_alignment.report import report


def retire_absent_receipts(local, present, token):
    """A retried remote job may be pending while a local old failure still exists."""
    local=Path(local).resolve();present=set(present)
    for receipt in list((local/'methods').glob('*/*/result.json')):
        relative=receipt.relative_to(local).as_posix()
        if relative in present:continue
        source=receipt.parent.resolve();target=local/'retired_snapshots'/token/source.relative_to(local)
        if not source.is_relative_to(local) or not target.resolve().is_relative_to(local):
            raise ValueError('Snapshot retirement escaped local dataset')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(target))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.',type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=50496)
    a=p.parse_args();repo=a.repo.resolve();cfg=read_json(a.config);validate_protocol(cfg)
    c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(a.host,port=a.port,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],
              timeout=20,auth_timeout=25,look_for_keys=False,allow_agent=False)
    token=str(time.time_ns());remote=cfg['remote_work']
    script=f"""from pathlib import Path
import json,tarfile,hashlib
p=Path({remote!r});items=[]
for receipt in sorted((p/'methods').glob('*/*/result.json')):
 r=json.loads(receipt.read_text());items.append(receipt)
 for name,expected in r.get('artifacts',{{}}).items():
  f=receipt.parent/name;assert f.resolve().is_relative_to(p.resolve()) and not f.is_symlink()
  assert hashlib.sha256(f.read_bytes()).hexdigest()==expected;items.append(f)
items+=list((p/'methods').glob('*/source_audit.json'))
items+=[p/n for n in ['queue_status.json','worker_state.json','deployment.json','native_zero_equivalence.json'] if (p/n).exists()]
t=p/'transport';t.mkdir(exist_ok=True);out=t/{('collection_'+token+'.tar.gz')!r}
with tarfile.open(out,'w:gz') as archive:
 for f in items:archive.add(f,arcname=f.relative_to(p).as_posix(),recursive=False)
print(json.dumps({{'file':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(items),'bytes':out.stat().st_size,
 'receipt_paths':[f.relative_to(p).as_posix() for f in items if f.name=='result.json']}}))
"""
    try:
        _,out,err=c.exec_command('python -c '+shlex.quote(script),timeout=120)
        text=out.read().decode();error=err.read().decode()
        if out.channel.recv_exit_status():raise RuntimeError(error)
        snapshot=json.loads(text);archive=repo/'data/archives'/cfg['name']/Path(snapshot['file']).name
        archive.parent.mkdir(parents=True,exist_ok=True)
        sftp=c.open_sftp();sftp.get(snapshot['file'],str(archive))
        if digest(archive)!=snapshot['sha256']:raise ValueError('Downloaded snapshot checksum differs')
        local=repo/'data/generated'/cfg['name']
        retire_absent_receipts(local,snapshot['receipt_paths'],token)
        safe_extract(archive,local,expected_sha256=snapshot['sha256'])
        counts={};proof=[]
        for condition in cfg['conditions']:
            records=[read_result(local,cfg,condition,x) for x in cfg['attempts']]
            counts[condition['id']]={s:sum(r is not None and r['status']==s for r in records)
                for s in ('complete','failed','skipped_condition_error')}
            if condition['id'] in ('unguided','M2'):
                proof += [{k:r.get(k) for k in ('condition','status','pic50','valid_connected','qualified_structure','seconds','artifacts')}
                           | {'attempt':r['attempt']['id'],'rank':r['attempt']['rank']}
                           for r in records if r and r['status']=='complete']
        report(local,cfg,bootstrap=False)
        receipt={'collected_unix':time.time(),'protocol_sha256':cfg['protocol_sha256'],'snapshot':snapshot,
                 'local_archive':archive.relative_to(repo).as_posix(),'counts':counts,'control_examples':proof,
                 'queue':read_json(local/'queue_status.json'),'worker':read_json(local/'worker_state.json')}
        atomic_json(repo/'docs/experiments'/cfg['name']/'latest_collection.json',receipt)
        sftp.remove(snapshot['file']);sftp.close()
        print(json.dumps(receipt,ensure_ascii=False,indent=2))
    finally:c.close()


if __name__=='__main__':main()
