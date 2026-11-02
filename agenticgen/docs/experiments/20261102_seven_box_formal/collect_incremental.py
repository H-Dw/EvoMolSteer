"""One-pass incremental collection of immutable outputs in the authorized suite."""
import argparse,hashlib,io,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone,timedelta

ROOT=Path('/workspace/agenticgen_seven_box_search_20261102')
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=True,indent=2)
IN_PROGRESS_QUALITY=[]
def safe(p):
 return (p.is_file() and not p.is_symlink() and 'launcher.log' not in p.name.lower()
  and not p.name.lower().startswith('core') and '__pycache__' not in p.parts
  and not p.name.endswith(('.lock','.tar.gz','.ckpt','.safetensors'))
  and 'quality_queue' not in p.parts and p.name!='QUALITY_RUNNING.json'
  and not any(base in p.parents for base in IN_PROGRESS_QUALITY)
  and (p.name!='quality.json' or (p.parent/'QUALITY_COMPLETE.json').exists() or (p.parent/'QUALITY_FAILED.json').exists()))

parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--previous');parser.add_argument('--inspection',required=True);parser.add_argument('--inventory-only',action='store_true')
a=parser.parse_args();out=Path(a.out)
assert out.parent==Path('/workspace/uniguide_audits') and out.name.startswith('seven-box-monitor-')
out.mkdir(parents=True,exist_ok=True)
write(out/'COLLECT_CLAIM.json',dict(timestamp=now(),source=str(ROOT),pid=os.getpid(),collector_sha256=sha(Path(__file__))))
previous=read(a.previous) if a.previous else dict(verified_locally=True,files={},previous_archive_sha256=None)
assert previous['verified_locally']
plan=read(ROOT/'PLAN.json');inspection=read(a.inspection)
assert inspection['actual_commit']==plan['commit']=='c5166bd1d250ea01aa2864cdebeafc6f4174fe35'
assert not inspection['runtime_source_changed'] and not inspection['protected_files_changed']
files=set();snapshots={};selected=[]
for job in plan['jobs']:
 files.add(Path(job['config']))
 r=Path(job['root']);result=read(ROOT/'job_results'/(job['id']+'.json')) if (ROOT/'job_results'/(job['id']+'.json')).exists() else None
 sealed=[p.parent for p in r.glob('**/COMPLETE.json')]
 for folder in sealed:
  if (folder/'records.json').exists() and not (folder/'QUALITY_COMPLETE.json').exists() and not (folder/'QUALITY_FAILED.json').exists():
   IN_PROGRESS_QUALITY.append(folder/'profiles')
 for marker in r.glob('box_exclusion/*/SELECTION_FROZEN.json'):sealed.append(marker.parent)
 # Completed round advice and the final explicit analysis are immutable too.
 sealed += [p.parent for p in r.glob('versions/**/ROUND_COMPLETE.json')]
 sealed += [p.parent for p in r.glob('versions/**/synthesis.receipt.json') if p.parent.name=='analysis_final']
 quality=[]
 for q in (r/'quality_queue/requests').glob('*.json'):
  request=read(q);f=Path(request['folder'])
  quality.append(dict(request=str(q),source=str(f),population=request['population'],scope=request['scope'],
    complete=(f/'QUALITY_COMPLETE.json').exists(),failed=(f/'QUALITY_FAILED.json').exists()))
 if result and result['status']=='completed' and all(q['complete'] or q['failed'] for q in quality):
  assert (r/'SAMPLING_FINISHED.json').exists();sealed=[r]
 selected.extend(str(p) for p in sealed)
 for folder in sorted(sealed,key=lambda p:len(p.parts)):
  files.update(p for p in folder.rglob('*') if safe(p))
 meta={}
 for p in [r/'attempts.json',r/'SUBJECTS.json',r/'QUALITY_MODULE_BINDING.json',r/'quality_queue/worker.json',r/'quality_queue/status.json',
   *r.glob('calibration/**/SEARCH_STATE.json'),*r.glob('calibration/**/domain.json'),*r.glob('calibration/**/TEACHER_STATUS.json')]:
  if p.exists():meta[str(p.relative_to(r))]=read(p)
 snapshots[job['id']]=dict(metadata=meta,result=result,quality=quality)
 # Only stage logs with exit receipts are closed; never include launcher logs.
 for exit_file in (ROOT/'logs'/job['id']).glob('*_EXIT.json'):
  log=exit_file.with_name(exit_file.stem[:-5].lower()+'.log')
  files.add(exit_file)
  if safe(log):files.add(log)
snapshot=out/'ACCOUNTING_SNAPSHOT.json';write(snapshot,dict(timestamp=now(),mutable_source_snapshot_only=True,jobs=snapshots))
files.update([snapshot,Path(a.inspection),ROOT/'PLAN.json',ROOT/'SOURCE_PRESERVATION.json'])
audit=Path('/root/private_data/uniguide_seven_box_search_20261102/audits/seven-box-search-c5166bd1')
for name in ('DEPLOY_PROCESS.json','DEPLOYMENT.json','REGRESSION.json','READY.json','LAUNCH.json','INTERACTIVE_ENVIRONMENT.json','regression.log','regression.xml'):
 p=audit/name
 if safe(p):files.add(p)
before={};skipped=[]
for p in sorted(files):
 s=p.stat();digest=sha(p);name=p.as_posix().lstrip('/')
 if previous['files'].get(name)==digest:skipped.append(dict(path=name,sha256=digest,bytes=s.st_size))
 else:before[p]=(s.st_size,s.st_mtime_ns,digest)
skipfile=out/'PREVIOUS_ARCHIVE_REFERENCES.json'
write(skipfile,dict(previous_archive_sha256=previous['previous_archive_sha256'],previous_manifest_sha256=sha(Path(a.previous)) if a.previous else None,files=skipped))
s=skipfile.stat();before[skipfile]=(s.st_size,s.st_mtime_ns,sha(skipfile))
inventory=dict(timestamp=now(),members=len(before),source_bytes=sum(row[0] for row in before.values()),
    hdf5_members=sum(p.name.endswith('.h5') for p in before),skipped_previous_files=len(skipped),
    excluded_live_quality_trees=len(IN_PROGRESS_QUALITY),scientific_processes_touched=False)
write(out/'INVENTORY.json',inventory)
if a.inventory_only:
 print(json.dumps(inventory,ensure_ascii=True));raise SystemExit(0)
archive=out/'completed_increment.tar.gz';content={};manifest=[]
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for p,(size,mtime,digest) in before.items():
  name=p.as_posix().lstrip('/');item=tarfile.TarInfo(name);item.size=size;item.mtime=mtime//10**9;item.mode=0o600
  if digest in content:
   item.type=tarfile.LNKTYPE;item.linkname=content[digest];item.size=0;tar.addfile(item)
  else:
   with p.open('rb') as stream:tar.addfile(item,stream)
   content[digest]=name
  manifest.append(dict(path=name,bytes=size,sha256=digest,source_mtime_ns=mtime))
 for p,(size,mtime,digest) in before.items():
  st=p.stat()
  if (st.st_size,st.st_mtime_ns)!=(size,mtime) or sha(p)!=digest:raise RuntimeError('Source changed: '+str(p))
 data=(json.dumps(manifest,ensure_ascii=True,indent=2)+'\n').encode();item=tarfile.TarInfo('MANIFEST.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
verified={};saved_manifest=None
with tarfile.open(archive,'r|gz') as tar:
 for member in tar:
  if member.name=='MANIFEST.json':saved_manifest=json.load(tar.extractfile(member));continue
  if member.islnk():verified[member.name]=verified[member.linkname]
  else:
   assert member.isfile();h=hashlib.sha256();count=0
   with tar.extractfile(member) as stream:
    for b in iter(lambda:stream.read(8<<20),b''):h.update(b);count+=len(b)
   verified[member.name]=(count,h.hexdigest())
assert saved_manifest==manifest and len(verified)==len(manifest)
for row in manifest:assert verified[row['path']]==(row['bytes'],row['sha256'])
write(out/'MANIFEST.json',manifest)
receipt=dict(timestamp=now(),archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
 members=len(manifest),hdf5_members=sum(p['path'].endswith('.h5') for p in manifest),
 previous_archive_sha256=previous['previous_archive_sha256'],previously_verified_files_referenced=len(skipped),
 selected_sealed_trees=len(selected),all_source_hashes_unchanged=True,all_members_verified=True,
 verification_method='single_pass_with_verified_hardlink_targets',launcher_core_weights_included=False,
 scientific_processes_touched=False,scientific_attempts_started=0,llm_calls=0)
write(out/'COLLECT_COMPLETE.json',receipt);print(json.dumps(receipt,ensure_ascii=True))
