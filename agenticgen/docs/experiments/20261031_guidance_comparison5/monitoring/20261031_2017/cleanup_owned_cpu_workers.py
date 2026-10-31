"""Stop only idle, source-bound CPU quality workers of the finalized comparison."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,time
ROOT=Path('/workspace/agenticgen_guidance_comparison5_20261031')
AUDIT=Path('/root/private_data/uniguide_comparison5_20261031/audits/comparison5-072e6f78')
COMMIT='072e6f7831bd91b398f7e1cbd65f4725268ec9f4'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def lock(p):
 if not p.exists():return 'absent'
 with p.open('r+')as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return 'held'
  fcntl.flock(f,fcntl.LOCK_UN);return 'free'
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  stat=(p/'stat').read_text().split();argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0')if x];status=dict(line.split(':',1)for line in (p/'status').read_text().splitlines()if ':'in line)
  return dict(pid=pid,state=stat[2],ppid=int(stat[3]),pgid=int(stat[4]),sid=int(stat[5]),starttime=stat[21],argv=argv,cwd=str((p/'cwd').resolve()),sigint_ignored=bool(int(status['SigIgn'].strip(),16)&2))
 except (OSError,ValueError,IndexError):return None
def active(p):return p and p['state']!='Z'
def processes():return [x for p in Path('/proc').glob('[0-9]*')if (x:=proc(int(p.name)))]
def save(name,d):
 path=ROOT/'monitoring/20261031_2017'/name
 with path.open('x')as f:json.dump(d,f,ensure_ascii=True,indent=2);f.write('\n')
 return path
args=argparse.ArgumentParser();args.add_argument('--stop',action='store_true');stop=args.parse_args().stop
plan=read(ROOT/'PLAN.json');repo=Path(plan['repo']);assert str(ROOT)==plan['root']and plan['commit']==COMMIT
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()==COMMIT
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no']).decode().strip()
assert (ROOT/'SUITE_FINISHED.json').exists()and (ROOT/'ALL_JOBS_ATTEMPTED.json').exists()
review=AUDIT/'actual_gpt_final_review.zh-CN.md';binding=read(AUDIT/'GPT_REVIEW_BINDING.json');assert sha(review)==binding['review_sha256']and binding['review_commit_pushed']and binding['science_commit']==COMMIT
assert lock(ROOT/'.suite.lock')=='free'and lock(Path(plan['global_executor_lock']))=='free'
ps=processes();sampling=[]
for p in ps:
 a=p['argv']
 if not active(p)or not a:continue
 if len(a)>1 and Path(a[1]).name in ['run_suite.py','run_uniguide_geometry.py']and (str(ROOT)in ' '.join(a)or str(repo)in ' '.join(a)):sampling.append(p)
 if len(a)>2 and a[1]=='-m'and a[2]=='agenticgen.uniguide':sampling.append(p)
 if len(a)>2 and a[1]=='-i'and a[2]==str(repo/'agenticgen/docs/experiments/20261031_guidance_comparison5/launch.sh'):sampling.append(p)
assert not sampling,sampling
workers=[];empty=[]
for job in plan['jobs']:
 config=read(Path(job['config']));root=Path(config['root']);assert root==ROOT/'jobs'/job['id']/'attempt_001';queue=root/'quality_queue'
 assert lock(root/'.executor.lock')in ['free','absent']and lock(root/'.campaign.lock')in ['free','absent']
 requests=[]
 for path in sorted((queue/'requests').glob('*.json')):
  rq=read(path);folder=Path(rq['folder']);assert folder.is_relative_to(root)and rq['population']==50
  assert (folder/'COMPLETE.json').exists()and (folder/'QUALITY_COMPLETE.json').exists()and not (folder/'QUALITY_FAILED.json').exists()
  receipt=read(folder/'QUALITY_COMPLETE.json');assert receipt['attempts']==50 and receipt['checked_attempts']==10 and receipt['not_selected']==40 and receipt['coverage']==0.2
  requests.append(dict(path=str(path),sha256=sha(path),source=str(folder),quality_complete_sha256=sha(folder/'QUALITY_COMPLETE.json')))
 if not (queue/'worker.json').exists():
  assert not requests;empty.append(job['id']);continue
 w=read(queue/'worker.json');b=read(root/'QUALITY_MODULE_BINDING.json');s=read(queue/'status.json');runtime=read(queue/'runtime.json');assert runtime['root']==str(root)
 assert b['commit']==COMMIT and b['module_directory']==w['module_directory']==str(repo/'agenticgen/src/agenticgen/uniguide')
 assert b['implementation_sha256']==w['implementation_sha256']
 for name,digest in b['implementation_sha256'].items():assert sha(Path(b['module_directory'])/name)==digest
 assert not s['active']and s['pending']==0 and s['pid']==w['pid']and time.time()-s['unix']<15
 pr=proc(w['pid']);assert active(pr)and pr['argv']==['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python','-m','agenticgen.uniguide.quality_queue','--runtime',str(queue/'runtime.json')]
 assert pr['cwd']==str(repo)and pr['sid']==pr['pid']==pr['pgid']and lock(queue/'.worker.lock')=='held'
 environ=dict(item.split(b'=',1)for item in (Path('/proc')/str(pr['pid'])/'environ').read_bytes().split(b'\0')if b'='in item)
 safeenv={k:environ.get(k.encode(),b'<absent>').decode()for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
 assert all(safeenv[k]==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and all(safeenv[k]=='1'for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
 members=[p for p in ps if p['sid']==pr['sid']and active(p)];assert all(p['pid']==pr['pid']or (p['cwd']==str(repo)and 'multiprocessing.'in ' '.join(p['argv']))for p in members)
 workers.append(dict(job=job['id'],root=str(root),queue=str(queue),worker=pr,owned_session_members=members,safe_environment=safeenv,status=s,binding=b,requests=requests,worker_log=w['log']))
assert len(workers)==5 and len(empty)==1 and sum(len(w['requests'])for w in workers)==47
record={'timestamp':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'science_commit':COMMIT,'review_commit':binding['review_commit'],'review_sha256':binding['review_sha256'],'workers':workers,'jobs_without_worker':empty,'active_sampling_before':sampling,'stop_requested':stop,'no_runtime_code_or_claims_modified':True}
if not stop:
 path=save('worker_cleanup_preflight.json',record);print(json.dumps({'preflight':str(path),'workers':len(workers),'requests_complete':47,'signals_not_sent':True}));raise SystemExit
path=save('worker_cleanup_before.json',record);signals=[]
for w in workers:
 pr=w['worker'];current=proc(pr['pid']);assert current and current['starttime']==pr['starttime']and current['argv']==pr['argv']
 if pr['sigint_ignored']:
  targets=w['owned_session_members'];sig=signal.SIGTERM
 else:targets=[pr];sig=signal.SIGINT
 for p in targets:
  current=proc(p['pid'])
  if active(current):
   assert current['starttime']==p['starttime']and current['sid']==pr['sid'];os.kill(p['pid'],sig);signals.append(dict(pid=p['pid'],signal=sig.name,job=w['job']))
end=time.monotonic()+15
while time.monotonic()<end:
 if all(not active(proc(p['pid']))for w in workers for p in w['owned_session_members']):break
 time.sleep(.25)
remaining=[(w,p)for w in workers for p in w['owned_session_members']if active(proc(p['pid']))]
for w,p in remaining:
 current=proc(p['pid']);assert current['starttime']==p['starttime']and current['sid']==w['worker']['sid'];os.kill(p['pid'],signal.SIGTERM);signals.append(dict(pid=p['pid'],signal='SIGTERM',job=w['job'],reason='owned_idle_session_still_active_after_interrupt'))
if remaining:time.sleep(2)
after=[]
for w in workers:
 members=[proc(p['pid'])for p in w['owned_session_members']];assert not any(active(p)for p in members),members
 workerlock=lock(Path(w['queue'])/'.worker.lock');assert workerlock=='free';after.append(dict(job=w['job'],worker_lock=workerlock,processes_after=members))
record.update(completed_timestamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),signals=signals,after=after,all_owned_cpu_workers_stopped=True,global_executor_lock=lock(Path(plan['global_executor_lock'])),suite_lock=lock(ROOT/'.suite.lock'),new_sampling_or_provider_calls=False)
path=save('worker_cleanup_complete.json',record);print(json.dumps({'receipt':str(path),'workers_stopped':len(after),'signals':signals,'worker_locks_released':True,'sampling_started':False}))
