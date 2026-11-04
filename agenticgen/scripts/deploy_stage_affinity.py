"""Deploy only a GitHub-published commit. SSH credential is process environment only."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import paramiko

BOOTSTRAP=r'''
import json,os,pathlib,subprocess,sys,time
from hashlib import sha256
R=pathlib.Path('/workspace/agenticgen_stage_affinity_campaign_20261104')
commit=sys.argv[1];repo=pathlib.Path('/workspace/uniguide_runs/stage-affinity-'+commit[:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return sha256(pathlib.Path(p).read_bytes()).hexdigest()
if R.exists() or repo.exists():raise RuntimeError('Existing campaign/checkout must be preserved')
R.mkdir(parents=True)
try:
 with (R/'git_fetch.log').open('wb') as log:
  result=subprocess.run(['git','clone','--no-checkout','--single-branch','--branch','codex/uniguide-flowr-agent','https://github.com/H-Dw/EvoMolSteer.git',str(repo)],stdout=log,stderr=subprocess.STDOUT,timeout=60,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'))
  if result.returncode:raise RuntimeError('GitHub clone failed')
except Exception as e:
 write(R/'GITHUB_FETCH_FAILED.json',dict(type=type(e).__name__,error=str(e),bundle_is_published_commit=True))
 if repo.exists():repo.rename(repo.with_name(repo.name+'.github_failed'))
 subprocess.run(['git','clone','--no-checkout','/workspace/stage-affinity-'+commit[:12]+'.bundle',str(repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['git','-C',str(repo),'checkout','--detach',commit],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
if actual!=commit:raise RuntimeError('Deployed commit differs')
old=read('/workspace/agenticgen_early_path_search_20261103/PLAN.json')
teachers=['6pyy_H7S_A_402','3w54_RNB_A_504'];bases={j['teacher']:read(j['config']) for j in old['jobs']}
jobs=[];first={}
for arm,overlay in [(1,'uniguide_stage_10_chemical.json'),(2,'uniguide_stage_10_coordinate_only.json'),(3,'uniguide_iterative_affinity.json')]:
 for teacher in teachers:
  i=len(jobs)+1;ident=f'{i:02d}_arm{arm}_{teacher[:4]}';cfg=dict(bases[teacher]);cfg.update(read(repo/'agenticgen/configs'/overlay))
  cfg['root']=str(R/'jobs'/ident/'attempt_001')
  cfg.pop('rmsd_outlier_policy',None)
  if arm==1:first[teacher]=cfg['root']
  if arm==3:cfg['fixed_scale_source']=first[teacher]
  config=R/'configs'/(ident+'.json');write(config,cfg)
  jobs.append(dict(id=ident,arm=arm,teacher=teacher,config=str(config),config_sha256=sha(config),rounds=cfg['rounds'],
   skill_release='2.16.0' if arm<3 else None,commit=commit,depends_on=next((j['id'] for j in jobs if j['arm']==1 and j['teacher']==teacher),None)))
protected={}
for p in [pathlib.Path(old['root'])/'SUITE_FINISHED.json',*pathlib.Path(old['root']).glob('jobs/*/attempt_001/attempts.json')]:
 if p.exists():protected[str(p)]=sha(p)
context=pathlib.Path(bases[teachers[0]]['generation_context_root'])
for p in [context/'attempts.json',context/'teachers.json']:
 if p.exists():protected[str(p)]=sha(p)
write(R/'SOURCE_PRESERVATION.json',protected)
plan=dict(root=str(R),repo=str(repo),commit=commit,branch='codex/uniguide-flowr-agent',jobs=jobs,
 global_executor_lock=old['global_executor_lock'],durable_snapshot_root='/root/private_data/agenticgen_stage_affinity_campaign_20261104/result_snapshots',
 scientific_candidate_cap=12500,stage_search_rounds=10,teacher_iteration_rounds=5,model_roles=['proposal'],
 condition_evaluation='condition-kabsch-mean-v1',timestamp=time.time(),old_sources_read_only=True)
write(R/'PLAN.json',plan)
env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join([str(repo/'agenticgen/src'),str(repo/'src'),'/workspace/uniguide_deps',bases[teachers[0]]['flowr']])
if not env.get('DEEPSEEK_API_KEY'):raise RuntimeError('API environment unavailable')
with (R/'launcher.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-u',str(repo/'agenticgen/scripts/run_stage_affinity_campaign.py'),'--plan',str(R/'PLAN.json')],cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
ticks=pathlib.Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()[19]
launch=dict(pid=p.pid,start_ticks=ticks,boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),commit=commit,repo=str(repo),root=str(R),timestamp=time.time())
write(R/'LAUNCH.json',launch)
print('CAMPAIGN_DEPLOYED:'+json.dumps(launch),flush=True)
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='ssh.zzai.scnet.cn');parser.add_argument('--port',type=int,default=11741);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/uniguide-flowr-agent'],cwd=repo,text=True).split()[0]
    if published!=commit:raise RuntimeError('Push and verify the current commit before deployment')
    with tempfile.TemporaryDirectory(prefix='guidance216-') as temporary:
        bundle=Path(temporary)/'code.bundle';bootstrap=Path(temporary)/'bootstrap.py'
        subprocess.run(['git','bundle','create',str(bundle),'codex/uniguide-flowr-agent'],cwd=repo,check=True)
        bootstrap.write_text(BOOTSTRAP,encoding='utf8')
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(args.host,port=args.port,username='root',password=os.environ['GUIDANCE_REMOTE_PASSWORD'],timeout=30,auth_timeout=30,banner_timeout=30)
            sftp=client.open_sftp();remote='/workspace/stage-affinity-'+commit[:12]
            sftp.put(str(bundle),remote+'.bundle');sftp.put(str(bootstrap),remote+'-bootstrap.py')
            inner='python '+shlex.quote(remote+'-bootstrap.py')+' '+shlex.quote(commit)+' >&3'
            command='bash -ic '+shlex.quote(inner)+' 3>&1 >/dev/null 2>&1'
            _,stdout,stderr=client.exec_command(command,timeout=180)
            for line in stdout:
                if line.startswith('CAMPAIGN_DEPLOYED:'):
                    print(line.strip());return
            raise RuntimeError('Deployment did not return a launch receipt; inspect remote campaign root without retry')
        finally:client.close()


if __name__=='__main__':main()
