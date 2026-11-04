"""Supersede unstarted arm 3 without modifying a live checkout or restarting arms 1/2."""
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
suite=pathlib.Path('/workspace/agenticgen_stage_affinity_campaign_20261104')
root=suite/'revisions/arm3_interaction_region';commit=sys.argv[1]
repo=pathlib.Path('/workspace/uniguide_runs/arm3-region-'+commit[:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return sha256(pathlib.Path(p).read_bytes()).hexdigest()
parent=read(suite/'PLAN.json');jobs=[j for j in parent['jobs'] if j['arm']==3]
if root.exists() or repo.exists():raise RuntimeError('Existing revision must be preserved')
if len(jobs)!=2:raise RuntimeError('Expected two unstarted iterative jobs')
for j in jobs:
 if pathlib.Path(read(j['config'])['root']).exists():raise RuntimeError('Arm 3 already has state; cannot change it silently')
launch=read(suite/'LAUNCH.json');status=read(suite/'STATUS.json')
if status.get('job_id') not in [j['id'] for j in parent['jobs'] if j['arm']<3]:raise RuntimeError('Parent is no longer in arms 1/2')
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
ticks=pathlib.Path(f"/proc/{launch['pid']}/stat").read_text().split(') ',1)[1].split()[19]
if boot!=launch['boot_id'] or ticks!=launch['start_ticks']:raise RuntimeError('Parent process identity changed')
root.mkdir(parents=True)
subprocess.run(['git','clone','--shared','--no-checkout',parent['repo'],str(repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
 with (root/'git_fetch.log').open('wb') as log:
  r=subprocess.run(['git','-C',str(repo),'fetch','https://github.com/H-Dw/EvoMolSteer.git','codex/uniguide-flowr-agent'],
   timeout=60,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('GitHub fetch failed')
except Exception as e:
 write(root/'GITHUB_FETCH_FAILED.json',dict(type=type(e).__name__,error=str(e)))
 subprocess.run(['git','-C',str(repo),'fetch','/workspace/arm3-region-'+commit[:12]+'.bundle','codex/uniguide-flowr-agent'],
  check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['git','-C',str(repo),'checkout','--detach',commit],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:raise RuntimeError('Revision commit differs')
overlay=read(repo/'agenticgen/configs/uniguide_iterative_affinity.json');revised=[]
for job in jobs:
 old_config=read(job['config']);cfg=dict(old_config);cfg.update(overlay)
 source_job=next(j for j in parent['jobs'] if j['id']==job['depends_on'])
 reference=read(source_job['config'])
 if cfg['chemical_inpainting']!=reference['chemical_inpainting']:raise RuntimeError('Chemical injection differs from experiment 1')
 cfg['root']=str(pathlib.Path(old_config['root']).parent/'attempt_002')
 config=root/'configs'/(job['id']+'.json');write(config,cfg)
 revised.append(dict(id=job['id'],teacher=job['teacher'],config=str(config),config_sha256=sha(config),
  original_config=job['config'],original_config_sha256=sha(job['config']),original_attempt=old_config['root'],
  reference_config=source_job['config'],reference_config_sha256=sha(source_job['config'])))
plan=dict(root=str(root),parent_suite=str(suite),parent_commit=parent['commit'],repo=str(repo),commit=commit,jobs=revised,
 global_executor_lock=parent['global_executor_lock'],chemical_inpainting=overlay['chemical_inpainting'],
 scientific_candidate_cap=500,reason='User requested arm 3 use the same interaction-region chemical policy as arm 1',timestamp=time.time())
write(root/'PLAN.json',plan)
# The frozen parent tests existence before launching an iterative child. Guards
# reserve no scientific attempts and leave every original config/source unchanged.
for job in revised:
 guard=pathlib.Path(job['original_attempt']);guard.mkdir(parents=True,exist_ok=False)
 write(guard/'SUPERSEDED.json',dict(revision_plan=str(root/'PLAN.json'),revision_commit=commit,
  replacement_config=job['config'],original_config_sha256=job['original_config_sha256'],scientific_attempts_reserved=0))
cfg=read(revised[0]['config']);env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(repo/'agenticgen/src'),str(repo/'src'),'/workspace/uniguide_deps',cfg['flowr']])
with (root/'launcher.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-u',str(repo/'agenticgen/scripts/run_arm3_region_revision.py'),'--plan',str(root/'PLAN.json')],
  cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
ticks=pathlib.Path(f'/proc/{p.pid}/stat').read_text().split(') ',1)[1].split()[19]
receipt=dict(pid=p.pid,start_ticks=ticks,boot_id=boot,commit=commit,repo=str(repo),root=str(root),timestamp=time.time())
write(root/'LAUNCH.json',receipt)
print('ARM3_REVISION_DEPLOYED:'+json.dumps(receipt),flush=True)
'''


def main():
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/uniguide-flowr-agent'],cwd=repo,text=True).split()[0]
    if commit!=published:raise RuntimeError('Revision must be published before deployment')
    with tempfile.TemporaryDirectory(prefix='arm3-region-') as temp:
        bundle=Path(temp)/'patch.bundle';bootstrap=Path(temp)/'bootstrap.py'
        subprocess.run(['git','bundle','create',str(bundle),'codex/uniguide-flowr-agent','^be431184764b551907b1fdc07360d26d0cd19a12'],cwd=repo,check=True)
        bootstrap.write_text(BOOTSTRAP,encoding='utf8')
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect('ssh.zzai.scnet.cn',port=11741,username='root',password=os.environ['GUIDANCE_REMOTE_PASSWORD'],timeout=30,auth_timeout=30,banner_timeout=30)
            s=client.open_sftp();remote='/workspace/arm3-region-'+commit[:12]
            s.put(str(bundle),remote+'.bundle');s.put(str(bootstrap),remote+'-bootstrap.py')
            inner='python '+shlex.quote(remote+'-bootstrap.py')+' '+shlex.quote(commit)+' >&3'
            _,out,_=client.exec_command('bash -ic '+shlex.quote(inner)+' 3>&1 >/dev/null 2>&1',timeout=180)
            for line in out:
                if line.startswith('ARM3_REVISION_DEPLOYED:'):
                    print(line.strip());return
            raise RuntimeError('Revision launch did not return a receipt; inspect before retry')
        finally:client.close()


if __name__=='__main__':main()
