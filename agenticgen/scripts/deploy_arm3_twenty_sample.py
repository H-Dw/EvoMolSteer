"""Deploy the user's five-round, twenty-candidate experiment-3 recovery."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import paramiko


BOOTSTRAP=r'''
import json,os,pathlib,subprocess,sys,time
from hashlib import sha256
commit=sys.argv[1]
suite=pathlib.Path('/workspace/agenticgen_stage_affinity_campaign_20261104')
previous=suite/'revisions/arm3_endpoint_device'
validation=suite/'revisions/arm3_teacher_graph_validation'
root=suite/'revisions/arm3_twenty_sample'
repo=pathlib.Path('/workspace/uniguide_runs/arm3-twenty-'+commit[:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,value):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return sha256(pathlib.Path(p).read_bytes()).hexdigest()
def attempts(p):
 p=pathlib.Path(p)/'attempts.json'
 return sum(x['count'] for x in read(p).values()) if p.exists() else 0
def alive(launch):
 p=pathlib.Path('/proc')/str(launch['pid'])/'stat'
 return p.exists() and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['boot_id'] and p.read_text().split(') ',1)[1].split()[19]==launch['start_ticks']
if root.exists() or repo.exists():raise RuntimeError('Recovery already has state; inspect instead of resubmitting')
if not (previous/'REVISION_FINISHED.json').exists() or alive(read(previous/'LAUNCH.json')):
 raise RuntimeError('Prior recovery is not terminal')
validated=read(validation/'native_pipeline/VALIDATION_COMPLETE.json')
if not read(validation/'VALIDATION_FINISHED.json')['complete'] or not validated['complete'] or sum(bool(x.get('prepared')) for x in validated['candidates'])<2:
 raise RuntimeError('Promoted-teacher native validation is incomplete')
prior=read(previous/'PLAN.json');job=prior['jobs'][0]
if sha(job['config'])!=job['config_sha256']:raise RuntimeError('Preserved prior configuration changed')
cfg=read(job['config']);failed=pathlib.Path(cfg['root']);new_attempt=failed.parent/'attempt_008'
if attempts(failed)!=150 or not (failed/'ENGINEERING_STOP.json').exists() or new_attempt.exists():
 raise RuntimeError('Preserved reservations or fresh destination differ')
parent=read(suite/'PLAN.json');spent={};teacher_spent=0
for item in parent['jobs']:
 for path in pathlib.Path(read(item['config'])['root']).parent.glob('attempt_*'):
  spent[str(path)]=attempts(path)
  if item['id']=='05_arm3_6pyy':teacher_spent+=spent[str(path)]
if sum(spent.values())!=6400 or teacher_spent!=150 or sum(spent.values())+100>12500 or teacher_spent+100>250:
 raise RuntimeError('Campaign or teacher scientific allowance differs')
frozen_scale=read(failed/'SCALE_SOURCE.json');source=pathlib.Path(cfg['fixed_scale_source'])
if (not read(source/'versions/GD01/FINALIZED.json')['complete'] or
 sha(source/'versions/GD01/FINALIZED.json')!=frozen_scale['final_sha256'] or
 sha(source/frozen_scale['source_id']/'aggregate.json')!=frozen_scale['aggregate_sha256'] or
 frozen_scale['scale']!=[50,100,40,60,80,200]):raise RuntimeError('Experiment-1 fixed scale changed')

root.mkdir(parents=True);subprocess.run(['git','init','--quiet',str(repo)],check=True)
# Register ancestors first to avoid nested shared-object depth limits. No frozen
# checkout is modified; only this new checkout's alternate list is written.
stores=[];seen=set()
def collect(p):
 p=p.resolve()
 if p in seen:return
 if len(seen)>30 or not p.is_dir():raise RuntimeError('Unexpected shared object store')
 seen.add(p);alternate=p/'info/alternates'
 if alternate.exists():
  for name in alternate.read_text().splitlines():collect(pathlib.Path(name) if pathlib.Path(name).is_absolute() else p/name)
 stores.append(p)
collect(pathlib.Path(read(validation/'PLAN.json')['repo'])/'.git/objects')
alternate=repo/'.git/objects/info/alternates';alternate.parent.mkdir(parents=True,exist_ok=True)
alternate.write_text('\n'.join(map(str,stores))+'\n')
try:
 with (root/'git_fetch.log').open('xb') as log:
  fetched=subprocess.run(['git','-C',str(repo),'fetch','--no-tags','https://github.com/H-Dw/EvoMolSteer.git','codex/uniguide-flowr-agent'],
   timeout=45,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),stdout=log,stderr=subprocess.STDOUT)
  if fetched.returncode:raise RuntimeError('GitHub fetch failed')
except Exception as error:
 write(root/'GITHUB_FETCH_FAILED.json',dict(type=type(error).__name__,error=str(error)))
 subprocess.run(['git','-C',str(repo),'fetch','--no-tags','/workspace/arm3-twenty-'+commit[:12]+'.bundle','codex/uniguide-flowr-agent'],
  check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['git','-C',str(repo),'checkout','--detach',commit],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:raise RuntimeError('Checkout commit differs')
cfg.update(read(repo/'agenticgen/configs/uniguide_iterative_affinity_20.json'))
cfg['root']=str(new_attempt);cfg.pop('engineering_replay',None)
if cfg['population']!=20 or cfg['teacher_iteration']['population']!=20 or cfg['rounds']!=5:raise RuntimeError('Expected five complete twenty-candidate rounds')
config=root/'configs/05_arm3_6pyy.json';write(config,cfg)
entry=dict(id='05_arm3_6pyy',teacher=job['teacher'],config=str(config),config_sha256=sha(config),previous_attempt=str(failed))
plan=dict(root=str(root),parent_suite=str(suite),repo=str(repo),commit=commit,jobs=[entry],
 global_executor_lock=parent['global_executor_lock'],scientific_candidate_cap=100,requires_scientific_budget_approval=True,
 supersedes_only_unstarted_arm3=False,previous_attempt_scientific_candidates=150,attempt_budget_audit=spent,
 campaign_scientific_candidates_before=6400,campaign_scientific_candidates_after_maximum=6500,
 per_teacher_cumulative_scientific_cap=250,scientific_population=20,engineering_population=20,rounds=5,
 scientific_seed_block_offset=7,scientific_seeds={str(r):[42+100003*(5*(r+7)+k) for k in range(5)] for r in range(1,6)},
 fixed_scale_source=frozen_scale,promoted_teacher_validation_sha256=sha(validation/'native_pipeline/VALIDATION_COMPLETE.json'),
 authorization='User explicitly requested five rounds with twenty samples per round instead of an additional 250-candidate rerun',timestamp=time.time())
write(root/'PLAN.json',plan)
write(root/'SCIENTIFIC_BUDGET_APPROVAL.json',dict(commit=commit,approved_new_scientific_candidates=100,
 config_sha256={entry['id']:entry['config_sha256']},authorization_source=plan['authorization']))
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(repo/'agenticgen/src'),str(repo/'src'),'/workspace/uniguide_deps',cfg['flowr']]))
with (root/'launcher.log').open('xb') as log:
 process=subprocess.Popen(['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python','-u',str(repo/'agenticgen/scripts/run_arm3_region_revision.py'),'--plan',str(root/'PLAN.json')],
  cwd='/workspace',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=process.pid,start_ticks=pathlib.Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),commit=commit,repo=str(repo),root=str(root),timestamp=time.time())
write(root/'LAUNCH.json',launch);print('ARM3_TWENTY_DEPLOYED:'+json.dumps(launch),flush=True)
'''


def main():
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/uniguide-flowr-agent'],cwd=repo,text=True).split()[0]
    if commit!=published:raise RuntimeError('Publish the fixed commit first')
    with tempfile.TemporaryDirectory(prefix='arm3-twenty-') as temp:
        folder=Path(temp);bundle=folder/'code.bundle';bootstrap=folder/'bootstrap.py'
        subprocess.run(['git','bundle','create',str(bundle),'codex/uniguide-flowr-agent','^672c74d31acf89a1ac1269cb491d0cdf14d4b13c'],cwd=repo,check=True)
        bootstrap.write_text(BOOTSTRAP,encoding='utf8')
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect('ssh.zzai.scnet.cn',port=11741,username='root',password=os.environ['GUIDANCE_REMOTE_PASSWORD'],timeout=20,auth_timeout=20,banner_timeout=20)
            sftp=client.open_sftp();sftp.get_channel().settimeout(30);remote='/workspace/arm3-twenty-'+commit[:12]
            sftp.put(str(bundle),remote+'.bundle');sftp.put(str(bootstrap),remote+'-bootstrap.py')
            inner='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python '+shlex.quote(remote+'-bootstrap.py')+' '+shlex.quote(commit)+' >&3'
            # Shell startup can contain unrelated sensitive text; discard it.
            _,out,_=client.exec_command('bash -ic '+shlex.quote(inner)+' 3>&1 >/dev/null 2>&1',timeout=80)
            for line in out:
                if line.startswith('ARM3_TWENTY_DEPLOYED:'):print(line.strip());return
            raise RuntimeError('Missing launch receipt; inspect the recovery before any retry')
        finally:client.close()


if __name__=='__main__':main()
