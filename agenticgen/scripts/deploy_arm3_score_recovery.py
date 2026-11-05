"""Deploy the authorized 6pyy teacher-score fix without replaying scientific samples."""
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
previous=suite/'revisions/arm3_teacher_score_binding'
root=suite/'revisions/arm3_integer_masks'
commit=sys.argv[1]
repo=pathlib.Path('/workspace/uniguide_runs/arm3-masks-'+commit[:12])

def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,value):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return sha256(pathlib.Path(p).read_bytes()).hexdigest()
def attempts(p):
 file=pathlib.Path(p)/'attempts.json'
 return sum(a['count'] for a in read(file).values()) if file.exists() else 0
def alive(launch):
 stat=pathlib.Path('/proc')/str(launch['pid'])/'stat'
 return stat.exists() and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['boot_id'] and stat.read_text().split(') ',1)[1].split()[19]==launch['start_ticks']

if root.exists() or repo.exists():raise RuntimeError('Recovery already has state; inspect it instead of resubmitting')
parent=read(suite/'PLAN.json');old=read(previous/'PLAN.json')
if not (suite/'SUITE_FINISHED.json').exists() or not (previous/'REVISION_FINISHED.json').exists():
 raise RuntimeError('Prior coordinators have not finished')
if any(alive(read(p/'LAUNCH.json')) for p in (suite,previous)):
 raise RuntimeError('Prior coordinator is still alive')
job=next(j for j in old['jobs'] if j['id']=='05_arm3_6pyy')
failure=next(j for j in read(previous/'job_results.json') if j['job_id']==job['id'])
if failure['status']!='failed' or read(previous/(job['id']+'_EXIT.json'))['returncode']==0:
 raise RuntimeError('Expected the preserved failed teacher-score attempt')
if sha(job['config'])!=job['config_sha256']:raise RuntimeError('Original configuration changed')
cfg=read(job['config']);failed=pathlib.Path(cfg['root'])
if attempts(failed) or list(failed.glob('rounds/*/CLAIM.json')) or any((failed/'acceptance'/name).exists() for name in
 ('terminal.pt.gz','COMPLETE.json')) or read(failed/'acceptance/steps.json') or read(failed/'acceptance/rng_ledger.json'):
 raise RuntimeError('Old attempt has executed or reserved work; no automatic replay is allowed')
error=read(failed/'acceptance/FAILED.json')
if error['type']!='RuntimeError' or 'indices' not in error['error'] or 'FloatTensor' not in error['error']:
 raise RuntimeError('Expected the preserved fresh-prior float mask/index failure')
source=pathlib.Path(cfg['fixed_scale_source'])
final=read(source/'versions/GD01/FINALIZED.json')
if not final['complete'] or final['configured_rounds']!=10:raise RuntimeError('Experiment 1 is incomplete')
frozen_scale=read(failed/'SCALE_SOURCE.json')
measured=source/frozen_scale['source_id']/'aggregate.json'
if sha(source/'versions/GD01/FINALIZED.json')!=frozen_scale['final_sha256'] or sha(measured)!=frozen_scale['aggregate_sha256']:
 raise RuntimeError('Fixed scale source changed')
spent={}
for j in parent['jobs']:
 for p in pathlib.Path(read(j['config'])['root']).parent.glob('attempt_*'):
  spent[str(p)]=attempts(p)
if sum(spent.values())+250>12500:raise RuntimeError('Campaign candidate cap would be exceeded')
new_attempt=failed.parent/'attempt_005'
if new_attempt.exists():raise RuntimeError('The scientific recovery attempt already exists')

root.mkdir(parents=True)
subprocess.run(['git','clone','--shared','--no-checkout',old['repo'],str(repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
try:
 with (root/'git_fetch.log').open('xb') as log:
  result=subprocess.run(['git','-C',str(repo),'fetch','https://github.com/H-Dw/EvoMolSteer.git','codex/uniguide-flowr-agent'],
   timeout=45,env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError('GitHub fetch failed')
except Exception as error:
 write(root/'GITHUB_FETCH_FAILED.json',dict(type=type(error).__name__,error=str(error)))
 subprocess.run(['git','-C',str(repo),'fetch','/workspace/arm3-masks-'+commit[:12]+'.bundle','codex/uniguide-flowr-agent'],
  check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['git','-C',str(repo),'checkout','--detach',commit],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=commit:raise RuntimeError('Recovery commit differs')
cfg['root']=str(new_attempt)
config=root/'configs'/(job['id']+'.json');write(config,cfg)
entry=dict(id=job['id'],teacher=job['teacher'],config=str(config),config_sha256=sha(config),
 previous_attempt=str(failed),previous_config=job['config'],previous_config_sha256=job['config_sha256'])
plan=dict(root=str(root),parent_suite=str(suite),repo=str(repo),commit=commit,jobs=[entry],
 global_executor_lock=parent['global_executor_lock'],scientific_candidate_cap=250,
 previous_revision=str(previous),previous_attempt_scientific_candidates=0,campaign_scientific_candidates_before=sum(spent.values()),
 attempt_budget_audit=spent,fixed_scale_source=frozen_scale,
 reason='User authorized repairing and rerunning experiment 3/6pyy; normalize both teacher and fresh-prior masks and retain the corrected fixed dose-shape binding',timestamp=time.time())
write(root/'PLAN.json',plan)
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join([str(repo/'agenticgen/src'),str(repo/'src'),'/workspace/uniguide_deps',cfg['flowr']])
python='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
with (root/'launcher.log').open('xb') as log:
 process=subprocess.Popen([python,'-u',str(repo/'agenticgen/scripts/run_arm3_region_revision.py'),'--plan',str(root/'PLAN.json')],
  cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
launch=dict(pid=process.pid,start_ticks=pathlib.Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),commit=commit,repo=str(repo),root=str(root),timestamp=time.time())
write(root/'LAUNCH.json',launch)
print('ARM3_SCORE_RECOVERY_DEPLOYED:'+json.dumps(launch),flush=True)
'''


def main():
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/uniguide-flowr-agent'],cwd=repo,text=True).split()[0]
    if commit!=published:raise RuntimeError('The recovery commit must be published first')
    with tempfile.TemporaryDirectory(prefix='arm3-score-') as temporary:
        bundle=Path(temporary)/'recovery.bundle';bootstrap=Path(temporary)/'bootstrap.py'
        subprocess.run(['git','bundle','create',str(bundle),'codex/uniguide-flowr-agent',
            '^e203ed1fabfd1a285de5c13293a5fdd8083e74ee'],cwd=repo,check=True)
        bootstrap.write_text(BOOTSTRAP,encoding='utf8')
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect('ssh.zzai.scnet.cn',port=11741,username='root',password=os.environ['GUIDANCE_REMOTE_PASSWORD'],
                timeout=20,auth_timeout=20,banner_timeout=20)
            sftp=client.open_sftp();sftp.get_channel().settimeout(30)
            remote='/workspace/arm3-masks-'+commit[:12]
            sftp.put(str(bundle),remote+'.bundle');sftp.put(str(bootstrap),remote+'-bootstrap.py')
            inner='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python '+shlex.quote(remote+'-bootstrap.py')+' '+shlex.quote(commit)+' >&3'
            # Existing shell initialization may print secrets; discard both streams.
            _,out,_=client.exec_command('bash -ic '+shlex.quote(inner)+' 3>&1 >/dev/null 2>&1',timeout=80)
            for line in out:
                if line.startswith('ARM3_SCORE_RECOVERY_DEPLOYED:'):
                    print(line.strip());return
            raise RuntimeError('No launch receipt returned; inspect the recovery root before any retry')
        finally:client.close()


if __name__=='__main__':main()
