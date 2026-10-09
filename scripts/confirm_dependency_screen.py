# Run locally after the frozen Luna screen collector completes.
import argparse,json,os,subprocess,time
from pathlib import Path
import paramiko
from evomolsteer.io import read_json as _read_json,write_json

def read_json(path):
 # Local collector may be replacing a compact report at this instant.
 for attempt in range(5):
  try:return _read_json(path)
  except json.JSONDecodeError:
   if attempt==4:raise
   time.sleep(.2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--work',required=True);p.add_argument('--commit',required=True);p.add_argument('--deadline-seconds',type=int,default=14400);a=p.parse_args()
 repo=Path(a.repo);study=repo/'docs/experiments/steer_dependency_20261009/luna_suite';deadline=time.monotonic()+a.deadline_seconds
 eligible=read_json(study/'protocol.json')['confirmation']['eligible_conditions']
 while time.monotonic()<deadline:
  qpath=repo/'results/steer_dependency_20261009/luna/screen/quality_summary.json'
  if qpath.exists():
   quality=read_json(qpath)
   if all(x in quality['outcomes'] for x in eligible):break
  time.sleep(30)
 else:raise TimeoutError('Frozen eligible screens unfinished')
 # Declared all-attempt predicted-affinity criterion; lexical tie-break.
 ordered=sorted(eligible,key=lambda x:(-quality['outcomes'][x]['affinity_mean_all'],x));winner=ordered[0]
 receipt={'selected_condition':winner,'selection_rule':'Highest screened all-attempt mean pIC50 among preregistered eligible zero-Steer-input/reward programs; lexical tie-break','screen_order':ordered,'means':{x:quality['outcomes'][x]['affinity_mean_all'] for x in ordered},'confirmation_batches':list(range(77,83)),'code_commit':a.commit,'not_final_validation':True}
 write_json(study/'confirmation_selection.json',receipt);print(json.dumps(receipt),flush=True)
 client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect('ksai.scnet.cn',port=10544,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=25)
 try:
  code=a.work+'/code_luna_confirmation_v1'
  # Existing worktree was pulled and pinned before any confirmation labels.
  script=f'''set -eu
code='{code}'
flowr=/root/private_data/MolSteer/flowr_root
work='{a.work}'
[ "$(git -C "$code" rev-parse HEAD)" = '{a.commit}' ]
source /opt/MolSteer/scripts/scnet/activate_dtk.sh
export LD_LIBRARY_PATH="/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${{LD_LIBRARY_PATH:-}}"
export PYTHONPATH="$code/src:$flowr:$flowr/experiments/evomolsteer_online_20261004/runtime_deps:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python - <<'PY'
from pathlib import Path
import subprocess,json
w=Path('{a.work}');r=w/'code_luna_confirmation_v1'
cmd=['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python','-u',str(r/'scripts/run_dependency_queue.py'),'--repo',str(r),'--flowr-root','/root/private_data/MolSteer/flowr_root','--work',str(w),'--specs','configs/experiments/steer_dependency_v1/luna/jobs/Confirm_{winner}.json','--predecessor',str(w/'luna_confirm_control.queue.json'),'--name','luna_confirm_candidate']
with (w/'luna_confirm_candidate.worker.log').open('wb') as log:
 proc=subprocess.Popen(cmd,cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({{'submitted_pid':proc.pid,'candidate':'{winner}'}}))
PY
'''
  ch=client.get_transport().open_session();ch.exec_command(script)
  output=ch.makefile('r').read().decode();error=ch.makefile_stderr('r').read().decode();rc=ch.recv_exit_status();print(output,flush=True)
  if rc:raise RuntimeError(error)
 finally:client.close()
 transport=repo/'test/data/steer_dependency/luna';reports=study/'transport_records'
 names=['Confirm_R26_native','Confirm_'+winner]
 while time.monotonic()<deadline:
  ready=[];client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy());client.connect('ksai.scnet.cn',port=10544,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],look_for_keys=False,allow_agent=False,timeout=25)
  try:
   with client.open_sftp() as sftp:
    for name in names:
     try:
      with sftp.open(a.work+'/'+name+'.status.json') as f:state=json.load(f)
     except FileNotFoundError:continue
     if state['status']=='failed':raise RuntimeError(name+str(state.get('error')))
     if state['status']!='complete':continue
     for suffix in ['.tar.gz','.tar.gz.json']:
      dest=transport/(name+suffix)
      if not dest.exists():sftp.get(a.work+'/archives/'+dest.name,str(dest))
     write_json(reports/(name+'.remote.json'),state);ready.append(name)
  finally:client.close()
  for name in ready:
   out=repo/'results/steer_dependency_20261009/luna'/name
   subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/evaluate_dependency_archive.py'),'--name',name,'--transport',str(transport),'--reports',str(reports),'--output',str(out),'--reference',str(repo/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'),'--workers','2'],check=True,cwd=repo)
  if len(ready)==2:break
  time.sleep(60)
 else:raise TimeoutError('Confirmation inference unfinished')
 cohorts=[{'name':label,'arm':arm,'metrics':f'results/steer_dependency_20261009/luna/{name}/candidate_metrics.csv'} for name,label,arm in [('Confirm_R26_native','native','unguided'),('Confirm_R26_native','R26','gradient'),('Confirm_'+winner,winner,'gradient')]]
 write_json(study/'confirmation_manifest.json',{'cohorts':cohorts,'controls':['native','R26'],'selected_condition':winner,'not_used_for_design':True})
 subprocess.run([str(repo/'.venv/Scripts/python.exe'),str(repo/'scripts/analyze_dependency_study.py'),'--manifest',str(study/'confirmation_manifest.json'),'--output',str(repo/'results/steer_dependency_20261009/luna/confirmation')],check=True,cwd=repo)
 write_json(study/'confirmation_complete.json',{'status':'complete','selected_condition':winner,'quality':read_json(repo/'results/steer_dependency_20261009/luna/confirmation/quality_summary.json')})
 print('CONFIRMATION_LOCAL_ANALYSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
