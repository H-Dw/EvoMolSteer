"""Collect completed remote cohorts and evaluate locally without active LLM calls."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import paramiko
from evomolsteer.io import read_json,write_json


def collect(args):
    repo=args.repo.resolve();study=args.study.resolve();study.mkdir(parents=True,exist_ok=True)
    transport=repo/'data/generated/m2_terminal_regions_20261014/transport';transport.mkdir(parents=True,exist_ok=True)
    plan=read_json(repo/'configs/experiments/m2_terminal_regions_20261014/jobs/queue.json')
    state_path=study/'collection_status.json'
    state=read_json(state_path) if state_path.exists() else {'completed':{},'failed':{},'started_unix':time.time()}
    deadline=time.monotonic()+args.watch_seconds
    while True:
        client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(args.host,port=args.port,username=args.user,password=os.environ[args.password_env],
                look_for_keys=False,allow_agent=False,timeout=25)
        except (OSError,paramiko.SSHException) as error:
            client.close()
            state['last_network_error']={'type':type(error).__name__,'unix':time.time()};write_json(state_path,state)
            if args.watch_seconds<=0 or time.monotonic()>=deadline:raise
            time.sleep(30);continue
        try:
            with client.open_sftp() as sftp:
                for job in plan['jobs']:
                    name=job['name']
                    if name in state['completed'] or name in state['failed']:continue
                    try:
                        with sftp.open(args.work+'/'+name+'.status.json') as stream:remote=json.load(stream)
                    except FileNotFoundError:continue
                    state['current_observed']=remote
                    write_json(study/'transport_records'/(name+'.remote.json'),remote)
                    if remote['status']=='failed':
                        state['failed'][name]=remote
                        for suffix in ['.log','.execution_audit.json']:
                            try:sftp.get(args.work+'/'+name+suffix,str(study/'transport_records'/(name+suffix)))
                            except FileNotFoundError:pass
                        continue
                    if remote['status']!='complete':continue
                    for suffix in ['.tar.gz','.tar.gz.json']:
                        path=transport/(name+suffix)
                        if not path.exists():
                            partial=path.with_suffix(path.suffix+'.partial')
                            sftp.get(args.work+'/archives/'+path.name,str(partial));partial.replace(path)
                    for suffix in ['.execution_audit.json','.bridge.json','.retire.json']:
                        try:sftp.get(args.work+'/'+name+suffix,str(study/'transport_records'/(name+suffix)))
                        except FileNotFoundError:pass
                    output=study/'evaluated'/name
                    try:
                        subprocess.run([sys.executable,str(repo/'scripts/evaluate_dependency_archive.py'),
                            '--name',name,'--transport',str(transport),'--reports',str(study/'transport_records'),
                            '--output',str(output),'--reference',str(repo/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'),
                            '--workers',str(args.workers)],cwd=repo,check=True)
                        from evomolsteer.storage.generation_archive import restoration_destination
                        view=read_json(restoration_destination(transport,name)/'results'/name/'EVALUATION_VIEW.json')
                        bank=read_json(repo/'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json')
                        checks={}
                        for item in view['batches']:
                            key=item['batch_path'].replace('gradient_zero/','unguided/')
                            checks[item['batch_path']]=item['initial_state_signature']==bank['result']['initial_state_signatures'][key]
                        write_json(study/'transport_records'/(name+'.input_pairing.json'),{'passed':all(checks.values()),'checks':checks})
                        if not all(checks.values()):raise ValueError('Generated inputs differ from the reused frozen control bank')
                    except (subprocess.CalledProcessError,ValueError) as error:
                        state['failed'][name]={'scope':'local_evaluation_or_input_verification','type':type(error).__name__,
                            'reason':str(error),'remote_generation_status':remote['status']}
                        write_json(state_path,state);continue
                    state['completed'][name]={'metrics':str(output/'candidate_metrics.csv'),'code_commit':remote['code_commit'],
                        'completed_unix':time.time(),'archive_sha256':remote['transport']['archive_sha256']}
                    write_json(state_path,state)
                    print(json.dumps({'evaluated':name,'completed':len(state['completed'])}),flush=True)
                try:
                    with sftp.open(args.work+'/queue_status.json') as stream:queue=json.load(stream)
                    state['remote_queue']=queue
                except FileNotFoundError:queue={}
        except (OSError,paramiko.SSHException) as error:
            state['last_network_error']={'type':type(error).__name__,'unix':time.time()};write_json(state_path,state)
            if args.watch_seconds<=0 or time.monotonic()>=deadline:raise
            time.sleep(30);continue
        finally:client.close()
        write_json(state_path,state)
        if any(n.startswith('M2REG_terminal_') for n in state['completed']):
            subprocess.run([sys.executable,str(repo/'scripts/report_terminal_region_experiment.py'),
                '--repo',str(repo),'--study',str(study),'--output',str(study/'comparison')],cwd=repo,check=True)
        finished=len(state['completed'])+len(state['failed'])==len(plan['jobs'])
        stopped=queue.get('status') in ['blocked_runtime_control_mismatch','complete','complete_with_failed_candidates']
        if finished or stopped:
            state['status']='complete' if not state['failed'] and finished else 'complete_with_failures_or_blocked'
            write_json(state_path,state);return state
        if args.watch_seconds<=0 or time.monotonic()>=deadline:
            state['status']='running';write_json(state_path,state);return state
        time.sleep(min(60,max(0,deadline-time.monotonic())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--study',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--user',default='root');p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD')
    p.add_argument('--work',required=True);p.add_argument('--watch-seconds',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    print(collect(p.parse_args()))
