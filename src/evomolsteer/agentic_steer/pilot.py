"""Coordinate local API decisions with a Git-deployed inference-only worker.

Only JSON jobs cross SFTP. Scientific source is verified at a pinned remote Git
runtime. Identical node/child-stream outcomes are shared without disclosure to
conditions that did not request them.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import time

import paramiko

from evomolsteer.io import read_json,write_json,digest
from .workflow import decision,summarize


def command(client,text):
    _,out,err=client.exec_command(text)
    result=out.read().decode();errors=err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError('Remote command failed: '+errors[-2500:])
    return result.strip()


def remote_trials(client,repo,cfg,jobs,runtime,phase):
    local=Path(repo)/cfg['study'];remote_output=cfg['remote_output']
    requested={trial['trial_id']:trial for job in jobs for trial in job['trials']}
    with client.open_sftp() as sftp:
        missing=[]
        for identity,trial in requested.items():
            remote=remote_output+'/'+identity+'.json'
            try:
                with sftp.open(remote) as f:saved=json.load(f)
                if saved['trial']!=trial or saved['status']!='complete':raise ValueError('Remote outcome binding mismatch')
            except FileNotFoundError:missing.append(trial)
        job=dict(jobs[0],trials=missing)
        job_file=local/'transport'/f'{phase}.job.json';write_json(job_file,job)
        remote_job=runtime+'/jobs/'+phase+'.json'
        sftp.put(str(job_file),remote_job)
    if missing:
        q=shlex.quote
        env='LD_LIBRARY_PATH=/opt/miniforge3/envs/molsteer-flowr-dtk/lib:/opt/hyhal/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib'
        pythonpath=runtime+'/src:/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps'
        log=runtime+'/logs/'+phase+'.log';exitfile=runtime+'/jobs/'+phase+'.'+str(time.time_ns())+'.exit'
        shell=f'cd {q(runtime)}; env {q(env)} PYTHONPATH={q(pythonpath)} PYTHONUNBUFFERED=1 /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/run_agentic_native_suffix.py --job {q(remote_job)} > {q(log)} 2>&1; printf "%s\\n" "$?" > {q(exitfile)}'
        # Jobs run detached from this SSH session; no visible helper windows.
        pid=command(client,f'nohup bash -c {q(shell)} < /dev/null > /dev/null 2>&1 & echo $!')
        write_json(local/'transport'/f'{phase}.dispatch.json',{'pid':int(pid),'runtime':runtime,'new_trials':len(missing),'requested_trials':len(requested),'job_sha256':digest(job_file)})
        print(json.dumps({'phase':phase,'dispatched_new_trials':len(missing),'pid':pid}),flush=True)
        while True:
            state=command(client,f'if test -f {q(exitfile)}; then cat {q(exitfile)}; else echo running; fi')
            if state!='running':
                if state!='0':raise RuntimeError('Native inference failed: '+command(client,'tail -45 '+q(log)))
                break
            progress=command(client,'tail -1 '+q(log))
            print(json.dumps({'phase':phase,'progress':progress}),flush=True)
            time.sleep(30)
    with client.open_sftp() as sftp:
        (local/'trials').mkdir(exist_ok=True)
        for identity in requested:
            sftp.get(remote_output+'/'+identity+'.json',str(local/'trials'/(identity+'.json')))
            try:sftp.get(remote_output+'/'+identity+'.sdf.gz',str(local/'trials'/(identity+'.sdf.gz')))
            except FileNotFoundError:pass
        sftp.get(remote_output+'/source_attestation.json',str(local/'transport'/(phase+'.source_attestation.json')))
    return len(missing)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD');p.add_argument('--runtime',required=True)
    p.add_argument('--source-commit',required=True);a=p.parse_args()
    repo=Path(a.repo).resolve();cfg=read_json(a.config);study=repo/cfg['study']
    client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(a.host,port=a.port,username='root',password=os.environ[a.password_env],look_for_keys=False,allow_agent=False,timeout=25)
    client.get_transport().set_keepalive(30)
    new=0
    try:
        deployed=command(client,'cat '+shlex.quote(a.runtime+'/SOURCE_COMMIT'))
        if deployed!=a.source_commit:raise ValueError('Pinned inference source differs from expected commit')
        # Do not compete with an earlier instance of this pilot worker.
        while command(client,"pgrep -f '^/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/run_agentic_native_suffix.py' || true"):
            print('Waiting for this pilot\'s already-dispatched native job',flush=True);time.sleep(30)
        conditions=['C0','B_diversity']+[f'C{i}' for i in range(1,7)]
        for phase in ['initial','feedback']:
            jobs=[]
            for condition in conditions:
                decision(repo,cfg,condition,phase)
                jobs.append(read_json(study/'conditions'/condition/(phase+'.job.json')))
                print(json.dumps({'condition':condition,'phase':phase,'decision_complete':True}),flush=True)
            new+=remote_trials(client,repo,cfg,jobs,a.runtime,phase)
        rows=summarize(repo,cfg)
        write_json(study/'completion.json',{'complete':True,'conditions':len(rows),'new_unique_trials_this_driver':new,
            'unique_trials_in_local_bank':len(list((study/'trials').glob('N*.json'))),'source_commit':a.source_commit,
            'logical_attempts':sum(r['attempts'] for r in rows),'interpretation':'Development screening, not significant superiority proof'})
        print(json.dumps(rows,ensure_ascii=False),flush=True)
    finally:client.close()


if __name__=='__main__':main()
