"""Local serial driver: retain -> diagnose -> restore -> clean -> push/pull -> infer.

Credentials stay in process memory. Remote work is inference/export only. The
driver continues the existing authorized campaign, including a running R14.
"""
import argparse
import getpass
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
import paramiko
from evomolsteer.io import read_json,digest
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.coordinate_backtracking import freeze,record
from evomolsteer.generation.coordinate_exploration import proposal,resolve_parent,regression_triggers
from evomolsteer.generation.coordinate_campaign_report import summarize


class Driver:
    def __init__(self,args):
        self.args=args;self.root=Path(args.repo).resolve();os.chdir(self.root)
        self.cfg=self.root/'configs/experiments/ck2_coordinate_seed42_v1'
        self.campaign=self.cfg/'campaign.json'
        self.evidence=self.root/'docs/experiments/ck2_coordinate_seed42_20261006'
        self.state=self.root/'test/coordinate_campaign_driver';self.state.mkdir(parents=True,exist_ok=True)
        self.python=str(self.root/'.venv/Scripts/python.exe')
        os.environ.update(PYTHONPATH=str(self.root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        self.ssh=paramiko.SSHClient();self.ssh.load_system_host_keys()
        # The existing SSH session has already accepted this host key.
        self.ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
        password=os.environ.get('EVOMOLSTEER_SSH_PASSWORD') or getpass.getpass('SSH password: ')
        self.ssh.connect(args.host,port=args.port,username=args.user,password=password,
            timeout=25,auth_timeout=25,banner_timeout=25,look_for_keys=False,allow_agent=False)
        self.ssh.get_transport().set_keepalive(30);password=None
        self.event('connected')

    def event(self,stage,**data):
        row={'stage':stage,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),**data}
        write_json(self.state/'status.json',row);print(json.dumps(row),flush=True)
        with (self.state/'events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')

    def remote(self,command):
        channel=self.ssh.get_transport().open_session();channel.exec_command(command)
        out=[];err=[]
        while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
            if channel.recv_ready():out.append(channel.recv(65536))
            if channel.recv_stderr_ready():err.append(channel.recv_stderr(65536))
            time.sleep(.05)
        code=channel.recv_exit_status();channel.close()
        if code:raise RuntimeError('Remote command failed: '+b''.join(err).decode(errors='replace')[-1000:])
        return b''.join(out).decode().strip()

    def command(self,number,name,args):
        self.event(name,round=number)
        with (self.state/f'round{number:02d}_{name}.log').open('wb') as log:
            subprocess.run(args,cwd=self.root,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)

    def py(self,number,name,script,*args):self.command(number,name,[self.python,'scripts/'+script,*map(str,args)])

    def push(self,number):
        self.command(number,'git_add',['git','add','--',str(self.campaign),str(self.cfg/f'backtrack_round{number:02d}.json'),str(self.evidence)])
        self.command(number,'git_commit',['git','commit','-m',f'Retain prior evidence and freeze rollback round {number}'])
        self.command(number,'github_push',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])

    def cleanup_local(self,number,label):
        report=self.evidence/f'round_{number:02d}/comparison/comparison.json'
        plan={'allowed_bases':[str(self.root/'data/generated'),str(self.root/'data/archives'),str(self.root/'results')],
            'protected':[str(self.root/p) for p in ('data/optimized','data/reference_terminal','src','configs')],
            'result_report':str(report),'targets':[str(self.root/'data/generated'/label),
                str(self.root/'data/archives'/(label+'.tar.gz')),str(self.root/'data/archives'/(label+'.tar.gz.json')),
                str(self.root/'results'/f'coordinate_round{number}')]}
        plan_path=self.state/f'cleanup_round{number:02d}.plan.json';write_json(plan_path,plan)
        audit=self.evidence/'backtracking'/f'cleanup_after_round{number}_local.json'
        self.py(number,'cleanup_local','retire_experiment_outputs.py','--plan',plan_path,'--report',audit,'--apply')

    def finish(self,r):
        number=r['round'];label=r['campaign'];q=shlex.quote
        marker=self.args.remote_work+f'/round{number:02d}.exit'
        while True:
            status=self.remote(f'if test -f {q(marker)}; then cat {q(marker)}; else echo RUNNING; fi')
            if status!='RUNNING':
                if status!='0':raise RuntimeError(f'Inference round{number} exited {status}')
                break
            self.event('inference_running',round=number,campaign=label);time.sleep(45)
        archive=self.root/'data/archives'/(label+'.tar.gz');archive.parent.mkdir(parents=True,exist_ok=True)
        self.event('download',round=number)
        with self.ssh.open_sftp() as sftp:
            for suffix in ('.tar.gz','.tar.gz.json'):
                target=self.root/'data/archives'/(label+suffix);partial=Path(str(target)+'.partial')
                sftp.get(self.args.remote_work+'/'+label+suffix,str(partial));os.replace(partial,target)
            sftp.get(self.args.remote_work+f'/cleanup_before_round{number:02d}.json',
                str(self.evidence/'backtracking'/f'cleanup_before_round{number}_remote.json'))
        dataset=self.root/'data/generated'/label;out=self.root/'results'/f'coordinate_round{number}'
        reference=self.root/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'
        arms=','.join(r['arms']);batches=','.join(map(str,r['batches']))
        self.py(number,'verify','verify_generation_archive.py','--archive',archive,'--metadata',str(archive)+'.json',
            '--destination',dataset,'--report',self.state/f'round{number}_archive_verification.json')
        self.py(number,'terminal','evaluate_terminal.py','--dataset',dataset,'--campaign',label,'--reference',reference,
            '--output',out,'--arms',arms,'--batches',batches,'--workers','4')
        self.py(number,'execution','audit_local_execution.py','--dataset',dataset,'--campaign',label,'--reference',reference,'--output',out)
        self.py(number,'coordinate','evaluate_coordinate_flowr.py','--dataset',dataset,'--campaign',label,'--output',out)
        self.py(number,'window','evaluate_window_flowr.py','--dataset',dataset,'--campaign',label,
            '--original',self.root/'data/optimized/main1000_w050/analysis_inputs_v2','--original-campaign','main1000_w050','--output',out/'window')
        self.py(number,'retain','preserve_terminal_reports.py','--results',out,'--dataset',dataset,'--campaign',label,
            '--output',self.evidence/f'round_{number:02d}/local')
        result=record(self.campaign,self.evidence,number)
        self.event('result_retained',round=number,shape_improvement=result['shape_improvement_fraction'],
            head_change=result['all_head_change_vs_native'],regressions=regression_triggers(result))
        self.cleanup_local(number,label)
        return result

    def start_next(self,number,result):
        parent,changes,reason=proposal(number,self.evidence)
        program,reference=resolve_parent(self.cfg,parent)
        name=f'coordinate_r{number:02d}_backtrack';output=self.cfg/f'backtrack_round{number:02d}.json'
        freeze(self.campaign,program,reference,output,self.evidence/'backtracking'/f'round_{number:02d}.plan.json',
            number,name,changes,reason)
        write_json(self.evidence/'backtracking'/f'round_{number:02d}.driver_decision.json',{
            'previous_round':number-1,'previous_outcome_sha256':digest(self.evidence/f'round_{number-1:02d}.outcome.json'),
            'regression_triggers':regression_triggers(result),'restored_parent':parent,'single_change':changes,'reason':reason,
            'policy':'Partial regression triggers restoration, not early stopping; these are declared sensitivity hypotheses, not independent validation.'})
        self.push(number);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        q=shlex.quote;remote_repo=self.args.remote_repo;work=self.args.remote_work
        self.event('remote_pull_start',round=number,inference_commit=commit)
        previous=f'docs/experiments/ck2_coordinate_seed42_20261006/round_{number-1:02d}/comparison/comparison.json'
        args=[str(number),name,remote_repo+'/'+output.relative_to(self.root).as_posix(),
            remote_repo+'/'+reference.relative_to(self.root).as_posix(),previous,'gradient','0,1']
        command=(f'set -eu\ngit -C {q(remote_repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet\n'
            f'test "$(git -C {q(remote_repo)} rev-parse HEAD)" = {q(commit)}\n'
            f'nohup bash {q(remote_repo+"/scripts/scnet_coordinate_round.sh")} '+ ' '.join(q(v) for v in args)+
            f' > {q(work+f"/round{number:02d}.log")} 2>&1 < /dev/null &\necho $!\n')
        pid=self.remote(command);config=read_json(self.campaign);r=config['rounds'][-1]
        if r['round']!=number or r['status']!='frozen':raise ValueError('Round state changed during launch')
        r.update(status='running',inference_commit=commit);config['rounds_started']=number;write_json(self.campaign,config)
        self.event('round_launched',round=number,remote_pid=pid,inference_commit=commit)

    def close(self):
        config=read_json(self.campaign);config.update(status='budget_complete',decision='All30 authorized bounded rounds retained; no success claim based on reward proxy alone.')
        write_json(self.campaign,config)
        summarize(self.campaign,self.evidence,self.root/'docs/experiments/ck2_terminal_seed42_20261005/reference/terminal_report.json',
            self.evidence/'backtracking/completed30',config['decision'])
        self.command(30,'git_add_final',['git','add','--',str(self.campaign),str(self.evidence)])
        self.command(30,'git_commit_final',['git','commit','-m','Retain complete30-round rollback campaign and final cleanup reports'])
        self.command(30,'github_push_final',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])
        q=shlex.quote;repo=self.args.remote_repo;work=self.args.remote_work
        # Last raw remote result is retired only after its report is pushed/pulled.
        self.remote(f'git -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet')
        c=read_json(self.campaign);label=c['rounds'][-1]['campaign']
        remote_plan={'allowed_bases':[work],'protected':[repo,'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003','/root/private_data/MolSteer/flowr_root/checkpoints'],
            'result_report':repo+'/docs/experiments/ck2_coordinate_seed42_20261006/round_30/comparison/comparison.json',
            'targets':[work+'/generated',work+'/'+label+'.tar.gz',work+'/'+label+'.tar.gz.json']}
        path=self.state/'final_remote_cleanup.plan.json';write_json(path,remote_plan)
        with self.ssh.open_sftp() as sftp:sftp.put(str(path),work+'/final_remote_cleanup.plan.json')
        python='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
        self.remote(f'cd {q(repo)}; PYTHONPATH={q(repo+"/src")} {q(python)} scripts/retire_experiment_outputs.py --plan {q(work+"/final_remote_cleanup.plan.json")} --report {q(work+"/final_remote_cleanup.json")} --apply')
        with self.ssh.open_sftp() as sftp:sftp.get(work+'/final_remote_cleanup.json',str(self.evidence/'backtracking/final_remote_cleanup.json'))
        self.command(30,'git_add_cleanup',['git','add','--',str(self.evidence/'backtracking/final_remote_cleanup.json')])
        self.command(30,'git_commit_cleanup',['git','commit','-m','Preserve final remote30-round cleanup audit'])
        self.command(30,'github_push_cleanup',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])
        self.event('campaign_complete',rounds_completed=30)

    def run(self):
        config=read_json(self.campaign)
        if config['maximum_rounds']!=30 or config['master_seed']!=42:raise ValueError('Existing authorized30/seed42 contract required')
        while True:
            config=read_json(self.campaign);current=config['rounds'][-1]
            if current['status']!='running':raise ValueError('Driver resumes an already running round; frozen state needs explicit launch')
            result=self.finish(current)
            if current['round']==30:self.close();return
            self.start_next(current['round']+1,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544);p.add_argument('--user',default='root')
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/opt/evomolsteer-coordinate-seed42-20261006')
    p.add_argument('--local-proxy',default='http://127.0.0.1:7897');p.add_argument('--remote-proxy',default='http://127.0.0.1:17897')
    args=p.parse_args();driver=None
    try:
        driver=Driver(args);driver.run()
    except Exception as error:
        if driver:driver.event('execution_error',error_type=type(error).__name__,message=str(error)[-1500:])
        raise
    finally:
        if driver:driver.ssh.close()
