"""Local real-API/Git orchestration; remote host executes inference only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time
import paramiko

from evomolsteer.io import read_json,write_json,digest
from .agents import analyze,signature,compile_programs,validate_analysis
from .campaign import prepare_controls,prepare_round,condition_metrics,compact_output,update_state


LIBRARIES='/opt/miniforge3/envs/molsteer-flowr-dtk/lib:/opt/hyhal/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib'
REMOTE_REPO='/root/private_data/MolSteer/EvoMolSteer'
REMOTE_PYTHON='/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
DEPENDENCIES='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_online_20261004/runtime_deps'


def emit(event,**kwargs):print(json.dumps({'event':event,**kwargs},ensure_ascii=False),flush=True)


def local(repo,args):
    result=subprocess.run(args,cwd=repo,text=True,encoding='utf-8',capture_output=True)
    if result.returncode:raise RuntimeError(f'{args[0]} failed: {result.stderr[-1200:]}')
    return result.stdout.strip()


def commit(repo,master,message):
    paths=[master['campaign_study'],'configs/experiments/partial_region_search_20261014',
        'skills/partial-region-analyst','skills/partial-region-designer']
    local(repo,['git','add','--',*paths])
    if local(repo,['git','diff','--cached','--name-only']):local(repo,['git','commit','-m',message])
    sha=local(repo,['git','rev-parse','HEAD']);local(repo,['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'])
    emit('commit_pushed',commit=sha,message=message);return sha


class Remote:
    def __init__(self):
        self.client=paramiko.SSHClient();self.client.load_system_host_keys();self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect('ksai.scnet.cn',port=50496,username='root',password=os.environ['MOLSTEER_SCNET_PASSWORD'],
            timeout=20,auth_timeout=25,banner_timeout=20,look_for_keys=False,allow_agent=False)
        transport=self.client.get_transport();transport.set_keepalive(30)
        def handler(channel,origin,server):
            def forward():
                import select
                upstream=socket.create_connection(('127.0.0.1',7897),timeout=20)
                try:
                    while True:
                        ready,_,_=select.select([channel,upstream],[],[],30)
                        for source in ready:
                            data=source.recv(65536)
                            if not data:return
                            (upstream if source is channel else channel).sendall(data)
                finally:channel.close();upstream.close()
            threading.Thread(target=forward,daemon=True).start()
        self.proxy_port=transport.request_port_forward('127.0.0.1',0,handler=handler)
        self.sftp=self.client.open_sftp()
    def command(self,command):
        _,stdout,stderr=self.client.exec_command(command,timeout=120)
        output=stdout.read().decode();errors=stderr.read().decode();code=stdout.channel.recv_exit_status()
        if code:raise RuntimeError(f'Remote command exited {code}: {errors[-1400:]} {output[-1400:]}')
        return output.strip()
    def exists(self,path):
        try:self.sftp.stat(path);return True
        except FileNotFoundError:return False
    def json(self,path):
        with self.sftp.open(path,'r') as stream:return json.loads(stream.read().decode())
    def download_tree(self,remote,local_path):
        import stat
        local_path=Path(local_path);local_path.mkdir(parents=True,exist_ok=True)
        for attr in self.sftp.listdir_attr(remote):
            source=remote+'/'+attr.filename;target=local_path/attr.filename
            if stat.S_ISDIR(attr.st_mode):self.download_tree(source,target)
            else:
                self.sftp.get(source,str(target))
                if target.stat().st_size!=attr.st_size:raise ValueError('Incomplete result download')
    def close(self):self.sftp.close();self.client.close()


def infer(repo,master,cfg,cfg_path,remote,sha):
    relative=Path(cfg_path).relative_to(repo).as_posix();root=master['remote_campaign'];runtime=root+'/runtime'
    # This workspace belongs exclusively to this sequential campaign. It has no raw trajectories/checkpoints.
    cleanup=f"from pathlib import Path; import shutil; p=Path({runtime!r}).resolve(); root=Path({root!r}).resolve(); assert p.parent==root and p.name=='runtime'; shutil.rmtree(p) if p.exists() else None; p.mkdir(parents=True)"
    archive_paths=['src','scripts/generate_final_region_guidance.py',relative,cfg['study']]
    shell='\n'.join(['set -eu',f'cd {shlex.quote(REMOTE_REPO)}',
        f'git -c http.proxy=http://127.0.0.1:{remote.proxy_port} fetch --quiet origin main',
        'git merge --ff-only --quiet origin/main',f'git cat-file -e {sha}^{{commit}}',
        'python -c '+shlex.quote(cleanup),
        f'git archive {sha} '+' '.join(shlex.quote(p) for p in archive_paths)+f' | tar -x -C {shlex.quote(runtime)}',
        f'mkdir -p {shlex.quote(cfg["remote_output"])}',
        f'export LD_LIBRARY_PATH={shlex.quote(LIBRARIES)}',
        f'export PYTHONPATH={shlex.quote(runtime+"/src:"+DEPENDENCIES)}',
        f'nohup {REMOTE_PYTHON} -u {runtime}/scripts/generate_final_region_guidance.py --repo {runtime} --config {runtime}/{relative} --flowr-root /root/private_data/MolSteer/flowr_root > {cfg["remote_output"]}/worker.log 2>&1 < /dev/null &',
        f'echo $! > {cfg["remote_output"]}/worker.pid',f'cat {cfg["remote_output"]}/worker.pid'])
    pid=int(remote.command(shell).splitlines()[-1]);emit('remote_inference_started',round=cfg.get('round_id','controls'),pid=pid,n=cfg['pilot_samples'])
    seen=set();started=time.time()
    while not remote.exists(cfg['remote_output']+'/COMPLETE.json'):
        import stat
        for attr in remote.sftp.listdir_attr(cfg['remote_output']):
            if not stat.S_ISDIR(attr.st_mode):continue
            path=cfg['remote_output']+'/'+attr.filename+'/COMPLETE.json'
            if attr.filename not in seen and remote.exists(path):seen.add(attr.filename);emit('condition_complete',**remote.json(path))
        alive=remote.command(f'if kill -0 {pid} 2>/dev/null; then echo running; else echo stopped; fi')
        if alive=='stopped':
            log=remote.command('tail -n 24 '+shlex.quote(cfg['remote_output']+'/worker.log'))
            raise RuntimeError('Inference worker stopped before completion: '+log)
        if time.time()-started>3600:raise RuntimeError('Inference exceeded one hour; investigate instead of advancing')
        time.sleep(10)
    local_output=repo/master.get('generated_root','data/generated/partial_region_search_20261014')/Path(cfg['remote_output']).name
    remote.download_tree(cfg['remote_output'],local_output)
    # Preserve complete structures locally; remote retains compact scores, coordinates and audits for reuse.
    for path in local_output.rglob('*.sdf.gz'):
        rel=path.relative_to(local_output).as_posix();remote.sftp.remove(cfg['remote_output']+'/'+rel)
    return local_output


def summarize(repo,master,folder):
    teacher=read_json(repo/master['base_study']/'candidates'/master['candidate_id']/'teacher.json')
    terminal=read_json(repo/master['campaign_study']/'evidence.json')['terminal_coords_A']
    return [condition_metrics(p,master['tail_top_k'],teacher,terminal) for p in sorted(folder.iterdir()) if p.is_dir() and (p/'COMPLETE.json').exists()]


def rollback(repo,master,state,number):
    if state['best_round']=='baseline':source=repo/master['campaign_study']/'core_skills/analyst/SKILL.md'
    else:source=repo/master['campaign_study']/state['best_round']/'skills/analyst/SKILL.md'
    target=repo/master['analyst_skill'];before=digest(target);target.write_text(source.read_text(encoding='utf-8'),encoding='utf-8')
    write_json(repo/master['campaign_study']/f'R{number:02d}'/'rollback.json',{'restored_round':state['best_round'],
        'restored_generator_commit':state.get('best_commit'),'before_skill_sha256':before,'after_skill_sha256':digest(target),
        'reason':'Two consecutive top-five-qualified tail decreases beyond predeclared tolerance; mathematical generator unchanged'})
    state['rollback_commits'].append(commit(repo,master,f'rollback(partial-search): restore {state["best_round"]} after consecutive declines'))
    emit('rollback',round=number,restored=state['best_round'])


def activate(repo,master,state):
    if state['best_round']=='baseline':path=repo/state['controls']['screen']['cfg']
    else:path=repo/'configs/experiments/partial_region_search_20261014'/(state['best_round']+'.json')
    cfg=read_json(path);cfg.update(only_program_ids=[state['best']['condition']],include_unguided=False,
        selected_policy_round=state['best_round'],selected_policy_commit=state['best_commit'])
    write_json(repo/'configs/experiments/partial_region_search_20261014/active_best.json',cfg)


def run(repo,master_path,stop_after):
    repo=Path(repo).resolve();master=read_json(master_path);study=repo/master['campaign_study'];state_path=study/'state.json'
    if state_path.exists() and (study/'COMPLETE.json').exists():
        completed=read_json(state_path)
        if completed['next_round']>master['max_rounds']:
            emit('campaign_already_complete',rounds=len(completed['rounds']),best_round=completed['best_round'])
            return completed
    remote=Remote()
    try:
        if state_path.exists():state=read_json(state_path)
        else:
            cfg,path=prepare_controls(repo,master,'screen',master['screen_samples'],0)
            sha=commit(repo,master,'experiment(partial-search): freeze matched screen controls')
            folder=infer(repo,master,cfg,path,remote,sha);metrics=summarize(repo,master,folder)
            partial=[m for m in metrics if m['condition'].startswith('baseline_local')]
            best=max(partial,key=lambda m:m['top5_qualified_mean_pic50'])
            state={'next_round':1,'best':best,'best_round':'baseline','best_commit':sha,'baseline':metrics,
                'rounds':[],'decline_streak':0,'seen_selections':[{'atom_slots':best['atom_slots'],'atom_weights':best['atom_weights']}],
                'rollback_commits':[],'controls':{'screen':{'cfg':str(path.relative_to(repo)),'local':str(folder.relative_to(repo))}},'numerical_cache':{}}
            write_json(study/'controls_screen/metrics.json',metrics);compact_output(folder,study/'controls_screen/transport.json.gz')
            write_json(state_path,state);activate(repo,master,state);commit(repo,master,'results(partial-search): record frozen matched controls')
            emit('baseline_ready',best_partial=best)
        for number in range(state['next_round'],min(stop_after,master['max_rounds'])+1):
            label=f'R{number:02d}';cfg_path=repo/'configs/experiments/partial_region_search_20261014'/f'{label}.json'
            if cfg_path.exists():cfg=read_json(cfg_path);feedback=read_json(repo/cfg['study']/'feedback_input.json')
            else:cfg,cfg_path,feedback=prepare_round(repo,master,number,state)
            emit('agent_call',round=label,strategy=cfg['strategy'])
            candidate_folder=repo/cfg['study']/'candidates'/master['candidate_id']
            if (candidate_folder/'bindings.json').exists():
                # A completed real API request is immutable; do not rebuild its prompt after protocol improvements.
                a=read_json(candidate_folder/'Analyst.json');d=read_json(candidate_folder/'Designer.json');t=read_json(candidate_folder/'teacher.json')
                validate_analysis(a,read_json(study/'evidence.json'))
                programs=compile_programs(cfg,a,d,t,digest(candidate_folder/'Analyst.json'),digest(candidate_folder/'Designer.json'),digest(candidate_folder/'teacher.json'))
                binding=read_json(candidate_folder/'bindings.json')
                if [read_json(repo/item['program']) for item in binding['programs']]!=programs:raise ValueError('Cached real decisions differ from current compiler')
                emit('cached_real_api_validated',round=label)
            else:programs=analyze(repo,cfg,feedback)
            if not programs:raise ValueError('Designer deferred; no invented replacement region')
            cfg['only_program_ids']=[p['program_id'] for p in programs];write_json(cfg_path,cfg)
            sha=commit(repo,master,f'experiment(partial-search): {label} {cfg["strategy"]} Analyst-bound partial reward')
            controls_key=f'batch{cfg["pilot_initial_batch"]}_n{cfg["pilot_samples"]}'
            if cfg.get('endpoint_mix',0):controls_key+='_joint'
            if (number>=19 or cfg.get('endpoint_mix',0)) and controls_key not in state['controls']:
                control_cfg,path=prepare_controls(repo,master,controls_key,cfg['pilot_samples'],cfg['pilot_initial_batch'],cfg['guidance_starts'],cfg.get('endpoint_mix',0))
                if number<19:
                    control_cfg['include_unguided']=False;write_json(path,control_cfg)
                control_sha=commit(repo,master,f'experiment(partial-search): freeze {controls_key} confirmation controls')
                control_folder=infer(repo,master,control_cfg,path,remote,control_sha);control_metrics=summarize(repo,master,control_folder)
                write_json(study/('controls_'+controls_key)/'metrics.json',control_metrics)
                compact_output(control_folder,study/('controls_'+controls_key)/'transport.json.gz')
                state['controls'][controls_key]={'cfg':str(path.relative_to(repo)),'local':str(control_folder.relative_to(repo))}
                write_json(state_path,state);sha=commit(repo,master,f'results(partial-search): {controls_key} matched controls')
            novel=[];reuse=[]
            pool=f"{cfg['pilot_initial_batch']}:{cfg['pilot_samples']}"
            for p in programs:
                key=pool+':'+signature(p)
                if key in state['numerical_cache']:reuse.append(state['numerical_cache'][key])
                else:novel.append(p['program_id'])
            if novel:
                cfg['only_program_ids']=novel;write_json(cfg_path,cfg)
                # Commit actual selected dispatch set before remote execution, even when it is a numerical subset.
                sha=commit(repo,master,f'experiment(partial-search): {label} dispatch only new numerical programs')
                folder=infer(repo,master,cfg,cfg_path,remote,sha);metrics=summarize(repo,master,folder)
                write_json(study/label/'transport_manifest.json',compact_output(folder,study/label/'transport.json.gz'))
                for p in programs:
                    if p['program_id'] in novel:state['numerical_cache'][pool+':'+signature(p)]=str((folder/p['program_id']).relative_to(repo))
            else:metrics=[];folder=None
            for path in reuse:metrics.append(condition_metrics(repo/path,master['tail_top_k']))
            if number<19:
                entry=update_state(state,label,metrics,master,sha)
                activate(repo,master,state)
                state['seen_selections'].append({'atom_slots':programs[0]['atom_slots'],'atom_weights':programs[0]['atom_weights']})
                if entry['rollback_required']:rollback(repo,master,state,number)
            else:
                key=controls_key;controls=summarize(repo,master,repo/state['controls'][key]['local'])
                entry={'round':label,'commit':sha,'confirmation':True,'controls':controls,'conditions':metrics,
                    'frozen_training_winner':state['best_round'],'no_validation_tuning':True,'reuse':reuse}
                state['rounds'].append(entry)
            entry['strategy']=cfg['strategy'];entry['skill_response_verified']=bool(novel);entry['new_numerical_programs']=len(novel)
            entry['instruction_and_contract_validated']=True
            if not novel:entry['no_new_execution_function']='This Agent output is numerically identical to an earlier reward; original results reused'
            write_json(study/label/'results.json',entry);state['next_round']=number+1;write_json(state_path,state)
            commit(repo,master,f'results(partial-search): {label} terminal feedback and rollback decision')
            emit('round_complete',round=label,best_round=state['best_round'],
                selected_points=len(programs[0]['atom_slots']),rollback=entry.get('rollback_required',False),
                scores=[{'start':m['start'],'top5':m['top5_qualified_mean_pic50'],'best_valid':m['best_qualified_pic50'],
                    'mean_all':m['mean_all_pic50'],'qualified':m['n_qualified'],'above_reference':m['n_above_reference']} for m in metrics])
        if state['next_round']>master['max_rounds']:
            # Publish the empirically best training policy, not the final exploratory failure.
            if state['best_round']!='baseline':
                (repo/master['analyst_skill']).write_text((study/state['best_round']/'skills/analyst/SKILL.md').read_text(encoding='utf-8'),encoding='utf-8')
            write_json(study/'COMPLETE.json',{'rounds':len(state['rounds']),'best_training_round':state['best_round'],
                'best_training':state['best'],'independent_confirmation':state['rounds'][-1],'rollback_commits':state['rollback_commits']})
            commit(repo,master,'results(partial-search): publish best partial policy and independent confirmation')
        return state
    finally:remote.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True);p.add_argument('--stop-after',type=int,default=20)
    a=p.parse_args();run(a.repo,a.config,a.stop_after)


if __name__=='__main__':main()
