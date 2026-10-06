"""Local evidence/decision/evaluation; remote inference only; fifteen new rounds.

Every plan is frozen and pushed before the corresponding remote pull/inference.
No molecular population selection, secret persistence or hidden LLM calls.
"""
import argparse,copy,json,os,select,shlex,socket,subprocess,threading,time,paramiko
from pathlib import Path
from resume_coordinate_campaign import Driver
from evomolsteer.io import read_json,digest
from evomolsteer.generation.prototypes import write_json
from evomolsteer.generation.coordinate_exploration import validate_retention
from evomolsteer.generation.motif_campaign import proposal,record,AXES


class MotifDriver(Driver):
    def __init__(self,args):
        args.config_root=getattr(args,'config_root','configs/experiments/ck2_motif_seed42_v1')
        args.evidence_root=getattr(args,'evidence_root','docs/experiments/ck2_motif_seed42_20261006')
        args.state_root=getattr(args,'state_root','test/motif_campaign_driver')
        args.memory_reconnect=True
        super().__init__(args)
        self.setup_forward()

    def setup_forward(self):
        self.forward_owned=False
        def relay(channel):
            upstream=None
            try:
                upstream=socket.create_connection(('127.0.0.1',7897),timeout=20)
                while True:
                    ready,_,_=select.select([channel,upstream],[],[],30)
                    for source,dest in ((channel,upstream),(upstream,channel)):
                        if source in ready:
                            data=source.recv(65536)
                            if not data:return
                            dest.sendall(data)
            except OSError:pass
            finally:
                if upstream:upstream.close()
                channel.close()
        def handler(channel,origin,server):threading.Thread(target=relay,args=(channel,),daemon=True).start()
        self.ssh.get_transport().request_port_forward('127.0.0.1',17897,handler=handler);self.forward_owned=True

    def remote(self,command):
        for attempt in range(4):
            try:return super().remote(command)
            except (paramiko.SSHException,OSError):
                self.event('transport_reconnect',attempt=attempt+1)
                self.ssh.close();time.sleep(min(5*(attempt+1),20))
                self.ssh=paramiko.SSHClient();self.ssh.load_system_host_keys();self.ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
                self.ssh.connect(self.args.host,port=self.args.port,username=self.args.user,password=self._connection_secret,
                    timeout=25,auth_timeout=25,banner_timeout=25,look_for_keys=False,allow_agent=False)
                self.ssh.get_transport().set_keepalive(30);self.setup_forward()
        raise RuntimeError('Authenticated transport recovery exhausted')

    def freeze(self,number):
        c=read_json(self.campaign)
        if number!=c['rounds_completed']+1 or c['rounds_started']!=c['rounds_completed'] or not 1<=number<=15:raise ValueError('Sequential fifteen-round budget')
        plan=proposal(number,self.evidence);parent=plan['parent_round']
        if parent:
            p=copy.deepcopy(read_json(self.cfg/f'backtrack_round{parent:02d}.json'))
            if 'dose_factor' in plan:p['native_rms_ratio']*=plan['dose_factor']
            if 'time_ramp_power' in plan:p['time_ramp_power']=plan['time_ramp_power']
            reference=next(v for v in self.cfg.glob('*reference.json.gz') if digest(v)==p['reference_sha256'])
            if plan.get('strict_scale'):
                from evomolsteer.generation.window_reference import load_reference
                old=load_reference(reference)
                if old.get('target_definition','instantaneous')=='instantaneous':
                    stem='all' if old['channels']==['all'] else 'cross'
                    reference=self.cfg/('strict_'+stem+'_reference.json.gz')
                    p['reference_sha256']=digest(reference)
        else:
            stem='all' if plan['channel']=='all' else 'cross'
            prefix='survival_' if plan.get('target_definition')=='boundary_survival' else ''
            reference=self.cfg/(prefix+stem+'_reference.json.gz')
            from evomolsteer.generation.window_reference import load_reference
            p={'schema_version':'current-coordinate-program-1.0','seed':42,'family':'LLM_evidence_bound_spatial_motif_design',
                'window':load_reference(reference)['window'],'reference_sha256':digest(reference),
                'reward_view':plan['reward_view'],'motif_components':plan['motif_components'],
                'mixture_temperature':.5,'robust_delta':2.,'native_rms_ratio':plan['native_rms_ratio'],
                'dose_reference':'predictive_flow','core_radius_A':5.,'time_ramp_power':plan['time_ramp_power'],
                'contrast_bound_nats':2.,'constraints':{'max_atom_step_A':.025,'max_cumulative_rms_A':1.25,'severe_receptor_clash_A':.8,'backtrack_attempts':7}}
        if number==9:
            # This intervention goes through the same grounded Analyst ->
            # Designer -> registered-formula compiler used by the API adapter.
            compiled=self.evidence/'llm_interface/boundary_agents/compiled_round09.json'
            designed=read_json(compiled)
            expected={'reference_sha256':digest(reference),'reward_view':plan['reward_view'],
                'motif_components':plan['motif_components'],'native_rms_ratio':plan['native_rms_ratio'],
                'time_ramp_power':plan['time_ramp_power'],'mixture_temperature':.5,'robust_delta':2.,
                'target_definition':'boundary_survival','window':p['window'],'seed':42}
            if any(designed.get(k)!=v for k,v in expected.items()):raise ValueError('Compiled Designer intervention differs from predeclared R9 contract')
            p=copy.deepcopy(designed);p['compiled_design_sha256']=digest(compiled)
        p.update(round=number,program_id=f'ck2_motif_round{number:02d}',
            derivation={'plan':plan,'evidence':'docs/experiments/ck2_motif_seed42_20261006/mining',
                        'source_manifest_sha256':digest(self.evidence/'mining/manifest.json')},
            topology_policy='Free native graph transitions; no pair-distance acceptance or graph equality veto')
        from evomolsteer.generation.window_reference import load_reference
        loaded=load_reference(reference)
        p['target_definition']=loaded.get('target_definition','instantaneous')
        if number>=9 and (not loaded.get('scale_score_times') or loaded['scale_score_times'][-1]>=p['window'][1]-1e-6):raise ValueError('Final-scope program uses outside-state normalization')
        p['agent_review_sha256']=digest(self.evidence/'agent_review/Analyst_Designer.final_review.v3.json')
        boundary_review=self.evidence/'agent_review/Analyst_Designer.boundary_review.json'
        if p['target_definition']=='boundary_survival' and boundary_review.exists():p['boundary_agent_review_sha256']=digest(boundary_review)
        arms=['gradient'];batches=[0,1];n=100;split='discovery'
        if number==1:arms=['unguided','gradient_zero','gradient']
        if number==13:arms=['unguided','gradient_zero','gradient'];batches=[14,15];split='validation'
        if number==14:arms=['unguided','gradient'];batches=[17];n=50;split='heldout'
        if number==15:arms=['unguided','gradient'];batches=[18,19];split='heldout'
        name=f'coordinate_r{number:02d}_motif15'
        write_json(self.cfg/f'backtrack_round{number:02d}.json',p)
        write_json(self.evidence/f'round_{number:02d}.plan.json',{'round':number,'parent_round':parent,'split':split,'plan':plan,
            'program_sha256':digest(self.cfg/f'backtrack_round{number:02d}.json'),'reference_sha256':digest(reference),'window':p['window'],
            'decision_basis':'Agent-reviewed motif hypotheses followed by bounded predeclared program search. Not fifteen independent LLM inventions.',
            'previous_outcome_sha256':digest(self.evidence/f'round_{number-1:02d}.outcome.json') if number>1 else None})
        c['rounds'].append({'round':number,'campaign':name,'status':'frozen','arms':arms,'batches':batches,'n_per_arm':n,'seed':42,
            'parent_round':parent,'reason':plan['reason'],'split':split,'reference_sha256':digest(reference)})
        write_json(self.campaign,c)

    def launch_frozen(self,r):
        number=r['round'];self.push(number);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        p=self.cfg/f'backtrack_round{number:02d}.json';reference=next(v for v in self.cfg.glob('*reference.json.gz') if digest(v)==r['reference_sha256'])
        previous=('docs/experiments/ck2_coordinate_seed42_20261006/round_30/comparison/comparison.json' if number==1 else
                  f'docs/experiments/ck2_motif_seed42_20261006/round_{number-1:02d}/comparison/comparison.json')
        repo,work=self.args.remote_repo,self.args.remote_work;q=shlex.quote
        args=['--repo',repo,'--work',work,'--round',str(number),'--campaign',r['campaign'],
              '--program',repo+'/'+p.relative_to(self.root).as_posix(),'--reference',repo+'/'+reference.relative_to(self.root).as_posix(),
              '--previous',previous,'--arms',','.join(r['arms']),'--batches',','.join(map(str,r['batches'])),'--n',str(r['n_per_arm'])]
        command=f'set -eu\ngit -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet\ntest "$(git -C {q(repo)} rev-parse HEAD)" = {q(commit)}\n'
        command+=f'/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python {q(repo+"/scripts/dispatch_motif_round.py")} '+' '.join(q(v) for v in args)
        launched=json.loads(self.remote(command));c=read_json(self.campaign);c['rounds'][-1].update(status='running',inference_commit=launched['inference_commit'])
        c['rounds_started']=number;write_json(self.campaign,c);self.event('round_launched',round=number,remote_pid=launched['pid'])

    def finish(self,r):
        number=r['round'];label=r['campaign'];q=shlex.quote;work=self.args.remote_work
        while True:
            status=self.remote(f'if test -f {q(work+f"/round{number:02d}.exit")}; then cat {q(work+f"/round{number:02d}.exit")}; else echo RUNNING; fi')
            if status!='RUNNING':
                if status!='0':raise RuntimeError(f'Remote inference exit {status}; inspect round log, do not silently skip a round')
                break
            self.event('inference_running',round=number);time.sleep(45)
        prefix=getattr(self.args,'output_prefix','motif')
        archive=self.root/'data/archives'/(label+'.tar.gz');dataset=self.root/'data/generated'/label;out=self.root/'results'/f'{prefix}_round{number}'
        local=self.evidence/f'round_{number:02d}/local';outcome=self.evidence/f'round_{number:02d}.outcome.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        with self.ssh.open_sftp() as sftp:
            for suffix in ('.tar.gz','.tar.gz.json'):
                dest=self.root/'data/archives'/(label+suffix)
                if not dest.exists():sftp.get(work+'/'+label+suffix,str(dest)+'.partial');os.replace(str(dest)+'.partial',dest)
            sftp.get(work+f'/cleanup_before_round{number:02d}.json',str(self.evidence/f'cleanup_before_round{number:02d}.remote.json'))
        if not (local/'retention.json').exists():
            if not dataset.exists():self.py(number,'verify','verify_generation_archive.py','--archive',archive,'--metadata',str(archive)+'.json','--destination',dataset,'--report',self.state/f'round{number}_archive_verification.json')
            ref=self.root/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz'
            jobs=[('terminal_report.json','terminal','evaluate_terminal.py',['--dataset',dataset,'--campaign',label,'--reference',ref,'--output',out,'--arms',','.join(r['arms']),'--batches',','.join(map(str,r['batches'])),'--workers','4']),
                  ('execution_report.json','execution','audit_local_execution.py',['--dataset',dataset,'--campaign',label,'--reference',ref,'--output',out]),
                  ('coordinate_audit.json','coordinate','evaluate_coordinate_flowr.py',['--dataset',dataset,'--campaign',label,'--output',out]),
                  ('window/report.json','window','evaluate_window_flowr.py',['--dataset',dataset,'--campaign',label,'--original',self.root/'data/optimized/main1000_w050/analysis_inputs_v2','--original-campaign','main1000_w050','--output',out/'window'])]
            for f,name,script,args in jobs:
                if not self.complete_json(out/f):self.py(number,name,script,*args)
            if getattr(self.args,'record_response',False) and not self.complete_json(out/'guidance_response.json'):
                self.py(number,'response','analyze_guidance_response.py','--dataset',dataset,'--campaign',label,'--output',out/'guidance_response.json')
            self.py(number,'retain','preserve_terminal_reports.py','--results',out,'--dataset',dataset,'--campaign',label,'--output',local)
        validate_retention(local,r['n_per_arm']*len(r['arms']))
        if outcome.exists():
            result=read_json(outcome)
            if result['round']!=number or result['parent_round']!=r['parent_round'] or result['split']!=r['split']:raise ValueError('Retained outcome/plan mismatch')
            # A crash can occur after the outcome is saved but before counters.
            # Report retention was validated above; repair only that final state.
            c=read_json(self.campaign)
            if c['rounds_completed']<number:
                if c['rounds_started']!=number or c['rounds'][-1]['round']!=number:raise ValueError('Nonsequential recovery')
                c['rounds'][-1]['status']='completed';c['rounds_completed']=number;write_json(self.campaign,c)
        else:result=record(self.campaign,self.evidence,number)
        self.event('result_retained',round=number,head_change=result['all_head_change_vs_native'],shape_improvement=result['shape_improvement_fraction'],MMFF_p90=result['MMFF_p90'])
        self.cleanup_local(number,label);return result

    def cleanup_local(self,number,label):
        r=next(v for v in read_json(self.campaign)['rounds'] if v['round']==number)
        validate_retention(self.evidence/f'round_{number:02d}/local',r['n_per_arm']*len(r['arms']))
        report=self.evidence/f'round_{number:02d}/comparison/comparison.json';audit=self.evidence/f'cleanup_after_round{number:02d}.local.json'
        plan={'allowed_bases':[str(self.root/p) for p in ('data/generated','data/archives','results')],
              'protected':[str(self.root/p) for p in ('data/optimized','data/reference_terminal','configs','src')],
              'result_report':str(report),'targets':[str(self.root/'data/generated'/label),str(self.root/'results'/f'{getattr(self.args,"output_prefix","motif")}_round{number}'),
                   str(self.root/'data/archives'/(label+'.tar.gz')),str(self.root/'data/archives'/(label+'.tar.gz.json'))]}
        if audit.exists():
            prior=read_json(audit)
            if prior['status']=='deleted' and prior['report_sha256']==digest(report) and not any(Path(t).exists() for t in plan['targets']):return
            raise ValueError('Cleanup evidence disagrees with retained report or surviving raw output')
        path=self.state/f'cleanup_round{number}.json';write_json(path,plan)
        self.py(number,'cleanup_local','retire_experiment_outputs.py','--plan',path,'--report',audit,'--apply')

    def push(self,number):
        self.command(number,'git_add',['git','add','--',str(self.cfg),str(self.evidence)])
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=self.root).returncode:
            self.command(number,'git_commit',['git','commit','-m',f'Preserve motif evidence and freeze new15 round {number}'])
        self.command(number,'github_push',['git','-c','http.proxy='+self.args.local_proxy,'push','origin','main'])

    def close(self):
        c=read_json(self.campaign);c['status']='budget_complete';write_json(self.campaign,c)
        from evomolsteer.generation.motif_reporting import report
        report(self.evidence,self.evidence,require_complete=True)
        self.push(15);q=shlex.quote;repo=self.args.remote_repo;work=self.args.remote_work
        self.remote(f'git -C {q(repo)} -c http.proxy={q(self.args.remote_proxy)} pull --ff-only --quiet')
        last=c['rounds'][-1];report=repo+'/docs/experiments/ck2_motif_seed42_20261006/round_15/comparison/comparison.json'
        plan={'allowed_bases':[work],'protected':[repo,'/root/private_data/MolSteer/flowr_root/experiments/ck2_clk3_lineage_20261003','/root/private_data/MolSteer/flowr_root/checkpoints'],
            'result_report':report,'targets':[work+'/generated',work+'/'+last['campaign']+'.tar.gz',work+'/'+last['campaign']+'.tar.gz.json']}
        path=self.state/'final_remote_cleanup.json';write_json(path,plan)
        with self.ssh.open_sftp() as sftp:sftp.put(str(path),work+'/final_remote_cleanup.plan.json')
        self.remote(f'cd {q(repo)}; PYTHONPATH={q(repo+"/src")} /opt/miniforge3/envs/molsteer-flowr-dtk/bin/python scripts/retire_experiment_outputs.py --plan {q(work+"/final_remote_cleanup.plan.json")} --report {q(work+"/final_remote_cleanup.json")} --apply')
        with self.ssh.open_sftp() as sftp:sftp.get(work+'/final_remote_cleanup.json',str(self.evidence/'final_remote_cleanup.json'))
        self.push(15);self.event('campaign_complete',rounds_completed=15)

    def run(self):
        c=read_json(self.campaign)
        if c['maximum_rounds']!=15 or c['master_seed']!=42:raise ValueError('New 15-round contract required')
        if not c['rounds']:self.freeze(1)
        while True:
            if (self.state/'USER_STOP.json').exists():self.event('stopped_by_user');return
            c=read_json(self.campaign);r=c['rounds'][-1]
            if r['status']=='frozen':self.launch_frozen(r);continue
            if r['status']=='running':self.finish(r)
            elif r['status']=='completed':self.cleanup_local(r['round'],r['campaign'])
            else:raise ValueError('Unknown round status')
            if r['round']==15:self.close();return
            self.freeze(r['round']+1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544);p.add_argument('--user',default='root')
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_motif_seed42_20261006')
    p.add_argument('--local-proxy',default='http://127.0.0.1:7897');p.add_argument('--remote-proxy',default='http://127.0.0.1:17897')
    d=None
    try:d=MotifDriver(p.parse_args());d.run()
    except Exception as e:
        if d:d.event('execution_error',error_type=type(e).__name__,message=str(e)[-1500:])
        raise
    finally:
        if d:
            if d.forward_owned:d.ssh.get_transport().cancel_port_forward('127.0.0.1',17897)
            d.ssh.close()
