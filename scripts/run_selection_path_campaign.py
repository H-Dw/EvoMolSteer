"""Sequential experimental proposals, matched final evaluation and real rollback."""
import argparse,copy,json,shlex,shutil,time
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import retain_round,summarize_tail,paired_effect
from evomolsteer.generation.selection_workflow import restore_incumbent
from evomolsteer.storage.generation_archive import verify_generation_archive
from run_sequential_path_campaign import Driver as TransportDriver,evaluate_isolated
from dispatch_path_round import verify_retention

TRIALS={
 2:({'quality':'rank'},{},'Continuous quality ordering replaces raw score amplitude; retains original endpoint teacher modes.'),
 3:({'quality':'rank'},{'teacher_score_beta':1.},'Weaker rank tilt tests whether the previous ordering gives excessive relative competition.'),
 4:({'ess_fraction':.5},{},'Minimum local teacher ESS at half the selected support; preserves ordering with minimal uniform mass.'),
 5:({'ess_fraction':.75},{},'Higher exploration base chance, changing only the ESS target in the same mechanism.'),
 6:({'legacy_prior_normalization':True},{},'Negative control: legacy NumPy normalization only, with unchanged R26 scalar and no actual ESS floor; not eligible as a learned-rule winner.'),
 7:({'niche_balance':.5},{},'Partial inverse niche occupancy discourages duplicate geometric modes without subtracting attraction.'),
 8:({'niche_balance':1.},{},'Equalize local niche prior mass before quality tilt; separate rare-mode coverage from mean quality.'),
 9:({'selection':'niche'},{},'Nearest representative per geometric niche before filling K; fixed teacher count and unchanged dose.'),
10:({'selection':'batch'},{},'Limit nearest teachers to distinct source batches, protecting independent evidence from duplicate source modes.'),
11:({'precision_mix':.1},{},'Mild within-niche, shrunk coordinate precision retains all teacher centers and complete point clouds.'),
12:({'precision_mix':.25},{},'Increase only local anisotropic coordinate metric contribution.'),
13:({'precision_mix':.5},{},'Stronger joint coordinate covariance metric; bounded SPD eigenvalues prevent pathological scale.'),
14:({'robust_aggregation':'atom'},{},'Atomwise robust spatial residual preserves full molecule attraction while reducing single-atom dominance.'),
15:({'robust_aggregation':'atom'},{'pointcloud_delta_A':2.},'Curvature refinement of atomwise spatial robust loss, anchored to unchanged R26 teacher source.'),
}

class Driver(TransportDriver):
    def __init__(self,args):
        self.args=args;self.root=Path(args.repo).resolve();self.cfg=self.root/'configs/experiments/selection_path20_v1'
        self.docs=self.root/'docs/experiments/selection_path20_20261008';self.state=self.root/'test/selection_path20_driver'
        self.state.mkdir(parents=True,exist_ok=True);self.work=args.remote_work;self.remote_repo=args.remote_repo
        self.threshold=read_json(self.docs/'protocol.json')['tail_threshold_pic50'];self.connect()
    def push(self,message):
        self.local(['git','add',str(self.cfg.relative_to(self.root)),str(self.docs.relative_to(self.root))])
        self.local(['git','diff','--cached','--check'])
        if self.local(['git','diff','--cached','--name-only']):self.local(['git','commit','-m',message])
        self.local(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'])
        commit=self.local(['git','rev-parse','HEAD']);q=shlex.quote
        self.remote(f'timeout 180 git -C {q(self.remote_repo)} -c http.proxy=http://127.0.0.1:17897 pull --ff-only origin main')
        if self.remote(f'git -C {q(self.remote_repo)} rev-parse HEAD')!=commit:raise ValueError('Exact source commit required')
        return commit
    def best(self,admissible_only=False,current_only=False):
        c=read_json(self.cfg/'campaign.json');rows=[r for r in c['rounds'] if 2<=r['round']<=16 and r['round']!=6]
        if current_only:rows=[r for r in rows if r['round']>=8]
        if admissible_only:rows=[r for r in rows if r['screening_admissible']]
        return max(rows,key=lambda r:(r['mean_vs_R26'], -r['round']))['round'] if rows else None
    def reference(self,p):
        for f in [self.cfg/'selection_reference.json.gz',self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']:
            if digest(f)==p['reference_sha256']:return f
        raise ValueError('Immutable reference missing')
    def freeze(self,number):
        file=self.cfg/f'round{number:02d}.json';planfile=self.docs/f'round{number:02d}_plan.json'
        if file.exists():
            plan=read_json(planfile);return file,self.reference(read_json(file)),plan
        base=read_json(self.root/'configs/experiments/skill_ablation_v1/incumbent.json');p=copy.deepcopy(base)
        batches=[41,42];arms='gradient';reason='';parent='historical_R26';spec={};extra={}
        if number in [1,17,19]:
            batches={1:[41,42],17:[43,44],19:[45,46]}[number];arms='unguided,gradient'
            reason='Matched native/R26 control panel with immutable baseline source and declared random streams.'
            if number==17:
                best=self.best(True,current_only=True);admissible=best is not None
                if best is None:best=self.best(current_only=True)
                selected=read_json(self.cfg/f'round{best:02d}.json');ref=self.reference(selected)
                write_json(self.docs/'frozen_validation.json',{'selected_round':best,'screening_admissible':admissible,
                  'program_sha256':digest(self.cfg/f'round{best:02d}.json'),'reference_sha256':digest(ref),
                  'confirmation_batches':[43,44,45,46],'master_seed':42,'before_confirmation_labels':True})
        elif number in [18,20]:
            frozen=read_json(self.docs/'frozen_validation.json');best=frozen['selected_round'];source=self.cfg/f'round{best:02d}.json'
            if digest(source)!=frozen['program_sha256']:raise ValueError('Frozen source mutated')
            p=read_json(source);batches=[43,44] if number==18 else [45,46];parent=best
            reason='Independent confirmation of the frozen proposal; no validation-driven retuning.'
        elif number==16:
            best=self.best(True)
            if best is None:best=self.best()
            p=read_json(self.cfg/f'round{best:02d}.json');parent=best
            reason='Refresh the best admissible parameter proposal under R26-matched Torch prior normalization before confirmation; if already current, this is an explicit reproducibility control, not a new dose change.'
        else:
            if number==2:
                for role in ['Analyst','Designer']:
                    folder=self.docs/'agents';request=read_json(folder/f'{role}.request.json');response=read_json(folder/f'{role}.response.json')
                    if request['input_sha256']!=digest(folder/f'{role}.input.json') or request['instruction_sha256']!=digest(folder/f'{role}.instructions.md'):
                        raise ValueError('Literal Agent inputs changed')
                    if any(response[k]!=request[k] for k in ['input_sha256','instruction_sha256']) or response['decision']!='test_registered_pilot':
                        raise ValueError('Agent deferred or stale response')
                    if response['accounting_representation']!='actual_current_and_proposal_world_A' or response['teacher_representation']!='predicted_endpoint_world_A':
                        raise ValueError('Agent merged current-state accounting and endpoint teachers')
                if response['selection_path']!={'quality':'rank'} or response['reward_view']!='endpoint_selection_path':raise ValueError('Designer changed registered pilot')
            spec,extra,reason=TRIALS[number];p.update(extra);p['reward_view']='endpoint_selection_path';p['selection_path']=spec
            if any(spec.get(k,0)>0 for k in ['precision_mix','niche_balance']) or spec.get('selection')=='niche':
                p['reference_sha256']=digest(self.cfg/'selection_reference.json.gz')
        p.update(round=number,program_id=f'selection_path20_round{number:02d}',seed=42,
                 derivation={'parent':parent,'single_mechanism':reason,'no_private_reasoning_transcript':True})
        write_json(file,p);reference=self.reference(p)
        plan={'round':number,'parent':parent,'reason':reason,'batches':batches,'arms':arms,'n_per_arm':100,
              'program_sha256':digest(file),'reference_sha256':digest(reference),'single_mechanism':p.get('selection_path',{}),
              'learned_window':p['window'],'baseline_default_until_confirmation':True}
        write_json(planfile,plan)
        # This is the actual trial router; a failed candidate is restored below.
        (self.cfg/'active_program.json').write_bytes(file.read_bytes())
        skills=read_json(self.cfg/'baseline_snapshot.json')['baseline_skills']
        if p['reward_view']=='endpoint_selection_path':skills['experimental_module']={'path':'skills/selection-pressure-path/SKILL.md','sha256':digest(self.root/'skills/selection-pressure-path/SKILL.md')}
        write_json(self.cfg/'active_skills.json',skills)
        write_json(self.cfg/'active_workflow.json',{'candidate_enabled':p['reward_view']!='endpoint_pointcloud' or number==16,
          'default':'temporary_registered_trial','program_sha256':digest(file),'reference_sha256':digest(reference),'skills':skills})
        return file,reference,plan
    def retire_local(self,number):
        verify_retention(self.docs/f'round{number:02d}/retention.json')
        folder=self.root/f'test/data/selection_path20/round{number:02d}';allowed=(self.root/'test/data/selection_path20').resolve()
        if not folder.resolve().is_relative_to(allowed) or folder.resolve()==allowed:raise ValueError('Unsafe local retirement')
        if folder.exists():shutil.rmtree(folder)
        for f in self.state.glob(f'round{number:02d}.tar.gz*'):f.unlink()
    def collect(self,number):
        campaign=f'selection_path_r{number:02d}';archive=self.state/f'round{number:02d}.tar.gz'
        for suffix in ['', '.json']:self.download(self.work+'/'+campaign+'.tar.gz'+suffix,str(archive)+suffix,number)
        dataset=self.root/f'test/data/selection_path20/round{number:02d}';evaluated=self.root/f'results/selection_path20/round{number:02d}'
        verified=verify_generation_archive(archive,str(archive)+'.json',dataset)
        cfg=read_json(dataset/'results'/campaign/'config.json')['experiment'];self.emit('local_evaluation',round=number)
        evaluate_isolated(self.root,dataset,campaign,evaluated,cfg,self.state/f'round{number:02d}.evaluation.log')
        baseline=self.docs/'round01/candidate_metrics.csv' if 2<=number<=16 else self.docs/f'round{number-1:02d}/candidate_metrics.csv' if number in [18,20] else None
        summary=retain_round(dataset,campaign,evaluated,self.docs/f'round{number:02d}',self.threshold,baseline)
        out=self.docs/f'round{number:02d}';ret=read_json(out/'retention.json');write_json(out/'archive_verification.json',verified)
        # Every round explicitly compares to existing Steer, without claiming pairing.
        summary['historical_Steer_reference']=read_json(self.docs/'steer_reference.json')
        admissible=False;gain=None
        if baseline:
            gain=summary['versus_gradient']['paired_mean_pic50'];m=summary['results']['gradient']
            admissible=gain>.005 and min(summary['versus_gradient']['batch_means'])>0 and m['valid_n']/m['n']>=.90 and m['pb_fast_rate']>=.90
            if number==6:admissible=False
        write_json(out/'comparison.json',summary)
        reason=('Candidate failed predeclared matched screening; restore R26 code path/program/Skills.' if baseline and not admissible
                else 'Provisional result retained; restore R26 default until frozen confirmation is complete.')
        rollback=restore_incumbent(self.root,'selection_path20_v1',reason,number);write_json(out/'rollback.json',rollback)
        for name in ['comparison.json','archive_verification.json','rollback.json']:
            ret['files']=[v for v in ret['files'] if v['path']!=name]
            ret['files'].append({'path':name,'sha256':digest(out/name),'bytes':(out/name).stat().st_size})
        write_json(out/'retention.json',ret)
        c=read_json(self.cfg/'campaign.json')
        if c['rounds_completed']!=number-1:raise ValueError('Sequential local evidence required')
        c['rounds_completed']=number;c['rounds'].append({'round':number,'status':'complete','kind':'actual_FLOWR_inference',
          'screening_admissible':admissible,'mean_vs_R26':gain,'results':summary,'retention_report':str((out/'retention.json').relative_to(self.root))})
        write_json(self.cfg/'campaign.json',c);self.emit('round_complete',round=number,mean_vs_R26=gain,admissible=admissible,results=summary['results'])
        self.push(f'Retain selection-path round {number}; restore immutable R26 workflow')
        self.retire_local(number)
    def wait_for_inference(self,number):
        q=shlex.quote
        while True:
            status=self.remote(f'if test -f {q(self.work+f"/round{number:02d}.exit")}; then cat {q(self.work+f"/round{number:02d}.exit")}; else echo running; fi')
            if status!='running':
                if status!='0':raise RuntimeError('Inference failed, retain raw state: '+self.remote(f'tail -n 30 {q(self.work+f"/round{number:02d}.log")}'))
                return
            time.sleep(30)
    def resume_existing(self,number):
        if read_json(self.cfg/'campaign.json')['rounds_completed']!=number-1:raise ValueError('Only the next unretained round can reconnect')
        active=json.loads(self.remote('cat '+shlex.quote(self.work+'/active.json')))
        if active['round']!=number or active['campaign']!=f'selection_path_r{number:02d}':raise ValueError('Active inference identity mismatch')
        self.emit('reconnected_existing_inference',round=number,original_inference_commit=active['inference_commit'],no_generation_rerun=True)
        self.wait_for_inference(number);self.collect(number)
        self.args.start=number+1
        if self.args.start<=self.args.last:self.run()
    def run(self):
        for number in range(self.args.start,self.args.last+1):
            if read_json(self.cfg/'campaign.json')['rounds_completed']!=number-1:raise ValueError('Previous report not retained')
            path,reference,plan=self.freeze(number);commit=self.push(f'Freeze R26-based selection-path test {number}')
            q=shlex.quote;command=['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python',self.remote_repo+'/scripts/dispatch_selection_round.py',
              '--repo',self.remote_repo,'--work',self.work,'--round',str(number),'--program',self.remote_repo+'/'+path.relative_to(self.root).as_posix(),
              '--reference',self.remote_repo+'/'+reference.relative_to(self.root).as_posix(),'--batches',','.join(map(str,plan['batches'])),
              '--arms',plan['arms'],'--n',str(plan['n_per_arm'])]
            if number>1:command+=['--previous',self.remote_repo+f'/docs/experiments/selection_path20_20261008/round{number-1:02d}/retention.json']
            self.remote('source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n'+' '.join(q(v) for v in command))
            self.emit('inference_started',round=number,commit=commit,batches=plan['batches'],arms=plan['arms'])
            self.wait_for_inference(number)
            self.collect(number)
        self.emit('requested_sequence_complete',last=self.args.last)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544)
    p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD');p.add_argument('--start',type=int,default=1);p.add_argument('--last',type=int,default=20)
    p.add_argument('--collect-only',type=int);p.add_argument('--resume-publish',type=int);p.add_argument('--resume-active',type=int)
    p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_selection_path20_20261008')
    a=p.parse_args()
    if not 1<=a.start<=a.last<=20:raise ValueError('Fresh twenty-round budget')
    d=Driver(a)
    try:
        if a.resume_publish is not None:
            n=a.resume_publish;verify_retention(d.docs/f'round{n:02d}/retention.json')
            if read_json(d.cfg/'campaign.json')['rounds_completed']!=n:raise ValueError('Only final retained publication can resume')
            d.push(f'Publish retained selection-path round {n} without a scientific rerun');d.retire_local(n)
        elif a.resume_active is not None:d.resume_existing(a.resume_active)
        elif a.collect_only is not None:d.collect(a.collect_only)
        else:d.run()
    except BaseException as error:
        # An infrastructure failure must not leave a hypothesis as the default.
        # Raw output remains for diagnosis; this does not consume a new round.
        state=restore_incumbent(d.root,'selection_path20_v1','Execution interrupted; restore baseline routing and retain raw data.',read_json(d.cfg/'campaign.json')['rounds_completed'])
        write_json(d.docs/'execution_interruption.json',{'error_type':type(error).__name__,'rollback':state,'raw_data_retained':True})
        raise
    finally:d.ssh.close()
