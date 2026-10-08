"""Sequential local science, exact Git deployment, remote-only FLOWR inference."""
import argparse,copy,json,shlex,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import retain_round,paired_effect
from evomolsteer.generation.selection_workflow import restore_incumbent
from evomolsteer.storage.generation_archive import verify_generation_archive
from run_selection_path_campaign import Driver as BaseDriver
from run_sequential_path_campaign import evaluate_isolated
from dispatch_path_round import verify_retention

SPEC={
 3:('innovation','field_strength_A',.05),4:('innovation','field_strength_A',.15),
 5:('innovation','field_strength_A',.3),6:('innovation','field_strength_A',.6),
 7:('innovation','field_strength_A',1.),8:('innovation','reliability_power',.5),
 9:('innovation','region_weight_mix',.1),10:('innovation','region_weight_mix',.25),
 11:('innovation','region_weight_mix',.5),12:('innovation','region_weight_mix',.75),
 13:('flow_control','parallel_component_scale',.5),14:('flow_control','parallel_component_scale',0.),
 15:('flow_control','parallel_component_scale',1.5),16:('flow_control','parallel_component_scale',2.),
 17:('flow_control','time_envelope_power',.25),18:('flow_control','time_envelope_power',.5),
 19:('flow_control','time_envelope_power',1.),20:('flow_control','time_envelope_power',2.),
 21:('flow_control','jacobian_gain_saturation',.1),22:('flow_control','jacobian_gain_saturation',.3),
 23:('flow_control','jacobian_gain_saturation',1.),24:('flow_control','jacobian_gain_saturation',3.)}
REASONS={
 'field_strength_A':'Bounded extrapolation from higher endpoint toward its coordinate contrast against local lower-scoring candidates; conditional empirical direction, not physical force or causal affinity gradient.',
 'reliability_power':'Reduce excessive confidence attenuation only after measuring the pilot direction; cannot turn unobserved late support into established advantage.',
 'region_weight_mix':'Emphasize atoms carrying joint local coordinate contrast, preserving mean cost scale and complete teacher conformations.',
 'parallel_component_scale':'Separate gradient acceleration along native flow from orthogonal spatial redirection; no chemical graph veto.',
 'time_envelope_power':'Taper relative dose as learned support ends, testing compatibility with later unconstrained native continuation; every learned step remains eligible.',
 'jacobian_gain_saturation':'Retain model response amplitude that RMS normalization removes; bounded real VJP/endpoint-gradient gain, not covariance learned from teacher dispersion.'}


class Driver(BaseDriver):
    def __init__(self,a):
        self.args=a;self.root=Path(a.repo).resolve();self.cfg=self.root/'configs/experiments/flowcompat30_v1'
        self.docs=self.root/'docs/experiments/flowcompat30_20261009';self.state=self.root/'test/flowcompat30_driver'
        self.state.mkdir(parents=True,exist_ok=True);self.work=a.remote_work;self.remote_repo=a.remote_repo
        self.threshold=read_json(self.docs/'protocol.json')['tail_threshold_pic50'];self.connect()

    def push(self,message):
        self.local(['git','add',str(self.cfg.relative_to(self.root))])
        # Preliminary/raw mining payloads never enter the published report set.
        paths=[p for p in self.docs.rglob('*') if p.is_file() and
            'joint_contrast_v1' not in p.parts and p.name!='augmented_reference.json.gz']
        if paths:self.local(['git','add','-f',*[str(p.relative_to(self.root)) for p in paths]])
        self.local(['git','diff','--cached','--check'])
        if self.local(['git','diff','--cached','--name-only']):self.local(['git','commit','-m',message])
        self.local(['git','-c','http.proxy=http://127.0.0.1:7897','push','origin','main'])
        commit=self.local(['git','rev-parse','HEAD']);q=shlex.quote
        self.remote(f'timeout 180 git -C {q(self.remote_repo)} -c http.proxy=http://127.0.0.1:17897 pull --ff-only origin main')
        if self.remote(f'git -C {q(self.remote_repo)} rev-parse HEAD')!=commit:raise ValueError('Exact deployment commit required')
        return commit

    def best(self,admissible_only=True):
        rows=read_json(self.cfg/'campaign.json')['rounds']
        rows=[r for r in rows if 3<=r['round']<=24 and (r['screening_admissible'] or not admissible_only)]
        return max(rows,key=lambda r:(r['mean_vs_R26'],-r['round']))['round'] if rows else None

    def reference(self,p):
        for f in [self.cfg/'innovation_reference.json.gz',self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']:
            if f.exists() and digest(f)==p['reference_sha256']:return f
        raise ValueError('Bound immutable teacher reference required')

    def freeze(self,number):
        file=self.cfg/f'round{number:02d}.json';planfile=self.docs/f'round{number:02d}_plan.json'
        if file.exists():return file,self.reference(read_json(file)),read_json(planfile)
        base=read_json(self.root/'configs/experiments/skill_ablation_v1/incumbent.json');p=copy.deepcopy(base)
        arms='gradient';batches=[47,48];parent='historical_R26';reason='';change={}
        if number in [1,2,25,27,29]:
            arms='unguided,gradient'
            if number>=25:batches=list(range(49+(number-25),51+(number-25)))
            reason='Paired native/R26 control; new-interface empty rule and measured Jacobian diagnostics.' if number==2 else 'Original matched native/R26 baseline and frozen initial random streams.'
            if number==25:
                winner=self.best();eligible=winner is not None
                if winner is None:winner=self.best(False)
                if winner is None:raise ValueError('No measured candidate available')
                source=self.cfg/f'round{winner:02d}.json'
                write_json(self.docs/'frozen_validation.json',{'selected_round':winner,'screening_admissible':eligible,
                    'program_sha256':digest(source),'reference_sha256':read_json(source)['reference_sha256'],
                    'before_confirmation_labels':True,'confirmation_batches':[49,50,51,52,53,54]})
        elif number in [26,28,30]:
            frozen=read_json(self.docs/'frozen_validation.json');parent=frozen['selected_round']
            source=self.cfg/f'round{parent:02d}.json'
            if digest(source)!=frozen['program_sha256']:raise ValueError('Frozen candidate changed')
            p=read_json(source);batches=list(range(49+(number-26),51+(number-26)))
            reason='Independent confirmation of unchanged frozen reward and controller; no retuning on confirmation labels.'
        else:
            parity=read_json(self.docs/'round02/implementation_feedback.json')
            if not parity['null_control_parity']:raise ValueError('No-op numerical control must pass before mechanism trials')
            gate=read_json(self.docs/'agents/Designer.validation.json')
            if not gate['passed']:raise ValueError('Actual bound Designer/tool response required')
            winner=self.best()
            if winner is not None:p=read_json(self.cfg/f'round{winner:02d}.json');parent=winner
            group,key,value=SPEC[number];p.setdefault(group,{})[key]=value
            if group=='innovation':
                p['reward_view']='endpoint_innovation';p['reference_sha256']=digest(self.cfg/'innovation_reference.json.gz')
            change={group:{key:value}};reason=REASONS[key]
            p['agent_request_sha256']=gate['request_sha256'];p['agent_response_sha256']=gate['response_sha256']
        p.update(round=number,program_id=f'flowcompat30_round{number:02d}',seed=42,
            derivation={'parent':parent,'scientific_hypothesis':reason,'changed_axis':change,'private_reasoning_transcript':False})
        write_json(file,p);reference=self.reference(p)
        plan={'round':number,'parent':parent,'reason':reason,'batches':batches,'arms':arms,'n_per_arm':100,
            'changed_axis':change,'program_sha256':digest(file),'reference_sha256':digest(reference),
            'learned_window':p['window'],'design_origin':'Actual bound LLM design families with sequential engineering parameter tests; not thirty independent LLM calls.'}
        write_json(planfile,plan);(self.cfg/'active_program.json').write_bytes(file.read_bytes())
        write_json(self.cfg/'active_workflow.json',{'candidate_enabled':number not in [1,2,25,27,29],
            'temporary_trial':True,'program_sha256':digest(file),'default_after_round':'historical_R26'})
        return file,reference,plan

    def feedback(self,dataset,campaign,number,output):
        root=dataset/'results'/campaign;states=self.state/'control_states';states.mkdir(exist_ok=True)
        rows=[];distances=[];null=True
        for folder in sorted(root.glob('*/batch_*')):
            batch=int(folder.name.split('_')[-1]);arm=folder.parent.name
            window=folder/'window_state.npz';key=f'{arm}_{batch:03d}.npz'
            with np.load(window,allow_pickle=False) as z:
                signature=digest(window);coords=z['coords'].astype(float)*float(z['coord_scale'])
            rows.append({'arm':arm,'batch':batch,'window_state_sha256':signature})
            old=states/key
            if number in [1,25,27,29]:shutil.copy2(window,old)
            elif old.exists():
                with np.load(old,allow_pickle=False) as z:delta=coords-z['coords'].astype(float)*float(z['coord_scale'])
                rms=np.sqrt((delta**2).sum(-1).mean(1));distances.append({'arm':arm,'batch':batch,'mean_RMS_A':float(rms.mean()),'max_RMS_A':float(rms.max())})
                null &= bool(np.array_equal(delta,np.zeros_like(delta)))
            trace=[json.loads(v) for v in (folder/'guidance_trace.jsonl').read_text().splitlines()]
            active=[r for r in trace if r['reward_evaluated']]
            fields=['contrast_teacher_shift_rms_A','flowcompat_native_cosine_before','flowcompat_native_cosine_after',
                'flowcompat_gradient_adjustment_relative_rms','flowcompat_schedule_factor','flowcompat_jacobian_gain','flowcompat_gradient_lag_cosine']
            rows[-1]['active_steps']=len(active)
            rows[-1]['mechanisms']={k:float(np.mean([r[k] for r in active if k in r])) for k in fields if any(k in r for r in active)}
            rows[-1]['module_sha256']=active[0].get('flowcompat_module_sha256') if active else None
        mechanism_response=any(r['mean_RMS_A']>1e-6 for r in distances if r['arm']=='gradient')
        report={'schema_version':'implementation-before-efficacy-1.0','round':number,
            'null_control_parity':null if number==2 else None,'actual_window_coordinate_response':mechanism_response,
            'window_differences_against_paired_R26':distances,'batch_diagnostics':rows,
            'interpretation':'Response proves deployment/perturbation, not beneficial affinity or causal module importance.'}
        write_json(output/'implementation_feedback.json',report);return report

    def collect(self,number):
        campaign=f'flowcompat_r{number:02d}';archive=self.state/f'round{number:02d}.tar.gz'
        for suffix in ['', '.json']:self.download(self.work+'/'+campaign+'.tar.gz'+suffix,str(archive)+suffix,number)
        dataset=self.root/f'test/data/flowcompat30/round{number:02d}';evaluated=self.root/f'results/flowcompat30/round{number:02d}'
        verified=verify_generation_archive(archive,str(archive)+'.json',dataset)
        cfg=read_json(dataset/'results'/campaign/'config.json')['experiment'];self.emit('local_evaluation',round=number)
        evaluate_isolated(self.root,dataset,campaign,evaluated,cfg,self.state/f'round{number:02d}.evaluation.log')
        baseline=self.docs/'round01/candidate_metrics.csv' if 2<=number<=24 else self.docs/f'round{number-1:02d}/candidate_metrics.csv' if number in [26,28,30] else None
        summary=retain_round(dataset,campaign,evaluated,self.docs/f'round{number:02d}',self.threshold,baseline)
        out=self.docs/f'round{number:02d}';feedback=self.feedback(dataset,campaign,number,out)
        if number==2:
            original=pd.read_csv(self.docs/'round01/candidate_metrics.csv');new=pd.read_csv(out/'candidate_metrics.csv')
            cols=['arm','batch','slot','pic50_on_rescore','smiles']
            parity=original[cols].equals(new[cols]);feedback['null_control_terminal_parity']=parity
            feedback['null_control_parity'] &= parity;write_json(out/'implementation_feedback.json',feedback)
        summary['historical_Steer_reference']=read_json(self.docs/'steer_reference.json')
        gain=summary.get('versus_gradient',{}).get('paired_mean_pic50');admissible=False
        if baseline and number>=3:
            m=summary['results']['gradient'];admissible=(gain>.005 and min(summary['versus_gradient']['batch_means'])>0 and m['valid_n']/m['n']>=.9 and m['pb_fast_rate']>=.9 and feedback['actual_window_coordinate_response'])
        write_json(out/'comparison.json',summary);write_json(out/'archive_verification.json',verified)
        rollback=restore_incumbent(self.root,'flowcompat30_v1','Candidate remains provisional; failed proposals never overwrite R26.',number)
        write_json(out/'rollback.json',rollback)
        retention=read_json(out/'retention.json')
        for name in ['comparison.json','archive_verification.json','implementation_feedback.json','rollback.json']:
            retention['files']=[r for r in retention['files'] if r['path']!=name]
            retention['files'].append({'path':name,'sha256':digest(out/name),'bytes':(out/name).stat().st_size})
        write_json(out/'retention.json',retention)
        c=read_json(self.cfg/'campaign.json')
        if c['rounds_completed']!=number-1:raise ValueError('Sequential retention required')
        c['rounds_completed']=number;c['rounds'].append({'round':number,'status':'complete','kind':'actual_FLOWR_inference',
            'screening_admissible':admissible,'mean_vs_R26':gain,'results':summary,'implementation_feedback':feedback})
        write_json(self.cfg/'campaign.json',c);self.emit('round_complete',round=number,mean_vs_R26=gain,admissible=admissible,results=summary['results'])
        self.push(f'Retain flow-compatibility round {number}; restore immutable R26')
        self.retire_local(number)

    def retire_local(self,number):
        verify_retention(self.docs/f'round{number:02d}/retention.json')
        folder=self.root/f'test/data/flowcompat30/round{number:02d}';allowed=(self.root/'test/data/flowcompat30').resolve()
        if not folder.resolve().is_relative_to(allowed) or folder.resolve()==allowed:raise ValueError('Unsafe retirement')
        if folder.exists():shutil.rmtree(folder)
        for f in self.state.glob(f'round{number:02d}.tar.gz*'):f.unlink()

    def run(self):
        for number in range(self.args.start,self.args.last+1):
            if read_json(self.cfg/'campaign.json')['rounds_completed']!=number-1:raise ValueError('Prior round not retained')
            path,reference,plan=self.freeze(number);commit=self.push(f'Freeze flow-compatibility hypothesis {number}')
            q=shlex.quote;command=['/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python',self.remote_repo+'/scripts/dispatch_flowcompat_round.py',
                '--repo',self.remote_repo,'--work',self.work,'--round',str(number),'--program',self.remote_repo+'/'+path.relative_to(self.root).as_posix(),
                '--reference',self.remote_repo+'/'+reference.relative_to(self.root).as_posix(),'--batches',','.join(map(str,plan['batches'])),'--arms',plan['arms'],'--n','100']
            if number>1:command+=['--previous',self.remote_repo+f'/docs/experiments/flowcompat30_20261009/round{number-1:02d}/retention.json']
            self.remote('source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n'+' '.join(q(v) for v in command))
            self.emit('inference_started',round=number,commit=commit,plan=plan)
            self.wait_for_inference(number);self.collect(number)
        self.emit('requested_sequence_complete',last=self.args.last)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--host',default='ksai.scnet.cn');p.add_argument('--port',type=int,default=10544)
    p.add_argument('--password-env',default='MOLSTEER_SCNET_PASSWORD');p.add_argument('--start',type=int,default=1);p.add_argument('--last',type=int,default=30)
    p.add_argument('--collect-only',type=int);p.add_argument('--remote-repo',default='/root/private_data/MolSteer/EvoMolSteer')
    p.add_argument('--remote-work',default='/root/private_data/MolSteer/flowr_root/experiments/evomolsteer_flowcompat30_20261009')
    a=p.parse_args()
    if not 1<=a.start<=a.last<=30:raise ValueError('Fresh thirty-round budget')
    d=Driver(a)
    try:
        if a.collect_only:d.collect(a.collect_only)
        else:d.run()
    finally:d.ssh.close()
