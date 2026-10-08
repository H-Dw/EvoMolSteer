"""Sequential local science, exact Git deployment, remote-only FLOWR inference."""
import argparse,copy,json,shlex,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.generation.path_evaluation import retain_round,paired_effect,execution_audit
from evomolsteer.generation.selection_workflow import restore_incumbent
from evomolsteer.storage.generation_archive import verify_generation_archive
from run_selection_path_campaign import Driver as BaseDriver
from run_sequential_path_campaign import evaluate_isolated
from dispatch_path_round import verify_retention
from evomolsteer.continuous.flow_response import conditional_formula_audit,mine_execution
from evomolsteer.continuous.flowcompat_agents import verify_execution_response

SPEC={
 3:('innovation','field_strength_A',.05),4:('innovation','field_strength_A',.15),
 5:('innovation','field_strength_A',.3),6:('innovation','field_strength_A',.6),
 7:('innovation','field_strength_A',1.)}
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
            not any(v in p.parts for v in ['joint_contrast_v1','branch_mutation_v1']) and p.name!='augmented_reference.json.gz']
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
        rows=[r for r in rows if 3<=r['round']<=24 and r['round'] not in [8,10] and (r['screening_admissible'] or not admissible_only)]
        return max(rows,key=lambda r:(r['mean_vs_R26'],-r['round']))['round'] if rows else None

    def reference(self,p):
        for f in [self.cfg/'branch_reference.json.gz',self.cfg/'innovation_reference.json.gz',self.root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']:
            if f.exists() and digest(f)==p['reference_sha256']:return f
        raise ValueError('Bound immutable teacher reference required')

    def freeze(self,number):
        file=self.cfg/f'round{number:02d}.json';planfile=self.docs/f'round{number:02d}_plan.json'
        if file.exists():return file,self.reference(read_json(file)),read_json(planfile)
        base=read_json(self.root/'configs/experiments/skill_ablation_v1/incumbent.json');p=copy.deepcopy(base)
        arms='gradient';batches=[47,48];parent='historical_R26';reason='';change={}
        if number in [1,2,8,25,27,29]:
            arms='unguided,gradient'
            if number==8:arms='gradient'
            if number>=25:batches=list(range(49+(number-25),51+(number-25)))
            reason='Paired native/R26 control; new-interface empty rule and measured Jacobian diagnostics.' if number in [2,8] else 'Original matched native/R26 baseline and frozen initial random streams.'
            if number==8:p['generation_interface']='flowcompat_v2'
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
        elif number>=9:
            parity=read_json(self.docs/'round08/implementation_feedback.json')
            if not parity['null_control_parity']:raise ValueError('V2 numerical null control must pass first')
            binding=read_json(self.cfg/'supplemental_binding.json')
            gate=read_json(self.root/binding['validation_path'])
            if not gate['passed']:raise ValueError('Actual supplemental Designer response required')
            from evomolsteer.continuous.flowcompat_supplemental import validate_response
            validate_response(self.root/binding['request_path'],self.root/binding['response_path'])
            specs=read_json(self.cfg/'adaptive_schedule_v2.json')['rounds']
            trial=specs[str(number)];parent=trial['parent'];reason=trial['reason']
            if parent=='branch_pilot':p=read_json(self.cfg/'branch_pilot.json')
            elif isinstance(parent,int):p=read_json(self.cfg/f'round{parent:02d}.json')
            else:p=copy.deepcopy(base)
            if number==9:
                p=read_json(self.cfg/'branch_pilot.json');change=p['flowcompat_provenance']['parameter_updates']
            else:
                group,key=trial['axis'].split('.') if '.' in trial['axis'] else ('',trial['axis'])
                value=trial['value']
                if trial.get('dose_match_gain_round'):
                    gain_program=read_json(self.cfg/f"round{trial['dose_match_gain_round']:02d}.json")
                    s=gain_program['flow_control']['jacobian_gain_saturation']
                    table=pd.read_parquet(self.docs/'round02/flow_response.parquet')
                    weight=table['predictive_flow_rms_A'].to_numpy();g=table['flowcompat_jacobian_gain'].to_numpy()
                    factor=float(np.sum(weight*g/(g+s))/np.sum(weight))
                    value=base['native_rms_ratio']*factor
                    reason+=' Frozen batch/time baseline proxy budget factor='+str(factor)+'; actual dose is separately compared.'
                if group:p.setdefault(group,{})[key]=value
                else:p[key]=value
                if group=='branch_mixture':
                    p['reward_view']='endpoint_branch_mixture';p['reference_sha256']=digest(self.cfg/'branch_reference.json.gz')
                provenance=copy.deepcopy(read_json(self.cfg/'branch_pilot.json')['flowcompat_provenance'])
                provenance.update(parameter_updates={trial['axis']:value},parameter_origin='Sequential engineering control from actual bound supplemental design; not a fresh LLM call',
                    formula_id=trial['formula_id'])
                registry=read_json(self.root/binding['registry_path'])
                formula=registry['formulas'][trial['formula_id']]
                if trial['axis'] not in formula['allowed_updates']:raise ValueError('Trial axis outside actually registered design family')
                provenance.update(reference_kind=formula['reference_kind'],code_files=formula['code_files'])
                p['reward_view']=formula['reward_view']
                p['flowcompat_provenance']=provenance;change={trial['axis']:value}
            p['generation_interface']='flowcompat_v2'
        else:
            parity=read_json(self.docs/'round02/implementation_feedback.json')
            if not parity['null_control_parity']:raise ValueError('No-op numerical control must pass before mechanism trials')
            binding=read_json(self.cfg/'agent_binding.json');gate=read_json(self.root/binding['validation_path'])
            if not gate['passed']:raise ValueError('Actual bound Designer/tool response required')
            winner=self.best()
            if winner is not None:p=read_json(self.cfg/f'round{winner:02d}.json');parent=winner
            group,key,value=SPEC[number];p.setdefault(group,{})[key]=value
            if group=='innovation':
                p['reward_view']='endpoint_innovation';p['reference_sha256']=digest(self.cfg/'innovation_reference.json.gz')
            change={group:{key:value}};reason=REASONS[key]
            p['agent_request_sha256']=gate['request_sha256'];p['agent_response_sha256']=gate['response_sha256']
            pilot=read_json(self.cfg/'designer_pilot.json')
            if number==3:
                p=pilot;change=pilot['flowcompat_provenance']['parameter_updates']
                reason='Actual Designer-compiled first pilot; literal input, tools, formula and code bound.'
            else:
                p['flowcompat_provenance']=copy.deepcopy(pilot['flowcompat_provenance'])
                p['flowcompat_provenance'].update(parameter_updates={group+'.'+key:value},
                    parameter_origin='Sequential registered engineering trial of Agent-vetted design families, not an independent LLM response')
        p.update(round=number,program_id=f'flowcompat30_round{number:02d}',seed=42,
            derivation={'parent':parent,'scientific_hypothesis':reason,'changed_axis':change,'private_reasoning_transcript':False})
        write_json(file,p);reference=self.reference(p)
        plan={'round':number,'parent':parent,'reason':reason,'batches':batches,'arms':arms,'n_per_arm':100,
            'changed_axis':change,'program_sha256':digest(file),'reference_sha256':digest(reference),
            'learned_window':p['window'],'design_origin':'Actual bound LLM design families with sequential engineering parameter tests; not thirty independent LLM calls.'}
        write_json(planfile,plan);(self.cfg/'active_program.json').write_bytes(file.read_bytes())
        write_json(self.cfg/'active_workflow.json',{'candidate_enabled':number not in [1,2,8,25,27,29],
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
                with np.load(old,allow_pickle=False) as z,np.load(window,allow_pickle=False) as current:
                    delta=coords-z['coords'].astype(float)*float(z['coord_scale'])
                    if number in [2,8]:
                        null &= all(z[key].dtype==current[key].dtype and z[key].tobytes()==current[key].tobytes() for key in z.files)
                rms=np.sqrt((delta**2).sum(-1).mean(1));distances.append({'arm':arm,'batch':batch,'mean_RMS_A':float(rms.mean()),'max_RMS_A':float(rms.max())})
                null &= bool(np.array_equal(delta,np.zeros_like(delta)))
            trace=[json.loads(v) for v in (folder/'guidance_trace.jsonl').read_text().splitlines()]
            active=[r for r in trace if r['reward_evaluated']]
            fields=['contrast_teacher_shift_rms_A','flowcompat_native_cosine_before','flowcompat_native_cosine_after',
                'flowcompat_gradient_adjustment_relative_rms','flowcompat_schedule_factor','flowcompat_jacobian_gain','flowcompat_gradient_lag_cosine',
                'branch_virtual_mass_mean','branch_virtual_teacher_shift_rms_A','branch_region_weight_rms_difference']
            rows[-1]['active_steps']=len(active)
            rows[-1]['mechanisms']={k:float(np.mean([r[k] for r in active if k in r])) for k in fields if any(k in r for r in active)}
            rows[-1]['module_sha256']=active[0].get('flowcompat_module_sha256') if active else None
        mechanism_response=any(r['mean_RMS_A']>1e-6 for r in distances if r['arm']=='gradient')
        report={'schema_version':'implementation-before-efficacy-1.0','round':number,
            'null_control_parity':null if number in [2,8] else None,'actual_window_coordinate_response':mechanism_response,
            'window_differences_against_paired_R26':distances,'batch_diagnostics':rows,
            'interpretation':'Response proves deployment/perturbation, not beneficial affinity or causal module importance.'}
        write_json(output/'implementation_feedback.json',report);return report

    def collect(self,number):
        campaign=f'flowcompat_r{number:02d}';archive=self.state/f'round{number:02d}.tar.gz'
        for suffix in ['', '.json']:self.download(self.work+'/'+campaign+'.tar.gz'+suffix,str(archive)+suffix,number)
        dataset=self.root/f'test/data/flowcompat30/round{number:02d}';evaluated=self.root/f'results/flowcompat30/round{number:02d}'
        verified=verify_generation_archive(archive,str(archive)+'.json',dataset)
        run=dataset/'results'/campaign;cfg=read_json(run/'config.json')['experiment']
        evaluated.mkdir(parents=True,exist_ok=True)
        feedback=self.feedback(dataset,campaign,number,evaluated)
        mine_execution(dataset,campaign,evaluated)
        program=self.cfg/f'round{number:02d}.json';p=read_json(program)
        formula=conditional_formula_audit(program,self.reference(p));write_json(evaluated/'conditional_formula_audit.json',formula)
        feedback['conditional_formula_audit']=formula
        execution=execution_audit(dataset,campaign)
        if number>=3 and 'flowcompat_provenance' in p:
            import hashlib,subprocess
            code_files=p['flowcompat_provenance']['code_files'];measured_code={}
            for file in code_files:
                blob=subprocess.check_output(['git','-C',str(self.root),'show',execution['code_commit']+':'+file])
                measured_code[file]=hashlib.sha256(blob).hexdigest()
            diagnostics={k:float(np.mean([r['mechanisms'][k] for r in feedback['batch_diagnostics'] if k in r['mechanisms']]))
                for k in ['contrast_teacher_shift_rms_A','flowcompat_gradient_adjustment_relative_rms']
                if any(k in r['mechanisms'] for r in feedback['batch_diagnostics'])}
            diagnostics['paired_window_coordinate_rms_A']=float(np.mean([r['mean_RMS_A'] for r in feedback['window_differences_against_paired_R26']]))
            diagnostics['contrast_gradient_relative_change']=formula['maximum_gradient_relative_change']
            diagnostics['branch_gradient_relative_change']=formula['maximum_gradient_relative_change']
            for key in ['branch_virtual_mass_mean','branch_virtual_teacher_shift_rms_A']:
                values=[r['mechanisms'][key] for r in feedback['batch_diagnostics'] if key in r['mechanisms']]
                if values:diagnostics[key]=float(np.mean(values))
            values=[r['mechanisms']['branch_region_weight_rms_difference'] for r in feedback['batch_diagnostics'] if 'branch_region_weight_rms_difference' in r['mechanisms']]
            if values:diagnostics['branch_region_weight_rms_change']=float(np.mean(values))
            base=read_json(self.root/'configs/experiments/skill_ablation_v1/incumbent.json')
            diagnostics['guidance_dose_relative_change']=abs(p['native_rms_ratio']/base['native_rms_ratio']-1)
            null_round=8 if p.get('generation_interface')=='flowcompat_v2' else 2
            diagnostics['flowcompat_schedule_absolute_change']=float(np.mean([abs(1-r['mechanisms']['flowcompat_schedule_factor']) for r in feedback['batch_diagnostics'] if 'flowcompat_schedule_factor' in r['mechanisms']]))
            gate_input={'program_sha256':digest(program),'request_sha256':p['flowcompat_provenance']['request_sha256'],
                'code_files':measured_code,'execution_checks':{k:True for k in ['actual_flowr_model','actual_endpoint_vjp','no_resampling','no_head_gradient','no_extra_production_forward','window_matches','initial_state_pair_matches']},
                'no_op_control':{'exact_baseline_arithmetic':formula['null_exact'] and read_json(self.docs/f'round{null_round:02d}/implementation_feedback.json')['null_control_parity'],
                    'max_gradient_relative_change':0 if formula['null_exact'] else None},'diagnostics':diagnostics}
            write_json(evaluated/'implementation_gate_input.json',gate_input)
            try:
                checker=verify_execution_response
                if p['flowcompat_provenance'].get('workflow_version')=='flowcompat-supplemental-agent-2.0':
                    from evomolsteer.continuous.flowcompat_supplemental import verify_execution_response as checker
                checker(program,evaluated/'implementation_gate_input.json',evaluated/'implementation_gate.json')
                feedback['implementation_passed']=True
            except ValueError as error:
                feedback['implementation_passed']=False;feedback['implementation_rejection']=str(error)
                write_json(evaluated/'implementation_gate.json',{'implementation_passed':False,'reason':str(error),'efficacy_assessed':False})
        self.emit('implementation_checked_before_affinity',round=number,passed=feedback.get('implementation_passed'),
            actual_window_coordinate_response=feedback['actual_window_coordinate_response'])
        write_json(evaluated/'implementation_feedback.json',feedback)
        self.emit('local_evaluation',round=number)
        evaluate_isolated(self.root,dataset,campaign,evaluated,cfg,self.state/f'round{number:02d}.evaluation.log')
        baseline=self.docs/'round01/candidate_metrics.csv' if 2<=number<=24 else self.docs/f'round{number-1:02d}/candidate_metrics.csv' if number in [26,28,30] else None
        summary=retain_round(dataset,campaign,evaluated,self.docs/f'round{number:02d}',self.threshold,baseline)
        out=self.docs/f'round{number:02d}'
        # Preserve the pre-efficacy gate as its own retained artifact, not only
        # inside the growing campaign ledger. Confirmation reads this file.
        write_json(out/'implementation_feedback.json',feedback)
        for name in ['flow_response.parquet','decoder_sensitivity.parquet','flow_response_summary.json','conditional_formula_audit.json','implementation_gate_input.json','implementation_gate.json']:
            if (evaluated/name).exists():shutil.copy2(evaluated/name,out/name)
        if number in [2,8]:
            original=pd.read_csv(self.docs/'round01/candidate_metrics.csv');new=pd.read_csv(out/'candidate_metrics.csv')
            original=original[original.arm.isin(new.arm.unique())].reset_index(drop=True)
            new=new.reset_index(drop=True)
            cols=['arm','batch','slot','pic50_on_rescore','smiles']
            parity=original[cols].equals(new[cols]);feedback['null_control_terminal_parity']=parity
            feedback['null_control_parity'] &= parity;write_json(out/'implementation_feedback.json',feedback)
        summary['historical_Steer_reference']=read_json(self.docs/'steer_reference.json')
        gain=summary.get('versus_gradient',{}).get('paired_mean_pic50');admissible=False
        if baseline and number>=3 and number not in [8,10]:
            m=summary['results']['gradient'];admissible=(gain>.005 and min(summary['versus_gradient']['batch_means'])>0 and m['valid_n']/m['n']>=.9 and m['pb_fast_rate']>=.9 and feedback.get('implementation_passed',False))
        write_json(out/'comparison.json',summary);write_json(out/'archive_verification.json',verified)
        rollback=restore_incumbent(self.root,'flowcompat30_v1','Candidate remains provisional; failed proposals never overwrite R26.',number)
        write_json(out/'rollback.json',rollback)
        retention=read_json(out/'retention.json')
        for name in ['comparison.json','archive_verification.json','implementation_feedback.json','rollback.json',
            'flow_response.parquet','decoder_sensitivity.parquet','flow_response_summary.json','conditional_formula_audit.json','implementation_gate_input.json','implementation_gate.json']:
            if not (out/name).exists():continue
            retention['files']=[r for r in retention['files'] if r['path']!=name]
            retention['files'].append({'path':name,'sha256':digest(out/name),'bytes':(out/name).stat().st_size})
        write_json(out/'retention.json',retention)
        c=read_json(self.cfg/'campaign.json')
        if c['rounds_completed']!=number-1:raise ValueError('Sequential retention required')
        c['rounds_completed']=number;c['rounds'].append({'round':number,'status':'complete','kind':'actual_FLOWR_inference',
            'screening_admissible':admissible,'mean_vs_R26':gain,'results':summary,'implementation_feedback':feedback})
        write_json(self.cfg/'campaign.json',c);self.emit('round_complete',round=number,mean_vs_R26=gain,admissible=admissible,results=summary['results'])
        from report_flowcompat_campaign import report
        report(self.root)
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
    except BaseException as error:
        state=restore_incumbent(d.root,'flowcompat30_v1','Execution interrupted; preserve raw outputs and restore baseline.',read_json(d.cfg/'campaign.json')['rounds_completed'])
        write_json(d.docs/'execution_interruption.json',{'error_type':type(error).__name__,'rollback':state,'raw_data_retained':True})
        raise
    finally:d.ssh.close()
