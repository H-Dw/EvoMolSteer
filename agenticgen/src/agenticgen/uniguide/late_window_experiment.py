"""Eight paired raw-output arms extending guidance only at steps 91..99.

Historical science is imported exclusively from the frozen 0e7d4dc2 checkout.
No new Agent search, lambda calibration, coordinate inpainting or retries.
"""
import copy
import functools
import os
from pathlib import Path
import subprocess
import types

import torch

from . import time_window_experiment as base
from .io import frozen, read, sha, value_sha, write

POLICY = 'historical-coordinate-late-window-raw-output-v1'
ENDPOINTS = ('raw1', 'raw999')
COVERAGES = ('window', 'late_only', 'full_interior')


def arms():
    return {base.arm_name(c,e,w):dict(chemistry=c,endpoint=e,coverage=w)
            for c in base.CHEMISTRIES for e in ENDPOINTS for w in COVERAGES}


def contrasts():
    result=[]
    for c in base.CHEMISTRIES:
        for e in ENDPOINTS:
            for a,b,label in [('window','late_only','late_extension'),
                              ('late_only','full_interior','early_extension_with_late_fixed')]:
                result.append((base.arm_name(c,e,a),base.arm_name(c,e,b),label))
        for w in COVERAGES:
            result.append((base.arm_name(c,'raw1',w),base.arm_name(c,'raw999',w),'grid_only_bridge'))
    return result


def prepare(root,source,prior,frozen_repo,execution_repo):
    prior_spec=read(prior/'STUDY.json');final=read(prior/'FINALIZED.json')
    if final['results_sha256']!=sha(prior/'RESULTS.json'):
        raise ValueError('Prior finalized results changed')
    if prior_spec['policy']!='historical-coordinate-time-window-factorial-v1':
        raise ValueError('Wrong historical comparison protocol')
    base.verify_sources(prior_spec)
    head=subprocess.check_output(['git','-C',str(frozen_repo),'rev-parse','HEAD'],text=True).strip()
    if head!=base.FROZEN_COMMIT or subprocess.check_output(['git','-C',str(frozen_repo),'status','--porcelain']).strip():
        raise ValueError('Frozen scientific checkout differs')
    if subprocess.check_output(['git','-C',str(execution_repo),'status','--porcelain','--untracked-files=no']).strip():
        raise ValueError('Execution checkout must be clean')
    commit=subprocess.check_output(['git','-C',str(execution_repo),'rev-parse','HEAD'],text=True).strip()
    package=frozen_repo/'agenticgen/src/agenticgen/uniguide'
    if Path(base.sampler.__file__).resolve()!=package/'sampler.py':
        raise ValueError('Wrong scientific package loaded')
    config=copy.deepcopy(read(prior/'config.json'))
    config.pop('time_window_experiment',None)
    config.update(root=str(root),late_window_experiment=POLICY)
    if read(source/'teachers.json')!=read(prior/'teachers.json'):
        raise ValueError('Teacher identity changed')
    teachers=read(source/'teachers.json')
    if len(teachers)!=2 or {t['target'] for t in teachers}!=set(prior_spec['controls']):
        raise ValueError('Expected the two frozen teachers')
    protected=dict(prior_spec['protected_files'])
    for name in ('STUDY.json','FINALIZED.json','RESULTS.json','attempts.json','config.json','teachers.json'):
        protected[str(prior/name)]=sha(prior/name)
    for path in (execution_repo/'agenticgen/src/agenticgen/uniguide/time_window_experiment.py',
                 execution_repo/'agenticgen/src/agenticgen/uniguide/late_window_experiment.py',
                 execution_repo/'agenticgen/scripts/run_uniguide_late_window.py'):
        protected[str(path)]=sha(path)
    references={};reused=failed=0
    details=read(prior/'RESULTS.json')['groups']
    for teacher in teachers:
        target=teacher['target'];references[target]={}
        for c in base.CHEMISTRIES:
            for e in ENDPOINTS:
                for w in ('window','full_interior'):
                    arm=base.arm_name(c,e,w);entry=details[target+'/'+arm];folder=Path(entry['folder'])
                    if entry['status'] not in ('complete','failed'):
                        raise ValueError('A reference is not terminal')
                    if entry['status']=='complete':
                        receipt=read(folder/'COMPLETE.json');records=read(folder/'records.json')
                        if receipt['records_sha256']!=sha(folder/'records.json'):
                            raise ValueError('Reference records differ')
                        if len(records)!=50 or {(r['logical_seed'],r['logical_slot']) for r in records}!={(s,i) for s in base.SEEDS for i in range(10)}:
                            raise ValueError('Reference paired slots differ')
                        quality=read(folder/'QUALITY_COMPLETE.json')
                        if any(quality.get(k)!=v for k,v in dict(attempts=50,checked_attempts=10,not_selected=40,coverage=.2).items()):
                            raise ValueError('Reference diagnostic coverage differs')
                        reused+=50
                    else:
                        failure=read(folder/'FAILED.json')
                        if not failure.get('scientific_failure'):
                            raise ValueError('Reference has an engineering failure')
                        failed+=50
                    references[target][arm]=str(folder)
                    # Preserve source structures, terminal tensors, failure state and diagnostics.
                    for path in folder.iterdir():
                        if path.is_file():protected[str(path)]=sha(path)
    order=[base.arm_name(c,e,'late_only') for c in base.CHEMISTRIES for e in ENDPOINTS]
    spec=dict(policy=POLICY,code_commit=commit,scientific_commit=head,
        scientific_directory=str(frozen_repo),config_sha256=value_sha(config),
        controls=prior_spec['controls'],references=references,arms=arms(),new_arms=order,
        new_attempt_cap=400,reused_attempts=reused,referenced_failed_attempts=failed,
        population=50,seeds=base.SEEDS,logical_slots=list(range(10)),protected_files=protected,
        original_UG_remains_paused=True,scientific_loss=prior_spec['scientific_loss'],
        scale_protocol=prior_spec['scale_protocol'],coordinate_inpainting=False,
        chemical_state_projection_steps=[10,99],new_guidance_steps=[10,99],
        new_guidance_update_steps_affecting_returned_raw_prediction=[91,98],
        step99_update_excluded_from_returned_raw_prediction=True,
        exact_reference_prefix_steps=list(range(91)),contrasts=contrasts(),
        statistics=dict(prior_spec['statistics'],
            primary_comparisons='late_only minus window, separately per teacher, chemistry and raw output; raw999 minus raw1 at late_only',
            secondary_comparison='full_interior minus late_only; prior numerical failures remain unavailable',
            outliers='retain all 50; no post-hoc deletion or weaker retry'),
        report_note='本专项仅新增两教师×两种化学条件×raw1/raw999的10–99步引导，共400次。'
            '10–90与1–99均复用，历史数值失败仅作为不可评价对照。raw返回第99步更新前预测，'
            '故新增的第91–98步更新影响返回值，第99步更新只影响未返回的最终状态。不得删离群点或补样。')
    root.mkdir(parents=True,exist_ok=True)
    base.verify_sources(spec)
    frozen(root/'STUDY.json',spec);frozen(root/'config.json',config);frozen(root/'teachers.json',teachers)
    for name in ('initials','random_stream_reference'):
        link=root/name
        if not link.exists():link.symlink_to(source/name,target_is_directory=True)
        if not link.is_symlink() or link.resolve()!=(source/name).resolve():
            raise ValueError('Wrong original initial or random-stream source')
    return config,teachers,spec


def run(root,source,prior,frozen_repo,execution_repo):
    import fcntl
    from .prepare import load_model
    from .quality_queue import enqueue
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.executor.lock').open('a+') as own,(source/'.executor.lock').open('r+') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'FAILED.json').exists() or (root/'STARTED.json').exists():
            raise ValueError('Previous execution exists; do not automatically restart')
        config,teachers,spec=prepare(root,source,prior,frozen_repo,execution_repo)
        frozen(root/'STARTED.json',dict(pid=os.getpid(),time_shanghai=base.now(),code_commit=spec['code_commit']))
        for teacher in teachers:
            target=teacher['target'];model=load_model(config,teacher)
            features,metadata=base.teacher_features(model,teacher)
            frozen(root/'features'/(target+'.json'),metadata)
            original_generate=model._generate_selective;original_times=model._get_times
            native_source=(root/'provenance/native/instrumented_selective.py').read_text()
            schedule=spec['controls'][target]['aggregate']['schedule']
            for arm in spec['new_arms']:
                params=spec['arms'][arm];folder=root/'groups'/target/arm;folder.mkdir(parents=True,exist_ok=True)
                armconfig=dict(config,active_arm=dict(name=arm,**params));frozen(folder/'config.json',armconfig)
                frozen(folder/'candidates_batch.json',dict(teacher_id=target,logical_capacity=50,
                    physical_network_rows=10,historical_execution_shape_preserved=True,concurrent_network_units=1,
                    candidates=[dict(logical_seed=s,logical_slot=i,actual_schedule=schedule,**params)
                                for s in base.SEEDS for i in range(10)]))
                clock=base.ClockAudit();model.integrator._guard_warned=False
                def observed_times(this,times,lig_mask,inpaint_mask):
                    packed=original_times(times,lig_mask,inpaint_mask);clock.observe(times,packed);return packed
                model._get_times=types.MethodType(observed_times,model)
                compiled,rendered=base.compile_policy(original_generate,native_source,params['endpoint'])
                (folder/'effective_sampler.py').write_text(rendered,encoding='utf-8')
                native=types.MethodType(compiled,model)
                def generate(*args,**kwargs):
                    return base.finish(native(*args,**kwargs),model,teacher,features,folder,params,clock)
                model._generate_selective=generate
                reference=Path(spec['references'][target][base.arm_name(params['chemistry'],params['endpoint'],'window')])/'trace.h5'
                original_control=base.sampler.Control
                base.sampler.Control=functools.partial(base.TimeWindowControl,chemical_features=features,joint=False,
                    clock=clock,chemistry=params['chemistry'],coverage='late_only',
                    reference=reference,reference_steps=list(range(91)))
                executor=base.execute_variant(base.sampler,lambda ignored:base.StudyBudget(root),
                                               params['endpoint']+' output; see OUTPUT_POLICY.json')
                try:
                    executor(model,armconfig,teacher,schedule,folder,'formal',50,resume_path=None)
                    acceptance=read(folder/'PAIRED_TRACE_ACCEPTANCE.json')
                    if acceptance['all_equal'] is not True or acceptance['compared_steps']!=list(range(91)):
                        raise ValueError('The full matched prefix was not verified')
                    frozen(folder/'RAW_OUTPUT_CAUSALITY.json',dict(
                        guidance_steps=list(range(10,100)),returned_prediction_step=99,
                        last_guidance_update_affecting_returned_prediction=98,
                        newly_effective_steps=list(range(91,99)),
                        no_terminal_chemical_or_coordinate_projection=True))
                except FloatingPointError as error:
                    write(folder/'FAILED.json',dict(time_shanghai=base.now(),error_type=type(error).__name__,error=str(error),
                        charged_attempts=50,no_weaker_retry=True,scientific_failure=True))
                finally:
                    base.sampler.Control=original_control;model._generate_selective=original_generate;model._get_times=original_times
                    write(folder/'NETWORK_TIMES.json',clock.events)
                if (folder/'COMPLETE.json').exists():enqueue(armconfig,folder,teacher)
                write(root/'progress.json',dict(time_shanghai=base.now(),last_teacher=target,last_arm=arm,
                    completed_groups=len(list((root/'groups').glob('*/*/COMPLETE.json'))),
                    failed_groups=len(list((root/'groups').glob('*/*/FAILED.json')))))
                base.report(root)
            del model;torch.cuda.empty_cache();base.verify_sources(spec)
        ledger=read(root/'attempts.json')
        if len(ledger)!=8 or sum(v['count'] for v in ledger.values())!=400:
            raise ValueError('Sampling ledger differs from eight planned groups')
        frozen(root/'SAMPLING_FINISHED.json',dict(time_shanghai=base.now(),new_attempts=400,
            reused_attempts=spec['reused_attempts'],referenced_failed_attempts=spec['referenced_failed_attempts'],
            completed_groups=len(list((root/'groups').glob('*/*/COMPLETE.json'))),
            failed_groups=len(list((root/'groups').glob('*/*/FAILED.json')))))
    base.finalize(root)
