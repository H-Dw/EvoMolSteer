"""Paired historical-protocol endpoint x guidance-window experiment.

The CLI loads this module against the immutable 0e7d4dc2 scientific package.
No geometry-protocol multiplier, loss, molecular editing, or coordinate overlay.
"""
import ast
import copy
import datetime
import functools
import itertools
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import types

import h5py
import numpy as np
import torch

from . import sampler
from .chem_inpainting import teacher_features, project_chemistry
from .condition import loss
from .io import read, write, frozen, sha, value_sha
from .joint_endpoint import ClockAudit, EndpointControl, compile_endpoint, execute_variant
from .native_tools import save_pt, tensor_digest
from .protocol import SEEDS, expand

POLICY = 'historical-coordinate-time-window-factorial-v1'
FROZEN_COMMIT = '0e7d4dc2bb4109362ac797477ba148df119f8f31'
ENDPOINTS = ('native1', 'raw1', 'raw999')
CHEMISTRIES = ('gradient', 'chemical')
COVERAGES = ('window', 'full_interior')
TRACE_KEYS = ('current_A', 'guidance_delta_A', 'native_noise', 'guided_proposal', 'assignment')
SCIENCE_FILES = ('sampler', 'condition', 'prepare', 'inputs', 'native_tools', 'batching',
                 'chunking', 'random_streams', 'protocol', 'chem_inpainting', 'joint_endpoint',
                 'quality_queue', 'quality', 'profiling')
FrozenControl = sampler.Control


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def arm_name(chemical, endpoint, coverage):
    return f'{chemical}__{endpoint}__{coverage}'


def arms():
    return {arm_name(c, e, w): dict(chemistry=c, endpoint=e, coverage=w)
            for c, e, w in itertools.product(CHEMISTRIES, ENDPOINTS, COVERAGES)}


def expand_coverage(schedule, coverage):
    values = expand(schedule)
    if coverage == 'full_interior':
        values[1:10] = [schedule[0]] * 9
        values[91:100] = [schedule[3]] * 9
    elif coverage != 'window':
        raise ValueError('Unknown guidance coverage')
    assert values[0] == 0  # The original score-proxy VJP coefficient contains 1/t.
    return values


def compile_policy(original, source, endpoint):
    if endpoint not in ENDPOINTS:
        raise ValueError('Unknown output policy')
    function, rendered = compile_endpoint(original, source, endpoint != 'native1')
    if endpoint == 'raw1':
        parsed = ast.parse(rendered)
        grids = [n for n in ast.walk(parsed) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'torch.linspace']
        if len(grids) != 1 or ast.literal_eval(grids[0].args[1]) != .999:
            raise ValueError('Unexpected short-grid compiler output')
        grids[0].args[1] = ast.Constant(1)
        ast.fix_missing_locations(parsed)
        rendered = ast.unparse(parsed) + '\n'
        scope = dict(original.__func__.__globals__)
        exec(compile(rendered, 'time_window_native.py', 'exec'), scope)
        function = scope['_generate_selective']
    return function, rendered


class TimeWindowControl(EndpointControl):
    def __init__(self, *args, chemistry, coverage, reference=None, reference_steps=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.chemistry, self.coverage = chemistry, coverage
        self.schedule = expand_coverage(args[3], coverage)
        self.reference = Path(reference) if reference else None
        self.reference_steps = set(reference_steps or [])
        self.compared_steps = []
        frozen(self.folder/'GUIDANCE_TIMES.json', dict(
            coverage=coverage, actual_scale_by_step=self.schedule,
            active_steps=[i for i, s in enumerate(self.schedule) if s],
            endpoint_zero_skipped=True, terminal_forward_guidance=False))

    def after(self, current, step):
        result = (super().after(current, step) if self.chemistry == 'chemical'
                  else FrozenControl.after(self, current, step))
        if self.reference and step in self.reference_steps:
            with h5py.File(self.reference, 'r') as ref:
                comparisons = {key: bool(np.array_equal(self.trace[key][step], ref[key][step])) for key in TRACE_KEYS}
            if not all(comparisons.values()):
                write(self.folder/'PAIRING_FAILED.json', dict(step=step, comparisons=comparisons, reference=str(self.reference)))
                raise ValueError('Historical matched-trajectory comparison failed')
            self.compared_steps.append(step)
            write(self.folder/'PAIRED_TRACE_ACCEPTANCE.json', dict(
                all_equal=True, compared_steps=self.compared_steps, keys=TRACE_KEYS, reference=str(self.reference)))
        return result


def finish(output, model, teacher, features, folder, params, clock):
    save_pt(folder/'terminal_uncovered_before_projection.pt.gz', output)
    values, _, _, ids = loss(output['coords'], teacher['core_A'], output['mask'].bool())
    if params['endpoint'] != 'native1':
        with h5py.File(folder/'trace.h5', 'r') as trace:
            if not np.array_equal(output['coords'].detach().cpu().numpy(), trace['endpoint_A'][99]):
                raise ValueError('Raw output does not equal the final-loop uncovered coordinate prediction')
    projection = params['chemistry'] == 'chemical' and params['endpoint'] == 'native1'
    audit = None
    if projection:
        output, audit = project_chemistry(output, ids, features)
    write(folder/'OUTPUT_POLICY.json', dict(
        **params, grid_end=.999 if params['endpoint'] == 'raw999' else 1., intervals=100,
        intended_interval_width=.00999 if params['endpoint'] == 'raw999' else .01,
        output_source='native_terminal_corrector' if params['endpoint'] == 'native1' else 'last_loop_uncovered',
        extra_terminal_forward=params['endpoint'] == 'native1', terminal_chemical_projection=projection,
        coordinate_projection=False, raw_condition_atom_rmsd_angstrom=float((2*values).sqrt().mean()),
        last_loop=model._adaptive.last_loop, endpoint_rescore_time=.9999, chemical_projection_audit=audit))
    clock.phase = 'endpoint_rescore'
    return output


class StudyBudget:
    def __init__(self, root):
        self.root = Path(root)

    def reserve(self, run_id, stage, count, binding):
        spec = read(self.root/'STUDY.json')
        allowed = {f'groups/{t}/{a}' for t in spec['controls'] for a in spec['new_arms']}
        if run_id not in allowed or stage != 'formal' or count != 50:
            raise ValueError('Only the declared eighteen fifty-attempt groups are authorized')
        path = self.root/'attempts.json'
        ledger = read(path) if path.exists() else {}
        item = dict(stage=POLICY, count=count, binding=binding)
        if run_id in ledger:
            if ledger[run_id] != item:
                raise ValueError('Attempt identity changed')
            return
        if sum(v['count'] for v in ledger.values()) + count > 900:
            raise ValueError('900-attempt limit, including failed attempts')
        ledger[run_id] = item
        write(path, ledger)


def verify_sources(spec):
    changed = [p for p, digest in spec['protected_files'].items() if sha(p) != digest]
    if changed:
        raise ValueError('Protected source changed: ' + str(changed))


def prepare(root, source, chemical, joint, frozen_repo, execution_repo):
    if not all((p/'FINALIZED.json').exists() for p in (chemical, joint)):
        raise ValueError('Both historical inpainting experiments must be finalized')
    head = subprocess.check_output(['git', '-C', str(frozen_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != FROZEN_COMMIT or subprocess.check_output(['git', '-C', str(frozen_repo), 'status', '--porcelain']).strip():
        raise ValueError('Historical scientific checkout differs')
    if subprocess.check_output(['git', '-C', str(execution_repo), 'status', '--porcelain', '--untracked-files=no']).strip():
        raise ValueError('Experiment checkout must be clean')
    commit = subprocess.check_output(['git', '-C', str(execution_repo), 'rev-parse', 'HEAD'], text=True).strip()
    prior = read(chemical/'STUDY.json')
    joint_spec = read(joint/'STUDY.json')
    if prior['controls'] != joint_spec['controls']:
        raise ValueError('Historical control selection differs')
    protected = {str(source/p): d for p, d in prior['source_hashes'].items()}
    protected.update({str(chemical/p): d for p, d in joint_spec['prior_hashes'].items()})
    teachers = read(source/'teachers.json')
    references = {}
    for teacher in teachers:
        target = teacher['target']
        references[target] = {
            arm_name('gradient','native1','window'): str(source/prior['controls'][target]['group']),
            arm_name('chemical','native1','window'): str(chemical/'groups'/target),
            arm_name('chemical','raw999','window'): str(joint/'groups'/target/'chemical_raw999')}
        for initial in sorted((source/'initials'/target).glob('batch_*.pt.gz')):
            protected[str(initial)] = sha(initial)
        reference = source/'random_stream_reference'/(target+'.json')
        protected[str(reference)] = sha(reference)
        for path in references[target].values():
            folder = Path(path)
            receipt = read(folder/'COMPLETE.json')
            if receipt['records_sha256'] != sha(folder/'records.json'):
                raise ValueError('Historical records receipt differs')
            records = read(folder/'records.json')
            if {(r['logical_seed'],r['logical_slot']) for r in records} != {(s,i) for s in SEEDS for i in range(10)} or len(records)!=50:
                raise ValueError('Historical paired population differs')
            for name in ('COMPLETE.json','manifest.json','records.json','aggregate.json','trace.h5','steps.json',
                         'QUALITY_COMPLETE.json','quality.json','molecules_all_built.sdf'):
                protected[str(folder/name)] = sha(folder/name)
    package = frozen_repo/'agenticgen/src/agenticgen/uniguide'
    scientific = {str(package/(n+'.py')): sha(package/(n+'.py')) for n in SCIENCE_FILES}
    for path, digest in scientific.items():
        loaded = Path(sampler.__file__).parent/Path(path).name
        if loaded.resolve() != Path(path).resolve() or sha(loaded) != digest:
            raise ValueError('Wrong frozen scientific package')
    protected.update(scientific)
    for path in (chemical/'STUDY.json',chemical/'FINALIZED.json',joint/'STUDY.json',joint/'FINALIZED.json'):
        protected[str(path)] = sha(path)
    config = copy.deepcopy(read(chemical/'config.json'))
    config.pop('chemical_only_pilot',None)
    config.update(root=str(root), time_window_experiment=POLICY)
    all_arms = arms()
    reused = set(next(iter(references.values())))
    # Matched 0..1 bridge trajectories gate the new interventions against history.
    order = [arm_name(c,'raw1','window') for c in CHEMISTRIES]
    order += [arm_name('gradient','raw999','window')]
    order += [arm_name(c,e,'full_interior') for c in CHEMISTRIES for e in ENDPOINTS]
    assert set(order) == set(all_arms)-reused and len(order)==9
    spec = dict(policy=POLICY, code_commit=commit, scientific_commit=head, scientific_directory=str(frozen_repo),
        config_sha256=value_sha(config), controls=prior['controls'], references=references,
        arms=all_arms, new_arms=order, new_attempt_cap=900, reused_attempts=300, population=50,
        seeds=SEEDS, logical_slots=list(range(10)), protected_files=protected,
        original_UG_remains_paused=True, scientific_loss='historical absolute PLIP-core coordinate VJP',
        scale_protocol='frozen historical actual four-window scales; no new lambda or Agent calls',
        coordinate_inpainting=False, chemical_state_projection_steps=[10,99],
        full_guidance_steps=[1,99], endpoint_zero_excluded_due_to_original_formula=True,
        statistics=dict(unit='five paired seed clusters; ten slots per seed', confidence=.95,
            practical_rmsd_threshold_angstrom=.05, practical_relative_threshold=.10,
            exploratory=True, held_out=False, selection_seeds_reused=True,
            primary_comparisons='raw999-native1 at window coverage; full-window at native1, separately per chemistry and teacher',
            bridge='raw999-raw1 isolates grid; raw1-native1 changes terminal source and chemical terminal projection',
            multiple_comparisons='descriptive intervals, no family-wise significance claim'))
    root.mkdir(parents=True,exist_ok=True)
    verify_sources(spec)
    frozen(root/'STUDY.json',spec);frozen(root/'config.json',config);frozen(root/'teachers.json',teachers)
    for name in ('initials','random_stream_reference'):
        link=root/name
        if not link.exists():link.symlink_to(source/name,target_is_directory=True)
        if not link.is_symlink() or link.resolve()!=(source/name).resolve():raise ValueError('Wrong shared initial source')
    return config,teachers,spec


def paired_statistics(left,right,metric):
    from scipy.stats import t as student_t
    index=lambda rows:{(r['logical_seed'],r['logical_slot']):r for r in rows}
    a,b=index(left),index(right);expected={(s,i) for s in SEEDS for i in range(10)}
    if set(a)!=expected or set(b)!=expected or len(left)!=50 or len(right)!=50:
        return dict(status='unavailable_incomplete_population',denominator=50)
    if any(not isinstance(r.get(metric),(float,int,bool)) or not math.isfinite(r[metric]) for r in left+right):
        return dict(status='unavailable_nonfinite_or_unmeasured',denominator=50)
    deltas=[float(np.mean([b[s,i][metric]-a[s,i][metric] for i in range(10)])) for s in SEEDS]
    mean=float(np.mean(deltas));half=float(student_t.ppf(.975,4)*np.std(deltas,ddof=1)/np.sqrt(5))
    base=float(np.mean([r[metric] for r in left]));threshold=max(.05,.10*abs(base))
    return dict(status='complete',mean_delta=mean,seed_mean_deltas=deltas,
        exploratory_95pct_seed_cluster_interval=[mean-half,mean+half],clusters=5,paired_slots=50,
        practical_rmsd_threshold=threshold if metric=='condition_rmsd_A' else None,
        interval_excludes_zero=mean-half>0 or mean+half<0,
        practical_rmsd_difference=abs(mean)>=threshold if metric=='condition_rmsd_A' else None)


def diagnostic_summary(folder):
    rows=read(folder/'steps.json')
    maxima=[]
    for r in rows:
        for value in r['max_atom_increment_A']:
            if math.isfinite(value):maxima.append(value)
    intervals={}
    for name,lo,hi in [('early',1,9),('original',10,90),('late',91,99)]:
        chosen=[r for r in rows if lo<=r['step']<=hi]
        intervals[name]=dict(steps=len(chosen),active_steps=sum(bool(r['scale']) for r in chosen),
            maximum_single_atom_guidance_increment_angstrom=max((max(r['max_atom_increment_A']) for r in chosen),default=None))
    return dict(maximum_single_atom_guidance_increment_angstrom=max(maxima,default=None),
        intervals=intervals,large_finite_values_are_not_clipped=True)


def report(root):
    import csv
    spec=read(root/'STUDY.json');details={};table=[];comparisons=[]
    for target in spec['controls']:
        records={}
        for arm,params in spec['arms'].items():
            reused=arm in spec['references'][target]
            folder=Path(spec['references'][target][arm]) if reused else root/'groups'/target/arm
            row=dict(target=target,arm=arm,reused=reused,folder=str(folder),attempts=50)
            if not (folder/'COMPLETE.json').exists():
                row.update(status='failed' if (folder/'FAILED.json').exists() else 'pending')
                details[target+'/'+arm]=dict(**row,failure=read(folder/'FAILED.json') if (folder/'FAILED.json').exists() else None)
                table.append(row);continue
            rec=read(folder/'records.json');records[arm]=rec
            agg=read(folder/'aggregate.json');quality=None
            light=read(folder/'QUALITY_LIGHT_COMPLETE.json') if (folder/'QUALITY_LIGHT_COMPLETE.json').exists() else None
            if (folder/'QUALITY_COMPLETE.json').exists():
                quality=read(folder/'QUALITY_COMPLETE.json')
                if any(quality.get(k)!=v for k,v in dict(attempts=50,checked_attempts=10,not_selected=40,coverage=.2).items()):
                    raise ValueError('Quality coverage differs')
            row.update(status='complete',condition_atom_rmsd_angstrom=agg['C'],mean_pic50=agg['mean_pic50'],
                built=agg['built'],unique_smiles=agg['unique_smiles'],
                built_and_connected=light.get('valid_connected') if light else None,
                condition_atom_rmsd_median_angstrom=float(np.median([r['condition_rmsd_A'] for r in rec])),
                condition_atom_rmsd_maximum_angstrom=max(r['condition_rmsd_A'] for r in rec),
                quality_status='complete' if quality else 'engineering_failure' if (folder/'QUALITY_FAILED.json').exists() else 'pending',
                checked_attempts=10 if quality else None,posebusters_pass=quality.get('posebusters_pass') if quality else None)
            details[target+'/'+arm]=dict(**row,aggregate=agg,diagnostics=diagnostic_summary(folder),quality=quality,
                quality_light=light,
                quality_error=read(folder/'QUALITY_FAILED.json') if (folder/'QUALITY_FAILED.json').exists() else None,
                output_policy=read(folder/'OUTPUT_POLICY.json') if (folder/'OUTPUT_POLICY.json').exists() else dict(historical_native=True),
                network_times=read(folder/'NETWORK_TIMES.json') if (folder/'NETWORK_TIMES.json').exists() else None)
            table.append(row)
        contrast=[]
        for c in CHEMISTRIES:
            for w in COVERAGES:
                for a,b,label in [('native1','raw999','requested_output_package'),('raw1','raw999','grid_only_bridge'),('native1','raw1','terminal_output_bridge')]:
                    contrast.append((arm_name(c,a,w),arm_name(c,b,w),label))
            for e in ENDPOINTS:contrast.append((arm_name(c,e,'window'),arm_name(c,e,'full_interior'),'guidance_coverage'))
        for a,b,label in contrast:
            metrics={m:paired_statistics(records[a],records[b],m) for m in ('condition_rmsd_A','pic50_on_rescore','build_success')} if a in records and b in records else None
            comparisons.append(dict(target=target,baseline=a,treatment=b,contrast=label,paired_metrics=metrics,
                status='available' if metrics else 'unavailable_group_failure_or_pending',missing_not_imputed=True))
    verify_sources(spec)
    write(root/'RESULTS.json',dict(time_shanghai=now(),policy=POLICY,groups=details,rows=table,comparisons=comparisons,
        statistics=spec['statistics'],new_attempts=sum(v['count'] for v in read(root/'attempts.json').values()) if (root/'attempts.json').exists() else 0,
        source_files_unchanged=True,reused_attempts=300,coordinate_projection=False))
    keys=sorted({k for row in table for k in row})
    with (root/'summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(table)
    text=['# 网格、输出来源与引导时间范围：配对实验','',f'更新时间：{now()}','',
        '| 教师 | 组 | 状态 | 条件原子RMSD/Å | 平均pIC50 | 构建/50 | PB抽样通过/10 |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in table:text.append(f"| {r['target']} | {r['arm']} | {r['status']} | {r.get('condition_atom_rmsd_angstrom')} | {r.get('mean_pic50')} | {r.get('built')} | {r.get('posebusters_pass')} |")
    text+=['','## 配对条件原子RMSD差异（处理−对照）','','| 教师 | 对比 | 平均差/Å | 五种子探索性95%区间 |', '|---|---|---:|---|']
    for c in comparisons:
        m=(c['paired_metrics'] or {}).get('condition_rmsd_A',{})
        text.append(f"| {c['target']} | {c['treatment']} − {c['baseline']} | {m.get('mean_delta')} | {m.get('exploratory_95pct_seed_cluster_interval',c['status'])} |")
    text+=['','完整原始差异、实用差异阈值、失败、位移诊断和质量覆盖见RESULTS.json。五个种子同时用于历史配置选择，所有区间仅为探索性配对描述，不是独立留出验证或多重比较校正后的显著性结论。失败不补样，不用中位数替代含极端值的全量均值。',
        'raw999：100区间0–0.999，最后循环网络约0.98901；raw1：100区间0–1，最后循环网络约0.99；native1额外终态前向约0.9999。只有化学/native1在最终输出投影化学特征；全部组均不覆盖坐标。full_interior在1–99步引导，跳过奇异t=0。']
    (root/'report.zh-CN.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return details


def stop_own_idle_worker(root):
    receipt=root/'quality_queue/worker.json';status=root/'quality_queue/status.json'
    if not receipt.exists():return
    job=read(receipt);pid=job['pid'];proc=Path('/proc')/str(pid)
    if not proc.exists():return
    args=proc.joinpath('cmdline').read_bytes().split(b'\0')
    if b'agenticgen.uniguide.quality_queue' not in args or str(root/'quality_queue/runtime.json').encode() not in args:
        raise ValueError('Quality worker ownership differs')
    if not status.exists() or read(status)['active'] or read(status)['pending']:
        raise ValueError('Cannot stop a busy quality worker')
    if os.getpgid(pid)!=pid:raise ValueError('Quality process group is not private')
    os.killpg(pid,signal.SIGTERM)
    write(root/'quality_queue/STOP_REQUESTED.json',dict(pid=pid,time_shanghai=now(),all_requests_terminal=True))


def finalize(root,timeout=21600):
    start=time.monotonic()
    while True:
        pending=[p.parent for p in (root/'groups').glob('*/*/COMPLETE.json')
                 if not (p.parent/'QUALITY_COMPLETE.json').exists() and not (p.parent/'QUALITY_FAILED.json').exists()]
        if not pending:break
        if time.monotonic()-start>timeout:
            report(root);write(root/'POSTPROCESSING_PENDING.json',dict(time_shanghai=now(),groups=[str(p) for p in pending]));return
        time.sleep(10)
    details=report(root)
    # Receipts can precede the worker's idle status by one polling cycle.
    for _ in range(10):
        status=root/'quality_queue/status.json'
        if not status.exists() or (not read(status)['active'] and not read(status)['pending']):break
        time.sleep(2)
    stop_own_idle_worker(root)
    frozen(root/'FINALIZED.json',dict(time_shanghai=now(),results_sha256=sha(root/'RESULTS.json'),
        complete_groups=sum(r['status']=='complete' for r in details.values()),
        failed_groups=sum(r['status']=='failed' for r in details.values()),
        quality_engineering_failures=sum(r.get('quality_status')=='engineering_failure' for r in details.values()),
        original_sources_unchanged=True,independent_validation=False,new_attempts=sum(v['count'] for v in read(root/'attempts.json').values())))


def run(root,source,chemical,joint,frozen_repo,execution_repo):
    import fcntl
    from .prepare import load_model
    from .quality_queue import enqueue
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.executor.lock').open('a+') as own,(source/'.executor.lock').open('r+') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'FAILED.json').exists() or (root/'STARTED.json').exists():
            raise ValueError('Prior execution exists; no automatic restart')
        config,teachers,spec=prepare(root,source,chemical,joint,frozen_repo,execution_repo)
        frozen(root/'STARTED.json',dict(pid=os.getpid(),time_shanghai=now(),code_commit=spec['code_commit']))
        for teacher in teachers:
            target=teacher['target'];model=load_model(config,teacher)
            features,metadata=teacher_features(model,teacher);frozen(root/'features'/(target+'.json'),metadata)
            original_generate=model._generate_selective;original_times=model._get_times
            native_source=(root/'provenance/native/instrumented_selective.py').read_text()
            schedule=spec['controls'][target]['aggregate']['schedule']
            for arm in spec['new_arms']:
                params=spec['arms'][arm];folder=root/'groups'/target/arm;folder.mkdir(parents=True,exist_ok=True)
                armconfig=dict(config,active_arm=dict(name=arm,**params));frozen(folder/'config.json',armconfig)
                frozen(folder/'candidates_batch.json',dict(teacher_id=target,logical_capacity=50,physical_network_rows=10,
                    historical_execution_shape_preserved=True,concurrent_network_units=1,
                    candidates=[dict(logical_seed=s,logical_slot=i,actual_schedule=schedule,**params) for s in SEEDS for i in range(10)]))
                clock=ClockAudit();model.integrator._guard_warned=False
                def observed_times(this,times,lig_mask,inpaint_mask):
                    packed=original_times(times,lig_mask,inpaint_mask);clock.observe(times,packed);return packed
                model._get_times=types.MethodType(observed_times,model)
                compiled,rendered=compile_policy(original_generate,native_source,params['endpoint'])
                (folder/'effective_sampler.py').write_text(rendered,encoding='utf-8')
                native=types.MethodType(compiled,model)
                def generate(*args,**kwargs):
                    return finish(native(*args,**kwargs),model,teacher,features,folder,params,clock)
                model._generate_selective=generate
                reference=None;reference_steps=[]
                if params['coverage']=='window' and params['endpoint']=='raw1':
                    reference=Path(spec['references'][target][arm_name(params['chemistry'],'native1','window')])/'trace.h5'
                    reference_steps=list(range(10,100))
                elif params['coverage']=='window' and params['endpoint']=='raw999':
                    reference=Path(spec['references'][target][arm_name('chemical','raw999','window')])/'trace.h5'
                    reference_steps=list(range(11))
                original_control=sampler.Control
                sampler.Control=functools.partial(TimeWindowControl,chemical_features=features,joint=False,clock=clock,
                    chemistry=params['chemistry'],coverage=params['coverage'],reference=reference,reference_steps=reference_steps)
                executor=execute_variant(sampler,lambda ignored:StudyBudget(root),params['endpoint']+' output; see OUTPUT_POLICY.json')
                try:
                    executor(model,armconfig,teacher,schedule,folder,'formal',50,resume_path=None)
                except FloatingPointError as error:
                    write(folder/'FAILED.json',dict(time_shanghai=now(),error_type=type(error).__name__,error=str(error),
                        charged_attempts=50,no_weaker_retry=True,scientific_failure=True))
                finally:
                    sampler.Control=original_control;model._generate_selective=original_generate;model._get_times=original_times
                    write(folder/'NETWORK_TIMES.json',clock.events)
                if (folder/'COMPLETE.json').exists():enqueue(armconfig,folder,teacher)
                write(root/'progress.json',dict(time_shanghai=now(),last_teacher=target,last_arm=arm,
                    completed_groups=len(list((root/'groups').glob('*/*/COMPLETE.json'))),
                    failed_groups=len(list((root/'groups').glob('*/*/FAILED.json')))))
                report(root)
            del model;torch.cuda.empty_cache();verify_sources(spec)
        ledger=read(root/'attempts.json')
        if len(ledger)!=18 or sum(v['count'] for v in ledger.values())!=900:
            raise ValueError('Final sampling ledger does not match planned arms')
        frozen(root/'SAMPLING_FINISHED.json',dict(time_shanghai=now(),new_attempts=900,reused_attempts=300,
            completed_groups=len(list((root/'groups').glob('*/*/COMPLETE.json'))),failed_groups=len(list((root/'groups').glob('*/*/FAILED.json')))))
    # GPU campaign lock released; only this study's CPU diagnostics remain.
    finalize(root)
