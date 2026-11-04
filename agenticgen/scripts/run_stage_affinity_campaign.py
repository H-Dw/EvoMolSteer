"""Serial six-job campaign, bounded scientific attempts, independent CPU diagnostics."""
import argparse
import fcntl
import gzip
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from agenticgen.uniguide.io import read,write,frozen,sha

TESTS=['test_campaign216','test_stage_path_search','test_early_path_search','test_diverse_box_search',
    'test_seven_box_search','test_uniguide_geometry_batch','test_geometry_chemical_state','test_adaptive_calibration',
    'test_linear_guidance','test_coordinate_statistics','test_guidance_analysis_tools','test_strict_final_submission',
    'test_absolute_coordinate_protocol','test_dynamic_matching_jump','test_open_domain_search','test_zero_start_guidance',
    'test_synchronized_chemistry','test_deepseek_tools']


def snapshot(suite,force=False):
    import shutil
    plan=read(suite/'PLAN.json');dest=Path(plan['durable_snapshot_root']);dest.mkdir(parents=True,exist_ok=True)
    state_path=dest/'MIRROR_STATE.json';state=read(state_path) if state_path.exists() else {}
    try:
        for p in suite.rglob('*'):
            if not p.is_file() or p.is_symlink() or p.name=='launcher.log':continue
            if p.suffix not in ('.json','.jsonl','.sdf','.csv') and p.name!='promoted_teacher.pt.gz':continue
            rel=p.relative_to(suite).as_posix();stat=p.stat();signature=[stat.st_size,stat.st_mtime_ns]
            if state.get(rel,{}).get('signature')==signature:continue
            if shutil.disk_usage(dest).free<stat.st_size+(64<<20):raise OSError('Durable mirror space reserve reached')
            digest=sha(p);target=dest/(rel+'.gz' if not p.name.endswith('.gz') else rel)
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.copying')
            with p.open('rb') as a,temporary.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as b:shutil.copyfileobj(a,b)
            if sha(p)!=digest:temporary.unlink();continue
            temporary.replace(target);state[rel]=dict(signature=signature,source_sha256=digest,archive_sha256=sha(target))
        write(state_path,state);write(suite/'DURABLE_SNAPSHOT.json',dict(status='mirrored',files=len(state),full_traces_included=False))
    except Exception as error:write(suite/'DURABLE_SNAPSHOT.json',dict(status='error',error=str(error),sampling_not_restarted=True))


def runner(repo):
    path=repo/'agenticgen/docs/experiments/20261031_guidance_comparison5/run_suite.py'
    spec=importlib.util.spec_from_file_location('campaign_serial',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.durable_snapshot=lambda suite,force=False:snapshot(suite,force)
    def validate(config,job):
        if config['rounds']!=10 or config['agent_tools']['skill_release']!='2.16.0' or len(config['conditions'])!=1:
            raise ValueError('Campaign stage search requires the explicit 10-round protocol')
        if config['conditions'][0]['teacher_id']!=job['teacher']:raise ValueError('Teacher mismatch')
        return 10
    module.validate_job_config=validate
    original=module.quality_sources
    def sources(config):
        result=set(original(config));root=Path(config['root'])
        for p in (root/'calibration/path').glob('*/*/COMPLETE.json'):result.add(read(p.parent/'aggregate.json')['id'])
        return sorted(result)
    module.quality_sources=sources
    return module


def audit_search(config):
    from agenticgen.uniguide import condition_evaluation as f
    root=Path(config['root']);final=read(root/'versions/GD01/FINALIZED.json')
    if final['completed_rounds']!=10 or not final['complete']:raise ValueError('Search did not complete ten valid rounds')
    populations=[]
    for p in (root/'measurement_registry').glob('*.json'):
        folder=root/read(p)['source_group']
        if not (folder/'COMPLETE.json').exists():continue
        row=read(folder/'aggregate.json');records=read(folder/'records.json');mean=f.score(row)
        if len(records)!=50 or abs(sum(r[f.METRIC] for r in records)/50-mean)>1e-12:raise ValueError('Aligned mean differs from raw records')
        populations.append(dict(id=row['id'],mean=mean))
    if list(root.rglob('synthesis.receipt.json')) or list(root.rglob('box_selection.receipt.json')):raise ValueError('Disabled Agent role invoked')
    frozen(root/'ONLINE_PROTOCOL_AUDIT.json',dict(release='2.16.0',rounds=10,chemical_inpainting=config['chemical_inpainting'],
        complete_populations=populations,final_sha256=sha(root/'versions/GD01/FINALIZED.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan');parser.add_argument('--iterative-config');args=parser.parse_args()
    if args.iterative_config:
        from agenticgen.uniguide.iterative_affinity import run
        config=read(args.iterative_config);root=Path(config['root']);root.mkdir(parents=True,exist_ok=True)
        with (root/'.executor.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=run(config)
        print({'complete':result['complete'],'improvement':result['improvement']});return
    plan=read(args.plan);suite=Path(plan['root']);repo=Path(plan['repo'])
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=plan['commit']:raise ValueError('Checkout changed')
    with open(plan['global_executor_lock'],'a+') as lock:
        write(suite/'STATUS.json',dict(stage='waiting_global_lock',pid=os.getpid()))
        fcntl.flock(lock,fcntl.LOCK_EX)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        write(suite/'STATUS.json',dict(stage='cpu_preflight',pid=os.getpid()))
        with (suite/'cpu_preflight.log').open('xb') as stream:
            done=subprocess.run([sys.executable,'-m','pytest',*['tests/unit/'+n+'.py' for n in TESTS],'-q',
                '--junitxml='+str(suite/'cpu_preflight.xml')],cwd=repo/'agenticgen',env=env,stdout=stream,stderr=subprocess.STDOUT)
        frozen(suite/'CPU_PREFLIGHT_EXIT.json',dict(returncode=done.returncode,log_sha256=sha(suite/'cpu_preflight.log')))
        if done.returncode:
            write(suite/'SUITE_FAILED.json',dict(stage='cpu_preflight',returncode=done.returncode));return
        common=runner(repo);results=[]
        for job in plan['jobs']:
            config=read(job['config']);root=Path(config['root'])
            try:
                if job['arm']<3:
                    result=common.run_job(repo,suite,job);audit_search(config)
                else:
                    dependency=next(r for r in results if r['job_id']==job['depends_on'])
                    if dependency['status']!='completed':
                        results.append(dict(job_id=job['id'],status='dependency_blocked'));continue
                    if root.exists():raise ValueError('Iterative job requires a fresh result root')
                    log=suite/'logs'/job['id']/'iterative.log';log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('xb') as stream:
                        process=subprocess.Popen([sys.executable,'-u',__file__,'--iterative-config',job['config']],cwd=repo,
                            stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                        while process.poll() is None:
                            write(suite/'STATUS.json',dict(stage='iterative_affinity',job_id=job['id'],child_pid=process.pid))
                            write(suite/'HEARTBEAT.json',dict(timestamp=time.time(),job_id=job['id'],child_pid=process.pid))
                            snapshot(suite);time.sleep(20)
                    frozen(log.parent/'ITERATIVE_EXIT.json',dict(returncode=process.returncode,log_sha256=sha(log)))
                    if process.returncode:raise RuntimeError('Iterative process failed; inspect preserved log')
                    result=read(root/'FINALIZED.json')
                    if not result['complete']:raise RuntimeError('Iterative round failures retained')
                results.append(dict(job_id=job['id'],status='completed',result=result))
            except Exception as error:
                results.append(dict(job_id=job['id'],status='failed_skipped',error=str(error),automatic_retry=False))
            finally:
                write(suite/'job_results.json',results);snapshot(suite)
        from agenticgen.uniguide.quality_queue import snapshot as quality_snapshot
        while True:
            reports={}
            for job in plan['jobs']:
                config=read(job['config']);root=Path(config['root'])
                if job['arm']==3 or not (root/'SUBJECTS.json').exists():continue
                common.submit_quality(config);reports[job['id']]=quality_snapshot(root,common.quality_sources(config))
            write(suite/'QUALITY_STATUS.json',reports)
            if not any(r['pending'] for r in reports.values()):break
            write(suite/'STATUS.json',dict(stage='quality_diagnostics'));snapshot(suite);time.sleep(20)
        common.check_preservation(suite)
        write(suite/'SUITE_FINISHED.json',dict(results=results,all_success=all(r['status']=='completed' for r in results) and not any(r['failed'] for r in reports.values()),
            scientific_candidate_cap=12500,sampling_and_diagnostics_terminal=True))
        write(suite/'STATUS.json',dict(stage='finished'));snapshot(suite)


if __name__=='__main__':main()
