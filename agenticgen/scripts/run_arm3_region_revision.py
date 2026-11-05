"""Run the explicitly revised, unstarted arm 3 after the frozen parent campaign."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from agenticgen.uniguide.io import read,write,frozen,sha
from run_stage_affinity_campaign import snapshot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    plan=read(args.plan);root=Path(plan['root']);suite=Path(plan['parent_suite']);repo=Path(plan['repo'])
    from agenticgen.uniguide.affinity_replay import require_scientific_allowance
    require_scientific_allowance(plan)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()!=plan['commit']:
        raise ValueError('Revision checkout changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(root/'STATUS.json',dict(stage='cpu_preflight'))
    with (root/'cpu_preflight.log').open('xb') as stream:
        check=subprocess.run([sys.executable,'-m','pytest','tests/unit/test_campaign216.py','tests/unit/test_synchronized_chemistry.py','tests/unit/test_affinity_endpoint.py','tests/unit/test_teacher_preparation.py',
            '-q','--junitxml='+str(root/'cpu_preflight.xml')],cwd=repo/'agenticgen',env=env,stdout=stream,stderr=subprocess.STDOUT)
    frozen(root/'CPU_PREFLIGHT_EXIT.json',dict(returncode=check.returncode,log_sha256=sha(root/'cpu_preflight.log')))
    if check.returncode:
        write(root/'REVISION_FAILED.json',dict(stage='cpu_preflight',returncode=check.returncode,scientific_attempts=0));return
    with open(plan['global_executor_lock'],'a+') as lock:
        write(root/'STATUS.json',dict(stage='waiting_parent_global_lock'))
        fcntl.flock(lock,fcntl.LOCK_EX)
        results=[]
        for job in plan['jobs']:
            try:
                if sha(job['config'])!=job['config_sha256']:raise ValueError('Revision configuration changed')
                config=read(job['config']);source=Path(config['fixed_scale_source'])/'versions/GD01/FINALIZED.json'
                if not source.exists() or not read(source)['complete']:
                    results.append(dict(job_id=job['id'],status='dependency_blocked'));continue
                if Path(config['root']).exists():raise ValueError('Revision requires a fresh scientific attempt')
                log=root/(job['id']+'.log')
                with log.open('xb') as stream:
                    child=subprocess.Popen([sys.executable,'-u',str(repo/'agenticgen/scripts/run_stage_affinity_campaign.py'),
                        '--iterative-config',job['config']],cwd=repo,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    while child.poll() is None:
                        state=dict(stage='iterative_affinity',job_id=job['id'],child_pid=child.pid,timestamp=time.time(),log=str(log))
                        write(root/'STATUS.json',state);write(root/'HEARTBEAT.json',state)
                        snapshot(suite);time.sleep(20)
                frozen(root/(job['id']+'_EXIT.json'),dict(returncode=child.returncode,log_sha256=sha(log)))
                if child.returncode:raise RuntimeError('Revised iterative job failed; no automatic retry')
                result=read(Path(config['root'])/'FINALIZED.json')
                results.append(dict(job_id=job['id'],status='completed' if result['complete'] else 'failed',result=result))
            except Exception as error:
                results.append(dict(job_id=job['id'],status='failed',error=str(error),automatic_retry=False))
            finally:
                write(root/'job_results.json',results);snapshot(suite)
        search_ids={j['id'] for j in read(suite/'PLAN.json')['jobs'] if j['arm']<3}
        searches=[r for r in read(suite/'job_results.json') if r['job_id'] in search_ids]
        frozen(root/'REVISION_FINISHED.json',dict(results=results,scientific_candidate_cap=plan['scientific_candidate_cap'],
            supersedes_only_unstarted_arm3=plan.get('supersedes_only_unstarted_arm3',True),all_success=all(r['status']=='completed' for r in results),
            parent_search_results=searches,effective_campaign_all_success=len(searches)==4 and len(results)==2 and
            all(r['status']=='completed' for r in searches+results)))
        write(root/'STATUS.json',dict(stage='finished'));snapshot(suite)


if __name__=='__main__':main()
