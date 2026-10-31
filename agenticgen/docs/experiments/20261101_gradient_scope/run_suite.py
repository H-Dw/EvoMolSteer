"""Run six fixed measurements sequentially, with no provider or Agent entrypoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from agenticgen.uniguide.io import read, write, sha


def stamp(): return datetime.now(timezone(timedelta(hours=8))).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    plan=read(parser.parse_args().plan);root=Path(plan['root']);repo=Path(plan['repo'])
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==plan['commit']
    assert len(plan['jobs'])==6
    with (root/'.suite.lock').open('a+') as suite_lock, Path(plan['global_executor_lock']).open('a+') as global_lock:
        fcntl.flock(suite_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'STATUS.json',dict(timestamp=stamp(),stage='waiting_global_lock'))
        fcntl.flock(global_lock,fcntl.LOCK_EX)
        results=[]
        for job in plan['jobs']:
            try:
                assert sha(job['config'])==job['config_sha256']
                config=read(job['config']);jobroot=Path(config['root'])
                assert config['rounds']==0 and config['experiment_mode']=='fixed_gradient_atom_scope'
                assert not jobroot.exists(), 'Preserve existing task; no automatic restart'
                for phase in ['prepare','acceptance','run-fixed-scope']:
                    command=[sys.executable,'-u',str(repo/'agenticgen/scripts/run_uniguide_geometry.py'),'--config',job['config'],phase]
                    log=root/'logs'/job['id']/(phase+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('xb') as stream:
                        child=subprocess.Popen(command,cwd=repo,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        state=dict(timestamp=stamp(),job_id=job['id'],stage=phase,child_pid=child.pid,command=command,log=str(log))
                        write(root/'STATUS.json',state)
                        while child.poll() is None:
                            write(root/'HEARTBEAT.json',dict(state,timestamp=stamp(),child_alive=True))
                            time.sleep(10)
                    write(log.with_name(phase+'_EXIT.json'),dict(timestamp=stamp(),returncode=child.returncode))
                    if child.returncode:
                        raise RuntimeError(f'{phase} exited {child.returncode}; preserve failure and skip')
                result=dict(id=job['id'],status='sampling_finished',attempts=50,llm_calls=0,iterations=0)
            except Exception as error:
                result=dict(id=job['id'],status='failed_skipped',error=repr(error),automatic_retry=False)
            results.append(result);write(root/'job_results'/(job['id']+'.json'),dict(timestamp=stamp(),**result))
        write(root/'ALL_JOBS_ATTEMPTED.json',dict(timestamp=stamp(),results=results))
    # GPU locks are released while independent CPU diagnostics finish.
    deadline=time.monotonic()+3600
    while True:
        quality=[]
        for job in plan['jobs']:
            folder=Path(read(job['config'])['root'])/'fixed_population'
            if not (folder/'COMPLETE.json').exists():continue
            q=folder/'QUALITY_COMPLETE.json';f=folder/'QUALITY_FAILED.json'
            quality.append(dict(id=job['id'],complete=q.exists(),failed=read(f) if f.exists() else None))
        pending=[q for q in quality if not q['complete'] and not q['failed']]
        write(root/'STATUS.json',dict(timestamp=stamp(),stage='quality_wait' if pending else 'finished',quality=quality))
        if not pending or time.monotonic()>deadline:break
        time.sleep(10)
    write(root/'SUITE_FINISHED.json',dict(timestamp=stamp(),jobs=results,quality=quality,
          all_quality_terminal=not pending,llm_calls=0,iterations=0,actual_main_review_required=True))


if __name__=='__main__':main()
