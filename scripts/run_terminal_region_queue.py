"""Sequential remote generator queue; one failed candidate cannot stop others."""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from evomolsteer.io import read_json,write_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','flowr-root','work','plan']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();plan=read_json(a.plan);a.work.mkdir(parents=True,exist_ok=True)
    status={'status':'running','started_unix':time.time(),'jobs':[]};state=a.work/'queue_status.json'
    write_json(state,status)
    for job in plan['jobs']:
        status['current']=job['name'];write_json(state,status)
        code=subprocess.call([sys.executable,'-u',str(a.repo/'scripts/run_terminal_region_cohort.py'),
            '--repo',str(a.repo),'--flowr-root',str(a.flowr_root),'--work',str(a.work),'--spec',str(a.repo/job['spec'])])
        status['jobs'].append({'name':job['name'],'returncode':code,'completed_unix':time.time()})
        write_json(state,status)
        if code and job.get('required_bridge'):
            status.update(status='blocked_runtime_control_mismatch',current=None);write_json(state,status)
            sys.exit(code)
    status.update(status='complete_with_failed_candidates' if any(v['returncode'] for v in status['jobs']) else 'complete',
        completed_unix=time.time(),current=None);write_json(state,status)
