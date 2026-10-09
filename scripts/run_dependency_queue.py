"""Run explicitly frozen ablation cohorts serially, with a dependency receipt."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evomolsteer.io import read_json, write_json, digest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True); p.add_argument('--flowr-root', required=True)
    p.add_argument('--work', required=True); p.add_argument('--specs', required=True)
    p.add_argument('--predecessor'); p.add_argument('--name', required=True)
    a = p.parse_args(); repo, work = Path(a.repo), Path(a.work)
    output = work / (a.name + '.queue.json')
    if output.exists(): raise FileExistsError(output)
    specs = [repo / x for x in a.specs.split(',')]
    state = {'status': 'waiting', 'cohorts': [read_json(x)['name'] for x in specs],
        'spec_sha256': [digest(x) for x in specs], 'completed': [], 'started_unix': time.time()}
    write_json(output, state)
    try:
        if a.predecessor:
            deadline = time.time() + 21600
            while True:
                old = read_json(a.predecessor)
                if old['status'] == 'complete': break
                if old['status'] == 'failed' or time.time() > deadline:
                    raise ValueError('Predecessor failed or exceeded declared wait budget')
                time.sleep(30)
        for spec in specs:
            if shutil.disk_usage(work).free < 400 * 1024**2:
                raise ValueError('Insufficient private storage; no new cohort started')
            state.update(status='running', current=read_json(spec)['name']); write_json(output, state)
            subprocess.run([sys.executable, str(repo / 'scripts/run_dependency_cohort.py'), '--repo', str(repo),
                '--flowr-root', a.flowr_root, '--work', a.work, '--spec', str(spec)], check=True)
            state['completed'].append(read_json(spec)['name']); write_json(output, state)
        state.update(status='complete', completed_unix=time.time()); write_json(output, state)
    except Exception as error:
        state.update(status='failed', error=str(error)); write_json(output, state); raise
