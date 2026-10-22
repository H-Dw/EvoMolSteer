"""One-shot operational recovery: wait for the existing worker, then resume it.

Does not retry indefinitely or change scientific code. The actual recovery gate
is in recover_plip_council_analysis.py; its limit and preserved evidence apply.
"""
import argparse
import contextlib
import os
from pathlib import Path
import subprocess
import time

from agenticgen.adaptive_schedule.io import read, write, sha
from .recover_plip_council_analysis import recover, process_alive


@contextlib.contextmanager
def recovery_lock(root):
    import fcntl
    with (root / 'analysis_recovery.lock').open('a') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def resume(root, version, runtime, wait_seconds):
    root, runtime = Path(root).resolve(), Path(runtime).resolve()
    if not 0 <= wait_seconds <= 1800:
        raise ValueError('Wait must be bounded to at most 30 minutes')
    expected = read(root / 'versions' / version / 'deployment.json')['commit']
    if subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip() != expected:
        raise ValueError('Resume only the version original fixed deployment')
    # Verify the launcher entry points against their committed bytes as well.
    import hashlib
    for name in ('agenticgen/scripts/run.sh', 'agenticgen/src/agenticgen/cli.py', 'agenticgen/src/agenticgen/runtime/launch.py'):
        data = subprocess.check_output(['git', '-C', str(runtime), 'show', f'{expected}:{name}'])
        if hashlib.sha256(data).hexdigest() != sha(runtime / name):
            raise ValueError('Original launcher changed')
    with recovery_lock(root):
        op = root / 'operations' / f'analysis_resume_{version}_{time.time_ns()}'
        state = {'phase': 'waiting_for_existing_worker', 'version': version, 'runtime': str(runtime),
                 'runtime_commit': expected, 'pid': os.getpid(), 'time': time.time(), 'wait_limit_seconds': wait_seconds}
        write(op / 'journal.json', state)
        try:
            deadline = time.monotonic() + wait_seconds
            while process_alive(read(root / 'workers.json')['pid']):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Existing worker remains alive; no state changed')
                time.sleep(min(10, max(0, deadline - time.monotonic())))
            result = recover(root, version, apply=True)
            state.update(phase='recovered_pending_launch', recovery=result)
            write(op / 'journal.json', state)
            with (op / 'launch.log').open('wb') as log:
                subprocess.run(['bash', str(runtime / 'agenticgen/scripts/run.sh'), 'launch',
                                '--root', str(root), '--version', version], cwd=runtime,
                               env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=180)
            launch = read(root / 'launch.json')
            if launch['commit'] != expected or launch['version'] != version or not process_alive(launch['pid']):
                raise RuntimeError('Original runtime launch was not confirmed')
            state.update(phase='resumed', new_worker_pid=launch['pid'], completed=time.time())
        except Exception as exc:
            # The actual gate and API launcher already redact credential values.
            state.update(phase='failed', error_type=type(exc).__name__, completed=time.time())
            write(op / 'journal.json', state)
            raise
        write(op / 'journal.json', state)
        return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--version', required=True)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--wait-seconds', type=int, default=1800)
    a = p.parse_args()
    print(resume(a.root, a.version, a.runtime, a.wait_seconds))
