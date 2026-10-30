"""Bounded one-shot orchestration; no retry, resume, skill update or next version."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agenticgen.uniguide.io import frozen, read, sha, write
from agenticgen.uniguide.quality_queue import enqueue, snapshot


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def campaign_stages(config):
    stages = [('acceptance', ['acceptance'])]
    if config.get('sampling_recovery'):
        stages.append(('recovery', ['recover-saved']))
    return stages + [('calibrate', ['calibrate']), ('formal', ['run-version', 'GD01'])]


def main():
    repo, config_path = map(Path, sys.argv[1:3])
    config = read(config_path)
    root = Path(config['root'])
    lock = (root/'.campaign.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stages = campaign_stages(config)
    frozen(root/'CAMPAIGN_CLAIM.json', dict(pid=os.getpid(), timestamp=now(), script_sha256=sha(__file__),
        config_sha256=sha(config_path), execution_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        stages=[' '.join(args) for _, args in stages], rounds_per_teacher=5,
        maximum_new_calibration_attempts=700, maximum_new_formal_attempts=1500, automatic_retry=False))
    preserved = read(root/'SOURCE_PRESERVATION.json')
    teachers = {s['teacher']['target']: s['teacher'] for s in read(root/'SUBJECTS.json')}
    queued = set()
    quality_errors = {}

    def check_sources():
        changed = [p for p,digest in preserved.items() if sha(p) != digest]
        if changed:
            raise ValueError('Protected source changed: '+str(changed))

    def submit_quality():
        for target,teacher in teachers.items():
            for complete in sorted((root/'versions/GD01'/target).glob('iteration_*/*/COMPLETE.json')):
                folder = complete.parent
                if str(folder) in queued:
                    continue
                try:
                    enqueue(config, folder, teacher)
                except Exception as error:
                    quality_errors[str(folder)] = dict(error=str(error), timestamp=now(), does_not_imply_molecular_failure=True)
                    write(root/'QUALITY_ENQUEUE_ERRORS.json', quality_errors)
                queued.add(str(folder))

    try:
        for stage,args in stages:
            check_sources()
            log = root/(stage+'.log')
            with log.open('xb') as stream:
                process = subprocess.Popen([sys.executable, '-u', str(repo/'agenticgen/scripts/run_uniguide_geometry.py'),
                    '--config', str(config_path), *args], stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                write(root/'STATUS.json', dict(timestamp=now(), stage=stage, status='running', pid=process.pid, log=str(log)))
                while process.poll() is None:
                    if stage == 'formal':
                        submit_quality()
                    write(root/'HEARTBEAT.json', dict(timestamp=now(), stage=stage, process_alive=process.poll() is None,
                        child_pid=process.pid, queued_quality_groups=len(queued)))
                    time.sleep(20)
                code = process.returncode
            if stage == 'formal':
                submit_quality()
            if code:
                raise RuntimeError(f'{stage} exited {code}; see {log}; no automatic retry')
            check_sources()
            frozen(root/(stage.upper()+'_FINISHED.json'), dict(timestamp=now(), exit_code=code, log=str(log), log_sha256=sha(log)))
        frozen(root/'SAMPLING_FINISHED.json', dict(timestamp=now(), version='GD01', rounds_per_teacher=5,
            awaiting_actual_external_review=True, no_next_version=True))
        while True:
            report = snapshot(root, [str(Path(p).relative_to(root)) for p in sorted(queued) if p not in quality_errors])
            write(root/'QUALITY_STATUS.json', report)
            write(root/'STATUS.json', dict(timestamp=now(), stage='quality', status='pending' if report['pending'] else 'finished',
                complete=report['complete'], failed=report['failed'], enqueue_errors=len(quality_errors), pending=report['pending']))
            if not report['pending']:
                break
            time.sleep(20)
        check_sources()
        finalized=read(root/'versions/GD01/FINALIZED.json')
        frozen(root/'CAMPAIGN_FINISHED.json', dict(timestamp=now(), sampling_complete=finalized.get('complete',True),
            quality=report, quality_enqueue_errors=quality_errors, requires_actual_external_review=True,
            original_source_files_unchanged=True,missing_teachers=finalized.get('missing_teachers',[]),
            skipped_branches=finalized.get('skipped_branches',[])))
        # Stop only this campaign's idle CPU worker, after confirming its runtime path and process identity.
        worker_path = root/'quality_queue/worker.json'
        if worker_path.exists():
            worker = read(worker_path)
            status = read(root/'quality_queue/status.json')
            proc = Path('/proc')/str(worker['pid'])/'cmdline'
            if proc.exists():
                command = proc.read_bytes().split(b'\0')
                if (b'agenticgen.uniguide.quality_queue' in command and
                    str(root/'quality_queue/runtime.json').encode() in command and
                    not status['active'] and status['pending'] == 0):
                    os.kill(worker['pid'], signal.SIGTERM)
                    frozen(root/'QUALITY_WORKER_STOPPED.json', dict(timestamp=now(), pid=worker['pid'], idle=True))
    except Exception as error:
        write(root/'CAMPAIGN_FAILED.json', dict(timestamp=now(), error=str(error), traceback=traceback.format_exc(),
            no_automatic_retry=True, source_artifacts_preserved=True))
        write(root/'STATUS.json', dict(timestamp=now(), stage='stopped', status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
