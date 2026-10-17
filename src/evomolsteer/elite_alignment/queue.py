"""Detached server-local sequential queue with failure isolation and resume."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from evomolsteer.io import read_json, digest
from evomolsteer.storage.transactions import atomic_json, exclusive_lock
from .common import validate_protocol, validate_condition, read_result, failure, result_path


def stop_child(child):
    if child.poll() is not None:
        return
    if os.name == 'nt':
        child.terminate()
    else:
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == 'nt': child.kill()
        else: os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def pending(work, config, condition):
    return [a for a in config['attempts'] if read_result(work, config, condition, a) is None]


def run_condition(repo, config_path, config, condition, work, command=None, pilot=False):
    """Returns normally after child failure; no child exception escapes the queue."""
    state = work / 'worker_state.json'
    scope = dict(config, attempts=[a for a in config['attempts'] if not pilot or a['rank'] in (1,50,100,200)])
    for restart in range(config['max_worker_restarts'] + 1):
        if not pending(work, scope, condition): return
        if state.exists(): state.unlink()
        args = command or [sys.executable, '-u', '-m', 'evomolsteer.elite_alignment.worker', '--repo', str(repo),
            '--config', str(config_path), '--work', str(work), '--condition', condition['id']]
        if pilot and command is None: args.append('--pilot')
        log_path = work / 'logs' / f'{condition["id"]}.worker.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        active = None; attempt_started = time.monotonic(); started = time.monotonic(); timed_out = False
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nWORKER_START restart={restart} unix={time.time()}\n'); log.flush()
            child = subprocess.Popen(args, cwd=repo, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     start_new_session=os.name != 'nt')
            try:
                while child.poll() is None:
                    try:
                        info = read_json(state) if state.exists() else {}
                    except (OSError, json.JSONDecodeError):
                        info = {}  # A missing observation must not kill the job.
                    if info.get('condition') == condition['id']:
                        key = info.get('attempt')
                        if key != active:
                            active = key; attempt_started = time.monotonic()
                    limit = config['sample_timeout_seconds'] if active else config['startup_timeout_seconds']
                    elapsed = time.monotonic() - (attempt_started if active else started)
                    if elapsed > limit:
                        timed_out = True; stop_child(child); break
                    time.sleep(config['poll_seconds'])
                code = child.wait()
            except BaseException:
                stop_child(child); raise
        error = f'Worker {"timed out" if timed_out else "exited"}: code={code}; restart={restart}; log={log_path.name}'
        remaining = pending(work, scope, condition)
        if not remaining: return
        current = next((a for a in remaining if a['id'] == active), None)
        if current is not None:
            failure(work, config, condition, current, error, phase='worker_timeout' if timed_out else 'worker_crash')
        if restart == config['max_worker_restarts'] or active is None:
            for attempt in pending(work, scope, condition):
                failure(work, config, condition, attempt, error, status='skipped_condition_error', phase='condition_worker')
            return


def summarize(work, config):
    conditions = []
    for condition in config['conditions']:
        records = [read_result(work, config, condition, a) for a in config['attempts']]
        counts = {status: sum(r is not None and r['status'] == status for r in records)
                  for status in ('complete', 'failed', 'skipped_condition_error')}
        conditions.append({'id': condition['id'], **counts, 'pending': sum(r is None for r in records)})
    return conditions


def run(repo, config_path, work, retry_failed=False):
    repo, work = Path(repo).resolve(), Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path); validate_protocol(config)
    status_path = work / 'queue_status.json'
    with exclusive_lock(work / '.queue.lock'):
        if status_path.exists() and read_json(status_path).get('protocol_sha256') != config['protocol_sha256']:
            raise ValueError('Work directory belongs to a different protocol')
        if retry_failed:
            for condition in config['conditions']:
                for attempt in config['attempts']:
                    record = read_result(work, config, condition, attempt)
                    if record and record['status'] != 'complete':
                        path = result_path(work, condition, attempt)
                        history = path.parent / 'history' / str(time.time_ns()); history.mkdir(parents=True)
                        for f in list(path.parent.iterdir()):
                            if f.name != 'history': shutil.move(str(f), str(history / f.name))
        status = {'status': 'running', 'protocol_sha256': config['protocol_sha256'], 'pid': os.getpid(),
                  'started_unix': time.time(), 'current': None}
        atomic_json(status_path, status)
        try:
            if not Path(config['checkpoint']).is_file() or digest(config['checkpoint']) != config['checkpoint_sha256']:
                raise RuntimeError('Shared model checkpoint unavailable or changed')
            import torch
            if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
            # Validate both numerical controls before spending the remaining pool.
            for condition in config['conditions'][:2]:
                try:
                    validate_condition(repo, condition)
                    run_condition(repo, config_path, config, condition, work, pilot=True)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    for attempt in config['attempts']:
                        if attempt['rank'] in (1,50,100,200) and read_result(work, config, condition, attempt) is None:
                            failure(work, config, condition, attempt, error, phase='pilot')
            for condition in config['conditions']:
                if shutil.disk_usage(work).free < config['minimum_free_bytes']:
                    raise RuntimeError('Insufficient free storage; progress retained')
                status.update(current=condition['id'], updated_unix=time.time(), conditions=summarize(work, config))
                atomic_json(status_path, status)
                try:
                    validate_condition(repo, condition)
                except Exception as error:
                    for attempt in pending(work, config, condition):
                        failure(work, config, condition, attempt, error, status='skipped_condition_error', phase='frozen_contract')
                    continue
                try:
                    run_condition(repo, config_path, config, condition, work)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    # A failed spawn or condition adapter must not abort peers.
                    for attempt in pending(work, config, condition):
                        failure(work, config, condition, attempt, error,
                                status='skipped_condition_error', phase='condition_supervisor')
                from .report import report
                try:
                    report(work, config, bootstrap=False)
                except Exception as error:
                    atomic_json(work / 'analysis_error.json', {'condition': condition['id'], 'error': str(error), 'time': time.time()})
            status.update(conditions=summarize(work, config), current=None, finished_unix=time.time())
            status['status'] = 'complete_with_errors' if any(x['failed'] or x['skipped_condition_error'] for x in status['conditions']) else 'complete'
            from .report import report
            try:
                report(work, config, bootstrap=True)
            except Exception as error:
                atomic_json(work / 'analysis_error.json', {'phase': 'final_report', 'error': str(error), 'time': time.time()})
                status['analysis_error'] = str(error)
                status['status'] = 'complete_with_errors'
            atomic_json(status_path, status)
            return 0 if status['status'] == 'complete' else 2
        except (KeyboardInterrupt, SystemExit):
            status.update(status='interrupted', updated_unix=time.time(), conditions=summarize(work, config))
            atomic_json(status_path, status); return 130
        except Exception as error:
            status.update(status='blocked_environment', error=str(error), updated_unix=time.time(), conditions=summarize(work, config))
            atomic_json(status_path, status); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path); p.add_argument('--config', required=True, type=Path)
    p.add_argument('--work', type=Path); p.add_argument('--resume', action='store_true', default=True)
    p.add_argument('--retry-failed', action='store_true'); p.add_argument('--dry-run', action='store_true')
    p.add_argument('--status', action='store_true'); a = p.parse_args()
    config = read_json(a.config); validate_protocol(config); work = a.work or Path(config['remote_work'])
    if a.status:
        print(json.dumps({'queue': read_json(work / 'queue_status.json') if (work / 'queue_status.json').exists() else None,
                          'worker': read_json(work / 'worker_state.json') if (work / 'worker_state.json').exists() else None}, indent=2)); return
    if a.dry_run:
        for condition in config['conditions']: validate_condition(a.repo, condition)
        for spec in config['initials'].values():
            if digest(work / spec['file']) != spec['sha256']: raise ValueError('Initial-state checksum differs')
        print(json.dumps({'valid': True, 'conditions': len(config['conditions']), 'attempts_per_condition': 200,
                          'formal_inferences': 3400, 'cohorts': config['cohorts']}, indent=2)); return
    def interrupted(*args): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(run(a.repo, a.config.resolve(), work, a.retry_failed))


if __name__ == '__main__':
    main()
