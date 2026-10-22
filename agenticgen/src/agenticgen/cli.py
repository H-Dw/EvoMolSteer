"""Formal four-hypothesis workflow. Launch inherits API/proxy environment without serializing values."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    from agenticgen.adaptive_schedule.group_protocol import GROUP_ROOT
    from agenticgen.adaptive_schedule.io import read, write, repository, commit
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor', 'analyze', 'prepare', 'adopt-unstarted-runtime', 'validate', 'run', 'launch', 'status', 'report', 'review-pack', 'review-check'])
    p.add_argument('--root', type=Path, default=Path(GROUP_ROOT))
    p.add_argument('--source-root', type=Path)
    p.add_argument('--version', default='v1')
    p.add_argument('--reason')
    a = p.parse_args()
    from agenticgen.adaptive_schedule import group_run
    if a.command == 'doctor':
        from agenticgen.runtime.paths import host_config, skill
        import agenticgen
        result = {'package': agenticgen.__file__, 'host': host_config(),
                  'analyst_skill_exists': skill('plip-group-analyst').is_file()}
    elif a.command == 'analyze':
        from agenticgen.reporting import analyze
        result = analyze(a.root)
    elif a.command == 'prepare':
        if not a.source_root: p.error('--source-root is required')
        result = group_run.prepare(a.root, a.source_root)
    elif a.command == 'adopt-unstarted-runtime':
        result = group_run.adopt_unstarted_runtime(a.root, a.reason)
    elif a.command == 'validate':
        from agenticgen.adaptive_schedule.group_validation import validate
        result = validate(a.root)
    elif a.command == 'run':
        try:
            from agenticgen.adaptive_schedule.environment import record_environment
            record_environment(a.root, read(a.root/'config.json'))
            result = group_run.run(a.root, a.version)
        except Exception as exc:
            import traceback
            failure = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(), 'time': time.time()}
            write(a.root/'operations'/f'failure_{time.time_ns()}.json', failure)
            write(a.root/'status.json', {'phase': 'attention', 'version': a.version, 'error_type': type(exc).__name__})
            raise
    elif a.command == 'launch':
        from agenticgen.runtime.launch import check_api, idle_locks
        group_run.assert_prepared(a.root)
        from agenticgen.adaptive_schedule.environment import record_environment
        record_environment(a.root, read(a.root/'config.json'))
        if not (a.root/'ACCEPTANCE_COMPLETE.json').exists():
            raise RuntimeError('Native acceptance is incomplete')
        with idle_locks(a.root):
            api = check_api()
            if a.version != 'v1':
                from agenticgen.adaptive_schedule.group_review import review_check
                review_check(a.root, a.version)
            command = [sys.executable, '-u', '-m', 'agenticgen', 'run', '--root', str(a.root), '--version', a.version]
            request = {'commit': commit(), 'command': command, 'api_check': api, 'time': time.time(), 'environment': 'full inherited; values omitted'}
            write(a.root/'operations'/f'launch_{time.time_ns()}.json', request)
        with (a.root/f'worker_{a.version}.log').open('ab') as log:
            process = subprocess.Popen(command, cwd=repository(), env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = {**request, 'pid': process.pid, 'version': a.version}
        write(a.root/'launch.json', result)
    elif a.command == 'status':
        result = {'status': read(a.root/'status.json') if (a.root/'status.json').exists() else {'phase': 'prepared'},
                  'versions': {f'v{i}': (a.root/f'FINALIZED_v{i}.json').exists() for i in range(1, 21)}}
    elif a.command == 'report':
        result = group_run.report(a.root, a.version)
    else:
        from agenticgen.adaptive_schedule.group_review import review_check, review_pack
        result = (review_check if a.command == 'review-check' else review_pack)(a.root, a.version)
    print(result)


if __name__ == '__main__':
    main()
