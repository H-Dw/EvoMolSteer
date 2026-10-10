"""Serial, committed, report-retained fifteen-round outcome campaign."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire


def dispatch(repo, work, manifest, previous=None, retry_failed=False):
    repo, work, manifest = map(lambda v: Path(v).resolve(), (repo, work, manifest))
    plan = json.loads(manifest.read_text())
    number = plan['round']
    if not 1 <= number <= 15 or plan['seed'] != 42:
        raise ValueError('Fresh fifteen-round seed-42 campaign required')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Remote source is dirty')
    for job in plan['jobs']:
        for key in ['program', 'reference']:
            path = (repo/job[key]).resolve()
            if not path.is_relative_to(repo/'configs/experiments/terminal_outcome15_v1'):
                raise ValueError('Committed campaign configuration required')
            subprocess.check_call(['git', '-C', str(repo), 'ls-files', '--error-unmatch', str(path.relative_to(repo))], stdout=subprocess.DEVNULL)
        p = json.loads((repo/job['program']).read_text())
        if p['round'] != number or p['seed'] != 42 or p.get('affinity_head_gradient') is not False:
            raise ValueError('Job round/seed/gradient contract')
    work.mkdir(parents=True, exist_ok=True)
    old = json.loads((work/'active.json').read_text()) if (work/'active.json').exists() else None
    if old:
        exit_file = work/f'round{old["round"]:02d}.exit'
        if (number != old['round']+1 and not (retry_failed and number == old['round'])) or not exit_file.is_file():
            raise ValueError('Prior serial inference incomplete')
        if retry_failed:
            if number != old['round'] or int(exit_file.read_text()) == 0:
                raise ValueError('Only a failed execution may be retried')
            attempt = 1+len(list(work.glob(f'round{number:02d}.failed*.log')))
            (work/f'round{number:02d}.log').rename(work/f'round{number:02d}.failed{attempt}.log')
            exit_file.rename(work/f'round{number:02d}.failed{attempt}.exit')
            for job in plan['jobs']:
                for suffix in ['source.before.json', 'source.after.json']:
                    record = work/(job['campaign']+'.'+suffix)
                    if record.is_file():
                        record.rename(record.with_name(record.name+f'.failed{attempt}'))
            report = repo/'docs/experiments/terminal_outcome15_20261010/protocol.json'
        elif not previous:
            raise ValueError('Prior result retention required')
        else:
            report = Path(previous).resolve()
            retained = verify_retention(report)
            if retained.get('round') != old['round']:
                raise ValueError('Retention does not describe the immediately preceding round')
    else:
        if number != 1:
            raise ValueError('Campaign starts at one')
        report = repo/'docs/experiments/terminal_outcome15_20261010/protocol.json'
    cleanup = {'allowed_bases': [str(work)], 'targets': [str(work/'generated'), str(work/'archives')],
        'result_report': str(report), 'protected': [str(repo), str(repo.parent/'flowr_root/checkpoints'),
            str(repo.parent/'flowr_root/experiments/ck2_clk3_lineage_20261003')]}
    cleanup_path = work/f'round{number:02d}.cleanup.plan.json'
    cleanup_path.write_text(json.dumps(cleanup))
    audit_path = work/f'round{number:02d}.cleanup.json'
    retire(cleanup_path, audit_path, True)
    if shutil.disk_usage(work).free < 600*1024**2:
        raise ValueError('Insufficient space after guarded cleanup')
    log = work/f'round{number:02d}.log'
    with log.open('wb') as stream:
        child = subprocess.Popen(['bash', str(repo/'scripts/scnet_outcome_round.sh'), str(manifest), str(work)],
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, 'FLOWR_ROOT': str(repo.parent/'flowr_root')})
    result = {'round': number, 'maximum_rounds': 15, 'pid': child.pid, 'manifest': str(manifest),
              'code_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
              'jobs': plan['jobs'], 'cleanup_audit': str(audit_path)}
    (work/'active.json').write_text(json.dumps(result))
    (work/f'round{number:02d}.launch.json').write_text(json.dumps(result))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['repo', 'work', 'manifest']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--previous')
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    dispatch(a.repo, a.work, a.manifest, a.previous, a.retry_failed)
