"""Retire only generated flow-compatibility outputs after every local report is verified."""
import argparse
import json
from pathlib import Path
import re

from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire


def finalize(repo, work, apply=False):
    repo, work = Path(repo).resolve(), Path(work).resolve()
    flowr = repo.parent / 'flowr_root'
    allowed = (flowr / 'experiments').resolve()
    if not work.is_relative_to(allowed) or work == allowed:
        raise ValueError('Explicit FLOWR experiment workspace required')
    cfg = json.loads((repo / 'configs/experiments/flowcompat30_v1/campaign.json').read_text())
    last = cfg['maximum_rounds']
    if (last != 30 or cfg['rounds_completed'] != last
            or [r['round'] for r in cfg['rounds']] != list(range(1, last + 1))
            or any(r['status'] != 'complete' for r in cfg['rounds'])):
        raise ValueError('All registered rounds must be retained first')
    reports = repo / 'docs/experiments/flowcompat30_20261009'
    for n in range(1, last + 1):
        if verify_retention(reports / f'round{n:02d}/retention.json')['campaign'] != f'flowcompat_r{n:02d}':
            raise ValueError('Retained round identity differs')
    active = json.loads((work / 'active.json').read_text())
    if (active['round'] != last or active['campaign'] != f'flowcompat_r{last:02d}'
            or (work / f'round{last:02d}.exit').read_text().strip() != '0'):
        raise ValueError('Final inference is not complete')
    targets = [work / 'generated']
    for archive in sorted(work.glob('flowcompat_r*.tar.gz')):
        match = re.fullmatch(r'flowcompat_r(\d{2})(?:\.full_transport)?\.tar\.gz', archive.name)
        if not match or not 1 <= int(match[1]) <= last:
            raise ValueError('Unrecognized disposable archive')
        targets.extend([archive, Path(str(archive) + '.json')])
    report = reports / f'round{last:02d}/retention.json'
    plan = {'allowed_bases': [str(work)], 'targets': [str(p) for p in targets],
            'result_report': str(report), 'protected': [str(repo), str(flowr / 'checkpoints'),
                str(flowr / 'experiments/ck2_clk3_lineage_20261003')]}
    plan_file = work / 'final_cleanup.plan.json'
    plan_file.write_text(json.dumps(plan, indent=2) + '\n')
    audit = work / 'final_cleanup.json'
    retire(plan_file, audit, apply)
    if apply:
        active.update(status='complete', rounds_completed=last, generated_outputs_retired=True)
        (work / 'active.json').write_text(json.dumps(active, indent=2) + '\n')
        ready = json.loads((work / 'round_ready.json').read_text())
        ready['status'] = 'complete'
        (work / 'round_ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    return json.loads(audit.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    finalize(args.repo, args.work, args.apply)

