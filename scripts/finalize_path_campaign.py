"""Report-gated final retirement after all sequential inference rounds finish."""
import argparse
import json
from pathlib import Path
import re

from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire


def finalize(repo, work, apply=False):
    repo, work = Path(repo).resolve(), Path(work).resolve()
    flowr = repo.parent / 'flowr_root'
    if not work.is_relative_to(flowr / 'experiments') or work == flowr / 'experiments':
        raise ValueError('Explicit FLOWR experiment workspace required')
    config = json.loads((repo / 'configs/experiments/elite_path20_v1/campaign.json').read_text())
    last = config['maximum_rounds']
    if config['rounds_completed'] != last or [x['round'] for x in config['rounds']] != list(range(1, last + 1)):
        raise ValueError('All registered rounds must be retained first')
    if any(x['status'] != 'complete' for x in config['rounds']):
        raise ValueError('Unresolved round')
    active = json.loads((work / 'active.json').read_text())
    if active['round'] != last or (work / f'round{last:02d}.exit').read_text().strip() != '0':
        raise ValueError('Final inference is not complete')
    campaign = f'elite_path_r{last:02d}'
    if active['campaign'] != campaign:
        raise ValueError('Wrong active campaign')
    report = repo / f'docs/experiments/elite_path20_20261007/round{last:02d}/retention.json'
    if verify_retention(report)['campaign'] != campaign:
        raise ValueError('Wrong retained final evidence')
    targets = [work / 'generated']
    for archive in work.glob('elite_path_r*.tar.gz'):
        match = re.fullmatch(r'elite_path_r(\d{2})(?:\.full_transport)?\.tar\.gz', archive.name)
        if not match or not 4 <= int(match[1]) <= last:
            raise ValueError('Unrecognized disposable archive')
        prior = repo / f'docs/experiments/elite_path20_20261007/round{int(match[1]):02d}/retention.json'
        verify_retention(prior)
        targets.extend([archive, Path(str(archive) + '.json')])
    plan = {'allowed_bases': [str(work)],
            'protected': [str(repo), str(flowr / 'experiments/ck2_clk3_lineage_20261003'), str(flowr / 'checkpoints')],
            'result_report': str(report), 'targets': [str(p) for p in targets]}
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
