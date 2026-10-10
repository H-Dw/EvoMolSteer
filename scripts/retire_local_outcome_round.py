"""Retire only this campaign's local transport after report verification."""
import argparse
from pathlib import Path
from evomolsteer.io import write_json
from dispatch_path_round import verify_retention
from retire_experiment_outputs import retire

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True)
    a = p.parse_args()
    if not 1 <= a.round <= 15:
        raise ValueError('Registered outcome trial required')
    report = DOC/f'round{a.round:02d}/retention.json'
    verify_retention(report)
    work = ROOT/'test/terminal_relabel15'
    plan = work/f'retire{a.round:02d}.json'
    write_json(plan, {'allowed_bases':[str(work)], 'targets':[str(work/f'round{a.round:02d}')],
        'protected':[str(ROOT/'data/raw'),str(ROOT/'src'),str(ROOT/'configs'),str(ROOT/'docs')],
        'result_report':str(report)})
    retire(plan, DOC/f'retirement/round{a.round:02d}.local.json', True)
