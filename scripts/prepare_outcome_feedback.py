"""Register and execute compact feedback calculations for a serial next round."""
import argparse
import re
from pathlib import Path
from evomolsteer.io import write_json
from evomolsteer.continuous.outcome_agents import run_tool

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--revision', help='New immutable feedback revision after a rollback or corrected request')
    a = p.parse_args()
    if not 3 <= a.round <= 15:
        raise ValueError('Feedback requires completed predecessor trials')
    evidence = (ROOT/a.evidence).resolve()
    reports = [DOC/f'round{r:02d}/summary.json' for r in range(1, a.round)]
    if not all(r.is_file() for r in reports):
        raise ValueError('Serial predecessor reports missing')
    if a.revision and not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', a.revision):
        raise ValueError('Simple revision identifier required')
    key = f'round{a.round:02d}' + ('.'+a.revision if a.revision else '')
    output = DOC/f'feedback/{key}'
    plan = DOC/f'tool_plans/{key}.feedback.json'
    receipt = DOC/f'tool_receipts/{key}.feedback.json'
    if plan.exists() or receipt.exists() or output.exists():
        raise FileExistsError('Use a fresh immutable feedback revision')
    write_json(plan, {'tool_id': 'outcome_feedback',
        'arguments': ['--evidence', str(evidence), '--reports', '|'.join(map(str, reports)), '--output', str(output)],
        'input_files': [str(evidence), *map(str, reports)], 'output_files': [str(output/'evidence.json')]})
    print(run_tool(plan, receipt))
