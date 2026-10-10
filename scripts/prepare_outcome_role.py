"""Export immutable role requests for one serial label/formula experiment."""
import argparse
from pathlib import Path
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.continuous.outcome_agents import export_request

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'
CFG = ROOT/'configs/experiments/terminal_outcome15_v1'

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True)
    p.add_argument('--kind', default='mean')
    p.add_argument('--base', required=True, help='Repository-relative committed/frozen parent program')
    p.add_argument('--role', choices=['Analyst', 'Designer'], required=True)
    p.add_argument('--allowed-update', action='append', default=[])
    p.add_argument('--task', required=True)
    p.add_argument('--analyst')
    p.add_argument('--revision', default='agents')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--summary-key')
    p.add_argument('--evidence', help='Registered extension/feedback evidence override')
    p.add_argument('--receipt', help='Receipt for the overridden evidence')
    a = p.parse_args()
    output = DOC/f'round{a.round:02d}'/a.revision
    registry_path = output/'registry.json'
    base = ROOT/a.base
    reference = CFG/'references'/(a.kind+'.json.gz')
    if not registry_path.exists():
        write_json(registry_path, {'task': a.task, 'round': a.round, 'allowed_updates': a.allowed_update,
            'base_programs': {read_json(base)['program_id']: {'path': str(base), 'sha256': digest(base)}},
            'reference': {'path': str(reference), 'sha256': digest(reference)},
            'formula_registry': ['endpoint_pointcloud', 'endpoint_branch_mixture'],
            'history': [str(p.relative_to(ROOT)) for p in sorted(DOC.glob('round*/summary.json'))]})
    summary_key = a.summary_key or a.kind
    evidence = DOC/'summaries'/summary_key/'evidence.json' if a.summary else DOC/'mining'/a.kind/'evidence.json'
    receipt = DOC/'tool_receipts'/(summary_key+'.summary.json' if a.summary else a.kind+'.json')
    if bool(a.evidence) != bool(a.receipt):
        raise ValueError('Evidence override and actual receipt must be supplied together')
    if a.evidence:
        evidence, receipt = Path(a.evidence), Path(a.receipt)
    print(export_request(a.role, evidence, registry_path, receipt, output, a.analyst))
