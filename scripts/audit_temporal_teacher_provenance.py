"""Audit local frozen teacher/role records; never invoke an API or generation.

This verifies the compact libraries and collected role ledger, not the remote
raw trajectories. Original-coordinate provenance is a separate prior audit.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path


def audit(repo):
    sources = {}

    def read(name):
        data = (repo / name).read_bytes()
        sources[name] = hashlib.sha256(data).hexdigest()
        return data

    def document(name):
        data = read(name)
        return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)

    cfg = document('configs/default.json')
    program = document('configs/experiments/steer_dependency_v1/D2_without_trajectory_static.json')
    static = document('configs/experiments/steer_dependency_v1/static.json.gz')
    dynamic = document('configs/experiments/skill_ablation_v1/endpoint_reference.json.gz')
    rows = list(csv.DictReader(read('docs/experiments/temporal_direction_comparison_20261013/latest_role_chains.csv').decode('utf-8-sig').splitlines()))
    checks = {
        'static_reference_hash_matches_program': program['reference_sha256'] == sources['configs/experiments/steer_dependency_v1/static.json.gz'],
        'static_geometry_equals_last_dynamic_frame_at_every_node': all(f['teacher_endpoint_A'] == dynamic['frames'][-1]['teacher_endpoint_A'] for f in static['frames']),
        'static_teacher_scores_all_zero': all(s == 0 for f in static['frames'] for s in f['teacher_scores']),
        'static_score_beta_zero': program['teacher_score_beta'] == 0,
        'matching_score_grids': static['times'] == dynamic['times'],
        'discovery_split_matches_teacher_library': cfg['discovery_batches'] == dynamic['discovery_batches'],
    }
    for row in rows:
        base = f"docs/experiments/{row['revision']}/calls/{row['condition']}/chain_{int(row['chain']):02d}"
        analyst = document(base + '/Analyst.json')
        designer = document(base + '/Designer.json')
        if len(analyst['hypotheses']) != int(row['hypotheses']):
            raise ValueError('Collected hypothesis count differs from literal output')
        expected = designer['edit']['value'] if designer['decision'] == 'edit' else 'retain_M2'
        if expected != row['recipe']:
            raise ValueError('Collected recipe differs from literal Designer output')
    for name in [
        'docs/experiments/steer_dependency_20261009/protocol_amendment_information.json',
        'docs/experiments/steer_dependency_20261009/luna_suite/protocol.json',
        'docs/experiments/steer_dependency_20261009/luna_suite/D2_without_trajectory/Analyst.response.json',
        'docs/experiments/steer_dependency_20261009/luna_suite/D2_without_trajectory/Designer.response.json',
        'src/evomolsteer/continuous/affinity_geometry.py',
        'src/evomolsteer/continuous/affinity_endpoint.py',
        'src/evomolsteer/continuous/dependency_ablation.py',
        'src/evomolsteer/continuous/path_graph.py',
        'src/evomolsteer/continuous/elite_path_credit.py',
        'src/evomolsteer/continuous/terminal_path_library.py',
    ]:
        read(name)
    if not all(checks.values()):
        raise ValueError(checks)
    s_rows = [r for r in rows if r['condition'].startswith('S')]

    def counts(items, key):
        return dict(sorted(collections.Counter(r[key] for r in items).items()))

    return {
        'scope': 'Local compact artifacts and literal role outputs; no new API, remote raw-data audit or inference',
        'checks': checks,
        'split': {k: cfg[k] for k in ('discovery_batches', 'validation_batches', 'heldout_batches')},
        'window': static['window'],
        'score_nodes': len(static['times']),
        'first_last_score_times': [static['times'][0], static['times'][-1]],
        'teachers_per_static_frame': sorted(set(len(f['teacher_endpoint_A']) for f in static['frames'])),
        'last_frame_teachers_by_batch': dict(sorted(collections.Counter(dynamic['frames'][-1]['teacher_batches']).items())),
        'role_pairs': len(rows),
        'curves_read_histogram_from_collected_ledger': counts(rows, 'curves_read'),
        'S_role_pairs': len(s_rows),
        'S_curves_read_histogram_from_collected_ledger': counts(s_rows, 'curves_read'),
        'S_recipes_verified_against_literal_outputs': counts(s_rows, 'recipe'),
        'source_sha256': sources,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, ensure_ascii=False))
