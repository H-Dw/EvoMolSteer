"""Audit existing teacher selection, target clocks and paired outcomes.

Read-only: no model loading, generation, affinity evaluation or API calls.
Input: an existing EvoMolSteer repository; output: small JSON/CSV evidence.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd


def audit(repo, output):
    sources = {}

    def read(relative):
        path = repo / relative
        raw = path.read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        if path.suffix == '.csv':
            return pd.read_csv(path)
        return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

    previous = read('docs/experiments/post_m2_checkpoint_audit_20261017/results.json')
    cases = {r['id']: r for r in previous['rows']}
    m2 = cases['M2_400']
    native = cases['unguided_400']
    left, right = [read(c['files'][0]) for c in [m2, native]]
    left = left.loc[left.batch.isin(m2['batches'])]
    right = right.loc[right.batch.isin(m2['batches'])]
    joined = left.merge(right, on=['batch', 'slot'], validate='one_to_one',
                        suffixes=('_M2', '_native'))
    if len(joined) != 400:
        raise ValueError('Incomplete frozen M2/native pool')
    change = joined.pic50_on_rescore_M2 - joined.pic50_on_rescore_native
    valid = joined.valid_connected_M2 & joined.valid_connected_native
    paired = dict(n=len(joined), mean_delta=float(change.mean()),
                  median_delta=float(change.median()),
                  positive=int((change > 0).sum()), negative=int((change < 0).sum()),
                  above_0p01=int((change > .01).sum()),
                  both_valid=int(valid.sum()), positive_both_valid=int((change[valid] > 0).sum()))

    ref = read('configs/experiments/steer_dependency_v1/static.json.gz')
    first = ref['frames'][0]['teacher_endpoint_A']
    if not all(f['teacher_endpoint_A'] == first for f in ref['frames']):
        raise ValueError('Frozen M2 reference unexpectedly changes with time')
    frozen = dict(nodes=len(ref['times']), last_score_time=ref['times'][-1],
                  teachers=len(first), donor_batches=len(set(ref['frames'][0]['teacher_batches'])),
                  representation='predicted_endpoint_world_A',
                  label_semantics=ref['label_semantics'])

    # Timing semantics are bound to the inspected production classes, not names.
    matrix = [
        dict(id='M2_400', selector='instantaneous head at .49',
             target='teacher endpoint forecast produced at .49, frozen',
             live='current seed endpoint forecast', clock='.49', series='M2'),
        dict(id='M2REG_terminal_topk_global_0_0p5', selector='final score; Top10 discovery finals',
             target='final-selected ancestors endpoint forecasts at .49, frozen',
             live='current seed endpoint forecast', clock='.49', series='Top10'),
        dict(id='M2REG_terminal_topk_global_0p1_0p5', selector='final score; same Top10',
             target='final-selected ancestors endpoint forecasts at .49, frozen',
             live='current seed endpoint forecast', clock='.49', series='Top10'),
        dict(id='S3_static_to_observed_control', selector='instantaneous scores at each saved node',
             target='scalar blend: frozen .49 forecasts to independently selected same-clock forecasts',
             live='current seed endpoint forecast', clock='t', series='temporal'),
        dict(id='local_motif20_v2_20261015/R15_v2', selector='final candidate F_b011_s030',
             target='actual ancestor current coordinates at .5, frozen',
             live='predictive bridge to .5', clock='.5', series='local_v2'),
    ]
    clocks = {
        'R01_v3': ('current', 'actual ancestor current coordinates at t', 'current coordinates'),
        'R02_v3': ('endpoint_prediction', 'teacher endpoint forecast produced at t', 'endpoint forecast'),
        'R03_v3': ('lookahead', 'actual ancestor current coordinates at h(t)>t', 'bridge to h(t)'),
        'R04_v3': ('decaying_lookahead', 'actual ancestor current coordinates at h(t)>t', 'bridge to h(t)'),
        'R09_v3': ('fixed', 'actual ancestors at .5; static modulo seed routing', 'bridge to .5'),
        'R10_v3': ('fixed', 'actual ancestors at .5; nearest residual routing', 'bridge to .5'),
        'R11_v3': ('fixed', 'actual ancestors at .5; nearest with persistence', 'bridge to .5'),
        'R12_v3': ('fixed', 'actual ancestors at .5; VJP response with persistence', 'bridge to .5'),
        'R14_v3': ('fixed', 'actual ancestor at .5 plus typed endpoint probability field', 'bridge plus endpoint field'),
        'R30_v3': ('endpoint_prediction', 'teacher same-clock endpoint forecast; changed element matching', 'endpoint forecast'),
    }
    programs = {}
    prefix = 'docs/experiments/ancestral_modes35_v3_20261016'
    bank = read(prefix + '/bank.json')
    for rid, (policy, target, live) in clocks.items():
        p = read(f'{prefix}/{rid}/selected_program.json')
        if p['clock_policy'] != policy:
            raise ValueError(f'Clock contract changed: {rid}')
        programs[rid] = {k: p.get(k) for k in ['clock_policy', 'routing', 'window',
                           'mode_memory_t', 'switch_margin_A', 'assignment_element_cost',
                           'response_refresh_fraction', 'response_switch_fraction']}
        programs[rid]['templates'] = [dict(id=s['template_id'], n_points=len(s['atom_slots']),
            terminal_score_metadata=bank['modes'][s['template_id']]['terminal_score'],
            root=bank['modes'][s['template_id']]['root_id']) for s in p['mode_specs']]
        matrix.append(dict(id=f'ancestral_modes35_v3_20261016/{rid}',
            selector='final-selected path(s); Analyst chooses mode/support from registered options',
            target=target, live=live, clock=policy, series='ancestral_v3'))
    matrix.append(dict(id='ancestral_modes35_v3_20261016/R02_confirm400',
        selector='frozen R02 final-selected path', target='teacher endpoint forecast produced at t',
        live='endpoint forecast', clock='t', series='ancestral_v3'))
    for row in matrix:
        metrics = cases[row['id']]
        row.update({k: metrics[k] for k in ['n', 'mean', 'best', 'high_count', 'tail_rate']})
        row['commits'] = ','.join(metrics.get('commits', []))

    partial = {}
    for rid in ['R03', 'R12']:
        result = read(f'docs/experiments/partial_region_search_20261014/{rid}/results.json')
        partial[rid] = [{k: r[k] for k in ['condition', 'start', 'n', 'mean_all_pic50',
                         'best_qualified_pic50', 'atom_slots']} for r in result['conditions']]
    for a, b in zip(partial['R03'], partial['R12']):
        if a['start'] != b['start'] or a['atom_slots'] != b['atom_slots']:
            raise ValueError('Two-clock comparison has different times or points')
    single = read('docs/experiments/partial_region_search_20261014/R03/candidates/'
                  'F_b011_s030/programs/R03_0.1.json')
    mixed = read('docs/experiments/partial_region_search_20261014/R12/candidates/'
                 'F_b011_s030/programs/R12_0.1.json')
    for key in ['atom_slots', 'atom_weights', 'window', 'dose', 'teacher_time']:
        if single.get(key) != mixed.get(key):
            raise ValueError(f'Two-clock controlled parameter differs: {key}')
    if mixed['endpoint_mix'] != .5:
        raise ValueError('Two-clock coefficient changed')
    conditions = read('docs/experiments/local_motif20_v2_20261015/summary/conditions.csv')
    executed_endpoint = int(conditions.representation.eq('endpoint').sum())
    if executed_endpoint:
        raise ValueError('New pure-final condition requires updating this clock audit')

    for relative in ['src/evomolsteer/continuous/affinity_geometry.py',
                     'src/evomolsteer/continuous/dependency_ablation.py',
                     'src/evomolsteer/generation/affinity_geometry_reward.py',
                     'src/evomolsteer/ancestral_modes/reward.py',
                     'src/evomolsteer/ancestral_modes/programs.py',
                     'src/evomolsteer/partial_search/two_clock.py',
                     'src/evomolsteer/local_motif/reward.py']:
        sources[relative] = hashlib.sha256((repo / relative).read_bytes()).hexdigest()

    result = dict(schema_version='teacher-selection-clock-audit-1.0',
        new_generation=0, new_API_calls=0, new_head_calls=0, sources_sha256=sources,
        M2_paired_native=paired, M2_frozen_reference=frozen, matrix=matrix,
        routing_programs=programs, partial_two_clock=partial,
        local_v2_executed_pure_final_endpoint_conditions=executed_endpoint,
        caveats=['Historical teacher scores and source raw native scores are not experimental labels.',
                 'Representation packages change both target and live observable/derivative.',
                 'Different series or sample pools are not matched superiority comparisons.',
                 'No forward/SDE variance experiment was performed in this audit.'])
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix).to_csv(output / 'teacher_clock_matrix.csv', index=False, lineterminator='\n')
    (output / 'audit.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'matrix_rows': len(matrix), 'M2_paired_native': paired,
                      'pure_final_local_v2_runs': executed_endpoint}, ensure_ascii=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    audit(args.repo.resolve(), args.output.resolve())
