"""Reproduce descriptive results from the immutable remote audit; no inference."""
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
ARMS = ['chemical_native', 'joint_native', 'chemical_raw999', 'joint_raw999']
SEEDS = [42, 100045, 200048, 300051, 400054]
CONTRASTS = [('joint_native', 'chemical_native'), ('chemical_raw999', 'chemical_native'),
             ('joint_raw999', 'chemical_raw999'), ('joint_raw999', 'joint_native'),
             ('joint_raw999', 'chemical_native')]


def export_csv(name, rows):
    with (ROOT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with gzip.open(ROOT / 'audit.json.gz', 'rt', encoding='utf-8') as stream:
        audit = json.load(stream)
    assert len(audit['groups']) == 8
    assert audit['roots']['SAMPLING_FINISHED.json']['completed_groups'] == 6
    assert sum(r['count'] for r in audit['roots']['attempts.json'].values()) == 300
    assert all(audit['monitor']['source_hash_check'].values())
    summary, paired, contrasts, seed_rows = [], [], [], []
    for name, group in audit['groups'].items():
        target, arm = name.split('/')
        files = group['files']
        records, quality = files['records.json'], files['quality.json']
        receipt = files['QUALITY_COMPLETE.json']
        assert len(records) == 50
        assert {(r['logical_seed'], r['logical_slot']) for r in records} == {(s, i) for s in SEEDS for i in range(10)}
        assert (receipt['attempts'], receipt['checked_attempts'], receipt['not_selected'], receipt['coverage']) == (50, 10, 40, .2)
        selected = {(r['logical_seed'], r['logical_slot']): r['node_id'] for r in receipt['selection']}
        assert set(selected) == {(s, i) for s in SEEDS for i in [0, 1]}
        assert all(selected[(r['logical_seed'], r['logical_slot'])] == r['node_id'] for r in records if r['logical_slot'] < 2)
        assert group['source_sha256']['records.json'] == receipt['records_sha256']
        assert group['source_sha256']['quality.json'] == receipt['quality_sha256']
        if 'PAIRED_PREFIX_ACCEPTANCE.json' in files:
            assert all(files['PAIRED_PREFIX_ACCEPTANCE.json'].values())
        clocks = files.get('NETWORK_TIMES.json', [])
        if 'raw999' in arm:
            assert not any(e['phase'] == 'native_terminal_forward' for e in clocks)
            assert files['LAST_LOOP.json']['step'] == 99
            assert abs(files['LAST_LOOP.json']['network_times'][0] - .98901) < 1e-6
            assert files['OUTPUT_POLICY.json']['terminal_projection_applied'] is False
        failures = collections.Counter(k for row in quality for k, value in row['quality'].get('posebusters_checks', {}).items() if value is False)
        recovery = [r['plip_recovery'] for r in quality if r.get('plip_recovery') is not None]
        c = [r['condition_rmsd_A'] for r in records]
        steps = files.get('steps.json', [])
        largest = max(steps, key=lambda r: max(r['max_atom_increment_A']))
        index = largest['max_atom_increment_A'].index(max(largest['max_atom_increment_A']))
        row = dict(target=target, arm=arm, attempts=50,
                   mean_pic50=st.mean(r['pic50_on_rescore'] for r in records),
                   C_mean_A=st.mean(c), C_median_A=st.median(c), C_max_A=max(c),
                   final_C_is_imposed=arm == 'joint_native',
                   raw_C=files.get('OUTPUT_POLICY.json', {}).get('raw_C'),
                   built=sum(r['build_success'] for r in records),
                   connected=sum(r['build_success'] and r['connected'] for r in records),
                   unique_smiles=len({r['smiles'] for r in records if r['build_success']}),
                   checked_attempts=10, posebusters_pass=receipt['posebusters_pass'],
                   plip_evaluable=receipt['plip_evaluable'], plip_recovery_mean=st.mean(recovery),
                   posebusters_unknown=receipt['posebusters_unevaluable_built'],
                   plip_unknown=receipt['plip_unevaluable_built'],
                   pb_failed_checks=dict(failures),
                   largest_guidance_increment=dict(step=largest['step'], tensor_row=index,
                       seed=records[index]['logical_seed'], slot=records[index]['logical_slot'],
                       A=largest['max_atom_increment_A'][index], scale=largest['scale'],
                       native_ratio=largest['delta_native_ratio'][index]),
                   clock_checks=[e for e in clocks if e['phase'] in ['integration:99', 'native_terminal_forward', 'endpoint_rescore']],
                   build_failures=[{k: r[k] for k in ['logical_seed', 'logical_slot', 'node_id', 'failure_reason']} for r in records if not r['build_success']])
        assert abs(row['mean_pic50'] - files['aggregate.json']['mean_pic50']) < 1e-10
        assert abs(row['C_mean_A'] - files['aggregate.json']['C']) < 1e-10
        summary.append(row)
        for r in records:
            paired.append(dict(target=target, arm=arm, seed=r['logical_seed'], slot=r['logical_slot'],
                               node_id=r['node_id'], pic50=r['pic50_on_rescore'], C_A=r['condition_rmsd_A'],
                               built=r['build_success'], connected=r.get('connected'), smiles=r.get('smiles')))
    for target in audit['roots']['STUDY.json']['controls']:
        values = {a: {(r['logical_seed'], r['logical_slot']): r for r in audit['groups'][target + '/' + a]['files']['records.json']} for a in ARMS}
        for left, right in CONTRASTS:
            for metric in ['pic50_on_rescore', 'condition_rmsd_A']:
                deltas = [st.mean(values[left][(seed, i)][metric] - values[right][(seed, i)][metric] for i in range(10)) for seed in SEEDS]
                mean = st.mean(deltas)
                margin = 2.7764451051977987 * st.stdev(deltas) / math.sqrt(5)
                contrasts.append(dict(target=target, left=left, right=right, metric=metric,
                                      mean_delta=mean, seed_deltas=deltas, exploratory_ci95=[mean - margin, mean + margin],
                                      independent_holdout=False, seed_clusters=5))
                seed_rows.extend(dict(target=target, left=left, right=right, metric=metric, seed=seed, mean_delta=delta) for seed, delta in zip(SEEDS, deltas))
    result = dict(audit_sha256=hashlib.sha256((ROOT / 'audit.json.gz').read_bytes()).hexdigest(),
                  new_attempts=300, reused_attempts=100, groups=summary, contrasts=contrasts,
                  inference='Descriptive fixed-seed comparisons; no multiplicity adjustment or independent holdout. Imposed C is not evidence of improvement.')
    (ROOT / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    export_csv('paired_records.csv', paired)
    export_csv('seed_deltas.csv', seed_rows)
    export_csv('metrics.csv', [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in summary])
    print(json.dumps({'groups': len(summary), 'paired_records': len(paired), 'contrasts': len(contrasts), 'audit_sha256': result['audit_sha256']}))


if __name__ == '__main__':
    main()
