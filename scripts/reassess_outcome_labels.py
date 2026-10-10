"""Read-only comparison and byte-exact recovery of recorded ablation programs.

Explicit JSON input manifest -> compact tables, provenance, and recovery bundle.
Does not change active inference, launch generation, or fit a new reward model.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from analyze_dependency_study import metrics, paired


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')


def run(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    plan = read(manifest_path)
    root = Path(plan['repo']).resolve()
    output = Path(output).resolve()
    protected_before = {p:digest(root/p) for p in plan.get('protected_inputs', [])}
    frames, outcomes, sources = {}, {}, {}
    for item in plan['cohorts']:
        path = root/item['metrics']
        if digest(path) != item['metrics_sha256']:
            raise ValueError(f"Metric source changed: {item['name']}")
        frames[item['name']], outcomes[item['name']] = metrics(path, item['arm'])
        if sorted(frames[item['name']].batch.unique().tolist()) != plan['screen_batches']:
            raise ValueError('Mixed screens are not a matched ablation')
        sources[item['metrics']] = digest(path)
    eligible = plan['ranking_eligible']
    order = sorted(eligible, key=lambda n: (-outcomes[n]['affinity_mean_all'], n))
    winner = order[0]
    comparisons = {name: {control: paired(frames[name], frames[control])
                         for control in plan['paired_controls'] if name != control}
                   for name in eligible}

    recovery = {}
    for item in plan['recover']:
        folder = output/'recovered'/item['name']
        record = dict(item)
        for key in ('program', 'reference', 'job'):
            relative = item[key]
            data = subprocess.check_output(['git', 'show', f"{item['commit']}:{relative}"], cwd=root)
            expected = item[key+'_sha256']
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Historical {key} differs from execution: {relative}')
            folder.mkdir(parents=True, exist_ok=True)
            target = folder/Path(relative).name
            target.write_bytes(data)
            record[key+'_snapshot'] = target.relative_to(output).as_posix()
        job = read(folder/Path(item['job']).name)
        program = read(folder/Path(item['program']).name)
        if (job['program_sha256'] != item['program_sha256']
                or job['reference_sha256'] != item['reference_sha256']
                or program['reference_sha256'] != item['reference_sha256']):
            raise ValueError('Job/program/reference binding mismatch')
        archive = root/item['archive_verification']
        proof = read(archive)
        remote = read(root/item['remote_execution'])
        if (not proof['verified'] or proof['code_commit'] != item['commit']
                or remote['code_commit'] != item['commit']
                or proof['archive_sha256'] != remote['transport']['archive_sha256']):
            raise ValueError('Recovered execution lacks an archive verification')
        audit = read(root/item['execution_audit'])
        if not audit['passed'] or any(row['controlled_steps'] != 50 for row in audit['batches']):
            raise ValueError('Historical 50-step boundary needs separate handling')
        record['controlled_steps'] = 50
        record['nominal_state_boundary'] = .5
        record['promotion_status'] = 'two_batch_screen_only_not_independently_confirmed'
        reference = json.loads(gzip.decompress((folder/Path(item['reference']).name).read_bytes()))
        record['reference_label_semantics'] = reference.get('label_semantics')
        if item['name'] == 'M2':
            fixed = all(f['teacher_endpoint_A'] == reference['frames'][0]['teacher_endpoint_A']
                        for f in reference['frames'])
            equal = all(len(set(f['teacher_scores'])) == 1 for f in reference['frames'])
            if not fixed or not equal or program['teacher_score_beta'] != 0:
                raise ValueError('M2 fixed equal-score teacher contract changed')
            record['fixed_teacher_coordinates_verified'] = fixed
            record['equal_score_teacher_prior_verified'] = equal
        recovery[item['name']] = record
        for key in ('archive_verification', 'remote_execution', 'execution_audit'):
            sources[item[key]] = digest(root/item[key])

    outcome_dir = root/plan['outcome_reports']
    label_contrasts = {}
    for number, key in ((3, 'R26_mean/R26_051'), (4, 'R11_mean/R11_051')):
        path = outcome_dir/f'round{number:02}/summary.json'
        summary = read(path)
        label_contrasts[f'TO{number:02}'] = summary['paired_comparisons'][key]
        sources[path.relative_to(root).as_posix()] = digest(path)
    decision_path = outcome_dir/'final_decision.json'
    decision = read(decision_path)
    sources[decision_path.relative_to(root).as_posix()] = digest(decision_path)
    descendants_path = outcome_dir/'summaries/mean/summary.json'
    descendants = read(descendants_path)['descendants']
    sources[descendants_path.relative_to(root).as_posix()] = digest(descendants_path)

    historical_path = root/plan['historical_steer']
    hist = pd.read_csv(historical_path)
    valid = hist[hist.valid_connected]
    historical = {'n': len(hist), 'valid': len(valid),
                  'mean_all': float(hist.pic50_on_rescore.mean()),
                  'mean_valid': float(valid.pic50_on_rescore.mean()),
                  'p95_all': float(hist.pic50_on_rescore.quantile(.95)),
                  'p95_valid': float(valid.pic50_on_rescore.quantile(.95)),
                  'elite_valid': int(valid.pic50_on_rescore.ge(plan['elite_threshold']).sum())}
    sources[plan['historical_steer']] = digest(historical_path)
    remote_path = root/plan['remote_status'] if 'remote_status' in plan else output/'remote_status.json'
    remote_status = read(remote_path)
    sources['remote_status.json'] = digest(remote_path)
    result = {'schema_version': 'outcome-label-reassessment-1.0', 'screen_order': order,
              'screen_best': winner, 'screen_outcomes': outcomes, 'paired_screen': comparisons,
              'recovered': recovery, 'isolated_label_development_contrasts': label_contrasts,
              'frozen_outcome_confirmation': {k: decision[k] for k in ('metrics', 'paired', 'candidate_promoted', 'uncertainty')},
              'historical_steer_unpaired': historical, 'ancestor_support': descendants,
              'remote_confirmation_queues': remote_status['queues'],
              'sources_sha256': sources, 'manifest_sha256': digest(manifest_path),
              'analysis_script_sha256': digest(__file__),
              'protected_inputs_sha256': protected_before,
              'active_program_changed': False, 'new_generation_submitted': False,
              'interpretation': 'No superiority for this final-label implementation; not a universal impossibility proof. Screen best is not an independently confirmed default.'}
    write(output/'reassessment.json', result)
    pd.DataFrame([{'condition': n, **{k:v for k,v in m.items() if not isinstance(v, dict)}}
                  for n,m in outcomes.items()]).to_csv(output/'screen_metrics.csv', index=False)
    # Small, exact metric inputs survive future retirement of large trajectories.
    replay = copy.deepcopy(plan)
    for item in replay['cohorts']:
        source = root/item['metrics']
        target = output/'input_metrics'/f"{item['metrics_sha256']}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        if digest(target) != item['metrics_sha256']:
            raise ValueError('Recovery metric copy changed')
        item['metrics'] = str(target.relative_to(root)).replace('\\', '/') if target.is_relative_to(root) else str(target)
    write(output/'replay_manifest.json', replay)
    if protected_before != {p:digest(root/p) for p in protected_before}:
        raise ValueError('Read-only reassessment modified a protected input')
    print(json.dumps({'screen_best': winner, 'screen_n': outcomes[winner]['n'],
                      'mean': outcomes[winner]['affinity_mean_all'], 'recovered': list(recovery),
                      'active_program_changed': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)
