"""Local full-continuation evaluation and compact retained round comparison."""
import argparse
from pathlib import Path
import shutil

import pandas as pd
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.generation.terminal_evaluation import evaluate_terminal
from evomolsteer.generation.path_evaluation import retain_round, summarize_tail, paired_effect
from evomolsteer.continuous.outcome_execution import response_records

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT/'docs/experiments/terminal_outcome15_20261010'


def score(manifest, dataset, scratch, output):
    plan = read_json(manifest)
    dataset, scratch, out = map(Path, (dataset, scratch, output))
    threshold = read_json(DOC/'protocol.json')['tail_threshold']
    results = {}
    for job in plan['jobs']:
        cohort, campaign = job['cohort'], job['campaign']
        evaluated = scratch/cohort
        target = out/'results'/cohort
        if not (evaluated/'terminal_report.json').exists():
            evaluate_terminal(dataset, campaign, ROOT/'configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz',
                              evaluated, job['arms'].split(','), job['batches'], workers=2)
        if not target.exists():
            retain_round(dataset, campaign, evaluated, target, threshold)
        results[cohort] = read_json(target/'comparison.json')['results']
    comparisons = {}
    for cohort in results:
        d = pd.read_csv(out/'results'/cohort/'candidate_metrics.csv')
        candidate = d[d.arm == 'gradient']
        if len(d[d.arm == 'unguided']):
            comparisons[cohort+'/within_job_native'] = paired_effect(candidate, d[d.arm == 'unguided'])
        # Frozen confirmation jobs carry clock-matched historical controls on
        # the same new batches. They must not silently reuse development data.
        for control in results:
            if control == cohort:
                continue
            control_table = pd.read_csv(out/'results'/control/'candidate_metrics.csv')
            control_table = control_table[(control_table.arm == 'gradient')&
                                          control_table.batch.isin(candidate.batch.unique())]
            if len(control_table) != len(candidate):
                continue
            execution = read_json(out/'results'/cohort/'execution_report.json')
            control_execution = read_json(out/'results'/control/'execution_report.json')
            for batch in candidate.batch.unique():
                key = 'gradient/'+str(batch)
                if execution['initial_state_signatures'][key] != control_execution['initial_state_signatures'][key]:
                    raise ValueError('Same-round candidate/control initial states differ')
            comparisons[cohort+'/same_round_'+control] = paired_effect(candidate, control_table)
        for control, folder, arm in [('native', DOC/'round01/results/R26', 'unguided'),
                                    ('R26_051', DOC/'round01/results/R26', 'gradient'),
                                    ('R11_051', DOC/'round02/results/R11', 'gradient')]:
            path = folder/'candidate_metrics.csv'
            if not path.is_file():
                continue
            old = pd.read_csv(path)
            old = old[(old.arm == arm)&old.batch.isin(candidate.batch.unique())]
            if len(old) != len(candidate):
                continue
            execution = read_json(out/'results'/cohort/'execution_report.json')
            old_execution = read_json(folder/'execution_report.json')
            for batch in candidate.batch.unique():
                if execution['initial_state_signatures']['gradient/'+str(batch)] != old_execution['initial_state_signatures'][arm+'/'+str(batch)]:
                    raise ValueError('Cross-campaign paired initial coordinate/atom/bond states differ')
            comparisons[cohort+'/'+control] = paired_effect(candidate, old)
        executed = read_json(out/'results'/cohort/'inference_config/reward_program.json')
        binding = executed.get('outcome_binding', {})
        parent_sha = binding.get('base_program_sha256')
        if parent_sha:
            parents = [p for p in (ROOT/'configs/experiments/terminal_outcome15_v1').glob('round*.json')
                       if '.jobs.' not in p.name and digest(p) == parent_sha]
            if len(parents) != 1:
                raise ValueError('Registered candidate parent cannot be identified uniquely')
            parent = read_json(parents[0])
            for folder in (DOC/f'round{parent["round"]:02d}/results').glob('*'):
                old_program = folder/'inference_config/reward_program.json'
                if not old_program.is_file() or read_json(old_program) != parent:
                    continue
                old = pd.read_csv(folder/'candidate_metrics.csv')
                old = old[(old.arm == 'gradient')&old.batch.isin(candidate.batch.unique())]
                if len(old) != len(candidate):
                    continue
                execution = read_json(out/'results'/cohort/'execution_report.json')
                prior = read_json(folder/'execution_report.json')
                for batch in candidate.batch.unique():
                    key = 'gradient/'+str(batch)
                    if execution['initial_state_signatures'][key] != prior['initial_state_signatures'][key]:
                        raise ValueError('Candidate and registered immediate parent initial states differ')
                comparisons[cohort+'/immediate_parent'] = {**paired_effect(candidate, old),
                    'parent_round': parent['round'], 'parent_program_id': parent['program_id'],
                    'parent_program_sha256': parent_sha}
    original = pd.read_csv(ROOT/'docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv')
    summary = {'round': plan['round'], 'status': 'complete', 'seed': 42, 'results': results,
        'paired_comparisons': comparisons, 'historical_steer_unpaired': summarize_tail(original, threshold),
        'interpretation': 'Fixed development batches are adaptive screening; frozen new batches required for confirmation',
        'label_response': {}, 'manifest_sha256': digest(manifest)}
    for job in plan['jobs']:
        p = read_json(ROOT/job['program'])
        executed = read_json(out/'results'/job['cohort']/'inference_config/reward_program.json')
        if p != executed:
            raise ValueError('Executed reward program differs from committed local configuration')
        execution = read_json(out/'results'/job['cohort']/'execution_report.json')
        # Runtime support is always derived; this count is computed from clocks.
        import numpy as np
        count = int(np.sum((np.arange(100)/100 >= p['window'][0]-2e-6)&
                           ((np.arange(100)+1)/100 <= p['window'][1]+2e-6)))
        if any(r['controlled_steps'] != count for r in execution['batch_results'] if r['arm'] == 'gradient'):
            raise ValueError('Observed guidance support does not respond to committed boundary')
        binding = p.get('outcome_binding')
        summary['label_response'][job['cohort']] = {'controlled_steps': count,
            'score_window': p.get('score_window'), 'state_control_window': p['window'],
            'reference_sha256': p['reference_sha256'], 'label_source': binding['label_source'] if binding else 'instantaneous_control',
            'agent_binding_present': bool(binding), 'mean_cumulative_rms_A':
                [r['mean_cumulative_rms_A'] for r in execution['batch_results'] if r['arm'] == 'gradient'],
            'actual_response': response_records(dataset, job['campaign'], p)}
    write_json(out/'summary.json', summary)
    files = [{'path': p.relative_to(out).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'retention.json']
    # The aggregate report gate covers all cohort reports, literal Agent inputs
    # and scientific comparisons before this round's structures are retired.
    write_json(out/'retention.json', {'status': 'complete', 'campaign': f'outcome_round{plan["round"]:02d}',
        'files': files, 'round': plan['round'], 'retention': 'Report-only after verified local evaluation'})
    print({'round': plan['round'], 'results': {k: {a: v['valid_mean_pic50'] for a,v in values.items()} for k,values in results.items()},
           'paired': {k: v['paired_mean_pic50'] for k,v in comparisons.items()}})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['manifest', 'dataset', 'scratch', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    score(a.manifest, a.dataset, a.scratch, a.output)
