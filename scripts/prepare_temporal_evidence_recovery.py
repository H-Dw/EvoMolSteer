"""Prepare a fresh evidence-interface retry; never call API, SSH or FLOWR.

Only previously failed role conditions are selected. Scientific recipes, RNG,
temperatures, reference files and comparison cohorts remain frozen. Numerical
results use the prior pinned collection audit. Completed roles are not overwritten.
"""
import argparse
import copy
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.evidence_protocol import PROTOCOL
from evomolsteer.temporal_guidance.workflow import inside, prepare


def build(repo, prior_runner, config_output, study, smoke_condition=None):
    prior_runner = inside(repo, prior_runner)
    runner = read_json(prior_runner)
    old = read_json(inside(repo, runner['plan_config']))
    state = read_json(inside(repo, runner['study']) / 'run_status.json')
    if state['status'] != 'complete_with_skipped_errors':
        raise ValueError('Retry requires an archived finished queue with failed conditions')
    selected = [c for c in old['conditions']
                if state['jobs'].get('roles_' + c['id'], {}).get('status') == 'skipped_error']
    if not selected:
        raise ValueError('No failed role conditions to retry')
    target, folder = inside(repo, config_output), inside(repo, study)
    if target.exists() or folder.exists():
        raise ValueError('New config and study directories are required; old evidence is immutable')
    if smoke_condition and smoke_condition not in {c['id'] for c in selected}:
        raise ValueError('Smoke condition must be one of the failed role conditions')
    plan = copy.deepcopy(old)
    plan.update(output=study, evidence_protocol=PROTOCOL, conditions=selected,
                enabled_conditions=[c['id'] for c in selected],
                delivery_skills=['skills/temporal-evidence-delivery-v3/SKILL.md'])
    runner.update(study=study, config_output=config_output,
                  reuse_only=True,
                  plan_config=config_output + '/run.json',
                  background_log_dir='test/temporal_guidance_transport/' + Path(study).name)
    write_json(target / 'run.json', plan)
    write_json(target / 'runner.json', runner)
    write_json(folder / 'activation.json', {
        'revision': PROTOCOL, 'execution_enabled': True, 'user_authorized': True,
        'condition_ids': plan['enabled_conditions'], 'controls_generated_again': False,
        'plan_config': runner['plan_config'], 'source_runner_sha256': digest(prior_runner),
        'source_state_sha256': digest(inside(repo, read_json(prior_runner)['study']) / 'run_status.json'),
        'reuse_numerical_study': runner['reuse_study'], 'new_numerical_candidates': 0,
        'error_policy': runner['module_error_policy'],
        'interpretation': 'Evidence-delivery retry, not a new scientific intervention or matched prompt ablation'})
    prepare(repo, plan)
    if smoke_condition:
        smoke = copy.deepcopy(plan)
        smoke.update(output=study + '/smoke', chains=1,
                     conditions=[c for c in selected if c['id'] == smoke_condition],
                     enabled_conditions=[smoke_condition])
        write_json(target / 'smoke.json', smoke)
        prepare(repo, smoke)
    return {'failed_conditions': plan['enabled_conditions'], 'protocol': PROTOCOL,
            'preparation_API_calls': 0, 'preparation_FLOWR_runs': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-runner', required=True)
    p.add_argument('--config-output', required=True)
    p.add_argument('--study', required=True)
    p.add_argument('--smoke-condition')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(build(repo, args.prior_runner, args.config_output, args.study, args.smoke_condition))
