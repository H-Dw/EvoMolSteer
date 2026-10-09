"""Activate an unchanged frozen proposal only after independent confirmation.

This routing operation does not alter a reward, model, Skill or frozen report.
Default dry-run; rejection occurs before any default-file mutation.
"""
import argparse
import copy
import os
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from report_flowcompat_campaign import report


def validated_plan(root, configuration, evidence, confirmation, integrity):
    root, configuration, evidence = map(lambda p: Path(p).resolve(), (root, configuration, evidence))
    if not configuration.is_relative_to(root / 'configs/experiments') or configuration == root / 'configs/experiments':
        raise ValueError('Explicit experiment configuration dataset required')
    if not confirmation.get('complete') or not confirmation.get('adopt_candidate'):
        raise ValueError('Candidate has not passed completed independent confirmation')
    if not integrity.get('integrity_passed') or not integrity.get('eligible_for_efficacy_decision'):
        raise ValueError('Independent integrity gate must pass')
    checks = confirmation.get('checks', {})
    expected_checks = {'confirmation_integrity', 'positive_independent_effect', 'positive_vs_native',
        'validity', 'pose_quality', 'strain_median', 'strain_p90', 'screened_before_confirmation', 'implementation'}
    if set(checks) != expected_checks or not all(value is True for value in checks.values()):
        raise ValueError('Every prespecified confirmation check must pass')
    state = read_json(configuration / 'campaign.json')
    if state['maximum_rounds'] != 30 or state['rounds_completed'] != 30 or [r['round'] for r in state['rounds']] != list(range(1, 31)) or any(r['status'] != 'complete' for r in state['rounds']):
        raise ValueError('Complete sequential campaign required')
    frozen = read_json(evidence / 'frozen_validation.json')
    if frozen != confirmation.get('frozen_candidate') or not frozen['screening_admissible'] or not frozen['before_confirmation_labels']:
        raise ValueError('Unchanged pre-label frozen candidate required')
    source = configuration / f'round{frozen["selected_round"]:02d}.json'
    if digest(source) != frozen['program_sha256']:
        raise ValueError('Frozen candidate changed')
    program = read_json(source)
    references = list(configuration.glob('*.json.gz')) + [root / 'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']
    reference = next((p for p in references if p.is_file() and digest(p) == frozen['reference_sha256']), None)
    if reference is None or program['reference_sha256'] != frozen['reference_sha256']:
        raise ValueError('Bound frozen reference missing')
    policy = read_json(configuration / 'baseline_snapshot.json')
    if digest(root / policy['program_path']) != policy['program_sha256']:
        raise ValueError('Protected baseline program changed')
    for path, sha in policy['baseline_algorithm_files'].items():
        if digest(root / path) != sha:
            raise ValueError('Protected baseline source changed: ' + path)
    skills = copy.deepcopy(policy['baseline_skills'])
    version = program['flowcompat_provenance']['workflow_version']
    profiles = {
        'flowcompat-supplemental-agent-2.0': ['flow-compatibility', 'flow-compatibility-supplemental'],
        'flowcompat-regional-agent-1.0': ['flow-compatibility', 'flow-compatibility-supplemental', 'flow-compatibility-regional'],
    }
    if version not in profiles:
        raise ValueError('Unregistered confirmed workflow')
    for name in profiles[version]:
        path = root / 'skills' / name / 'SKILL.md'
        skills[name] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}
    for item in policy['baseline_skills'].values():
        if digest(root / item['path']) != item['sha256']:
            raise ValueError('Protected baseline role Skill changed')
    return {'default': 'confirmed_flowcompatibility', 'candidate_enabled': True,
        'selected_round': frozen['selected_round'], 'after_round': state['rounds_completed'],
        'source_program': source.relative_to(root).as_posix(), 'program_sha256': digest(source),
        'reference_path': reference.relative_to(root).as_posix(), 'reference_sha256': digest(reference),
        'generation_interface': program['generation_interface'], 'workflow_version': version,
        'skills': skills, 'confirmation_checks': checks,
        'confirmation_sha256': digest(evidence / 'confirmation.json'),
        'integrity_sha256': digest(evidence / 'confirmation_integrity.json'),
        'protected_incumbent_sha256': policy['program_sha256']}


def activate(repo, input_dataset, configuration_dataset, output, apply=False):
    root = Path(repo).resolve()
    evidence, configuration = Path(input_dataset).resolve(), Path(configuration_dataset).resolve()
    # Re-evaluate the predeclared report and its independent checks; no fitting.
    standard_evidence = root / 'docs/experiments/flowcompat30_20261009'
    standard_configuration = root / 'configs/experiments/flowcompat30_v1'
    if evidence != standard_evidence or configuration != standard_configuration:
        raise ValueError('This activation profile requires its registered campaign datasets')
    destination = Path(output).resolve()
    if destination.is_relative_to(configuration):
        raise ValueError('Activation receipt cannot overwrite configuration')
    if destination.is_relative_to(evidence) and destination != evidence / 'activation.json':
        raise ValueError('Use the designated activation receipt within the evidence dataset')
    if destination.exists() and read_json(destination).get('schema_version') != 'flowcompat-confirmation-activation-1.0':
        raise ValueError('Activation receipt cannot overwrite an unrelated artifact')
    confirmation = report(root)
    integrity = read_json(evidence / 'confirmation_integrity.json')
    plan = validated_plan(root, configuration, evidence, confirmation, integrity)
    if apply:
        source = root / plan['source_program']
        staged = configuration / '.active_program.confirmation.tmp'
        staged.write_bytes(source.read_bytes())
        os.replace(staged, configuration / 'active_program.json')
        write_json(configuration / 'active_skills.json', plan['skills'])
        write_json(configuration / 'active_workflow.json', plan)
    write_json(destination, {'schema_version': 'flowcompat-confirmation-activation-1.0',
        'applied': bool(apply), 'plan': plan, 'frozen_files_modified': False,
        'script_sha256': digest(__file__)})
    return {'applied': bool(apply), 'selected_round': plan['selected_round']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--input-dataset', required=True)
    p.add_argument('--configuration-dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    print(activate(a.repo, a.input_dataset, a.configuration_dataset, a.output, a.apply))
