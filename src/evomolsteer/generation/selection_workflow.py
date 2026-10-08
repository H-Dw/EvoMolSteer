"""Byte-exact incumbent restoration; failed hypotheses never become defaults."""
from pathlib import Path
from ..io import digest,read_json,write_json


def restore_incumbent(repo, campaign, reason, round_number):
    root=Path(repo).resolve();folder=root/'configs/experiments'/campaign
    policy=read_json(folder/'baseline_snapshot.json')
    source=root/policy['program_path']
    if digest(source)!=policy['program_sha256']:raise ValueError('Immutable incumbent changed')
    for role,item in policy['baseline_skills'].items():
        if digest(root/item['path'])!=item['sha256']:raise ValueError('Baseline Skill changed: '+role)
    for path,sha in policy['baseline_algorithm_files'].items():
        if digest(root/path)!=sha:raise ValueError('Baseline algorithm changed: '+path)
    active=folder/'active_program.json';active.write_bytes(source.read_bytes())
    write_json(folder/'active_skills.json',policy['baseline_skills'])
    state={'default':'historical_R26','candidate_enabled':False,'reason':reason,'after_round':round_number,
      'program_sha256':digest(active),'reference_sha256':read_json(source)['reference_sha256'],
      'baseline_skills':policy['baseline_skills'],'baseline_algorithm_files':policy['baseline_algorithm_files']}
    write_json(folder/'active_workflow.json',state)
    return state


def activate_confirmed(repo, campaign, frozen, checks):
    """Activate only an unchanged, previously frozen and confirmed proposal."""
    root=Path(repo).resolve();folder=root/'configs/experiments'/campaign
    # Check every immutable source/Skill before altering the default router.
    restore_incumbent(root,campaign,'Verify baseline before confirmation adoption.',20)
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError('All predeclared confirmation checks must pass')
    if not frozen['screening_admissible'] or not frozen['before_confirmation_labels']:
        raise ValueError('Candidate must be admissible and frozen before labels')
    source=folder/f"round{frozen['selected_round']:02d}.json"
    if digest(source)!=frozen['program_sha256']:
        raise ValueError('Frozen proposal changed')
    program=read_json(source)
    reference=next((p for p in [folder/'selection_reference.json.gz',
        root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz']
        if p.is_file() and digest(p)==frozen['reference_sha256']),None)
    if reference is None or program['reference_sha256']!=digest(reference):
        raise ValueError('Frozen reference unavailable or changed')
    skills=read_json(folder/'baseline_snapshot.json')['baseline_skills']
    module=root/'skills/selection-pressure-path/SKILL.md'
    if program['reward_view']=='endpoint_selection_path':
        skills['experimental_module']={'path':module.relative_to(root).as_posix(),'sha256':digest(module)}
    (folder/'active_program.json').write_bytes(source.read_bytes())
    write_json(folder/'active_skills.json',skills)
    state={'default':'confirmed_selection_path','candidate_enabled':True,
        'selected_round':frozen['selected_round'],'after_round':20,
        'program_sha256':digest(source),'reference_sha256':digest(reference),
        'skills':skills,'confirmation_checks':checks}
    write_json(folder/'active_workflow.json',state)
    return state
