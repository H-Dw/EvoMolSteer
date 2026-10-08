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
