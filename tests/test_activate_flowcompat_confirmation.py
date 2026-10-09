"""Confirmation routing must reject stale/failed evidence without changing defaults."""
import copy
from pathlib import Path
import sys

import pytest

from evomolsteer.io import digest, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from activate_flowcompat_confirmation import validated_plan


@pytest.fixture
def routing(tmp_path):
    root = tmp_path
    cfg = root / 'configs/experiments/example'
    evidence = root / 'docs/example'
    cfg.mkdir(parents=True)
    evidence.mkdir(parents=True)
    source = cfg / 'round11.json'
    reference = cfg / 'branch_reference.json.gz'
    reference.write_bytes(b'opaque immutable fixture reference')
    write_json(source, {'reference_sha256': digest(reference), 'generation_interface': 'flowcompat_v2',
        'flowcompat_provenance': {'workflow_version': 'flowcompat-supplemental-agent-2.0'}})
    frozen = {'selected_round': 11, 'screening_admissible': True, 'before_confirmation_labels': True,
        'program_sha256': digest(source), 'reference_sha256': digest(reference)}
    write_json(evidence / 'frozen_validation.json', frozen)
    baseline = root / 'baseline.json'
    baseline.write_bytes(b'protected original')
    algorithm = root / 'algorithm.py'
    algorithm.write_bytes(b'protected algorithm')
    skills = {}
    for name in ['analyst', 'designer', 'flow-compatibility', 'flow-compatibility-supplemental']:
        p = root / 'skills' / name / 'SKILL.md'
        p.parent.mkdir(parents=True)
        p.write_text(name)
        if name in ['analyst', 'designer']:
            skills[name] = {'path': p.relative_to(root).as_posix(), 'sha256': digest(p)}
    write_json(cfg / 'baseline_snapshot.json', {'program_path': 'baseline.json',
        'program_sha256': digest(baseline), 'baseline_algorithm_files': {'algorithm.py': digest(algorithm)},
        'baseline_skills': skills})
    write_json(cfg / 'campaign.json', {'maximum_rounds': 30, 'rounds_completed': 30,
        'rounds': [{'round': i, 'status': 'complete'} for i in range(1, 31)]})
    confirmation = {'complete': True, 'adopt_candidate': True,
        'checks': {key: True for key in ['confirmation_integrity', 'positive_independent_effect', 'positive_vs_native',
            'validity', 'pose_quality', 'strain_median', 'strain_p90', 'screened_before_confirmation', 'implementation']},
        'frozen_candidate': frozen}
    integrity = {'integrity_passed': True, 'eligible_for_efficacy_decision': True}
    write_json(evidence / 'confirmation.json', confirmation)
    write_json(evidence / 'confirmation_integrity.json', integrity)
    (cfg / 'active_program.json').write_bytes(b'previous default')
    return root, cfg, evidence, confirmation, integrity


def test_plan_retains_full_reference_and_literal_profiles(routing):
    plan = validated_plan(*routing)
    assert plan['reference_path'].endswith('branch_reference.json.gz')
    assert plan['selected_round'] == 11
    assert set(plan['skills']) == {'analyst', 'designer', 'flow-compatibility', 'flow-compatibility-supplemental'}
    assert (routing[1] / 'active_program.json').read_bytes() == b'previous default'


@pytest.mark.parametrize('change', ['efficacy', 'integrity', 'ineligible', 'candidate', 'baseline', 'skill', 'incomplete', 'missing_check'])
def test_rejection_leaves_default_unchanged(routing, change):
    root, cfg, evidence, confirmation, integrity = routing
    confirmation, integrity = copy.deepcopy(confirmation), copy.deepcopy(integrity)
    if change == 'efficacy':
        confirmation['adopt_candidate'] = False
    elif change == 'integrity':
        integrity['integrity_passed'] = False
    elif change == 'ineligible':
        integrity['eligible_for_efficacy_decision'] = False
    elif change == 'candidate':
        (cfg / 'round11.json').write_text('{}')
    elif change == 'baseline':
        (root / 'baseline.json').write_bytes(b'changed')
    elif change == 'skill':
        (root / 'skills/designer/SKILL.md').write_text('changed')
    elif change == 'incomplete':
        write_json(cfg / 'campaign.json', {'maximum_rounds': 30, 'rounds_completed': 29, 'rounds': []})
    elif change == 'missing_check':
        confirmation['checks'].pop('positive_independent_effect')
    with pytest.raises(ValueError):
        validated_plan(root, cfg, evidence, confirmation, integrity)
    assert (cfg / 'active_program.json').read_bytes() == b'previous default'
