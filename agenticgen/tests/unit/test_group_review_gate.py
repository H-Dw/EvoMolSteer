"""Synthetic review fixtures exercise the real committed-file and coverage gate."""
import subprocess

import pytest

from agenticgen.adaptive_schedule import group_review as gate
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.adaptive_schedule.io import read, write, sha, value_sha


@pytest.fixture
def reviewed_transition(tmp_path, monkeypatch):
    repo, root = tmp_path / 'repo', tmp_path / 'results'
    repo.mkdir(); root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.STDOUT).decode().strip()
    git('init'); git('config', 'core.autocrlf', 'false')
    git('config', 'user.name', 'Review gate test'); git('config', 'user.email', 'test@example.invalid')
    for relative in gate.SKILLS.values():
        path = repo / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic test skill, not an experimental approval.\n', encoding='utf-8')
    c=configuration(); c['schema']='legacy-review-fixture'; c['versions']=['v1','v2']; c.pop('previous_protocol_root',None)
    write(root / 'config.json', c)
    sessions, coverage = [], []
    for teacher in ['teacher_a', 'teacher_b']:
        for round_no, role in [(0, 'bootstrap')] + [(r, role) for r in range(1, 6) for role in ['G1', 'G2', 'G3', 'G4', 'synthesis']]:
            identifier = f'v1/{teacher}/r{round_no:02d}/{role}'
            relative = f'versions/{identifier}/actual_request.json'
            write(root / relative, {'synthetic_session': identifier})
            files = {relative: sha(root / relative)}
            session = {'id': identifier, 'files_sha256': files, 'evidence_sha256': value_sha(files)}
            sessions.append(session)
            coverage.append({'id': identifier, 'evidence_sha256': session['evidence_sha256'],
                'assessment': 'Synthetic fixture verifies provenance rather than claiming scientific review.',
                'decision_update': 'Synthetic fixture has no experimental interpretation or deployment permission.',
                'counterexample_or_limitation': 'Synthetic fixture cannot stand in for a real GPT review of experiments.',
                'evidence_files': [relative]})
    shards = []
    for i in range(0, len(sessions), 5):
        relative = f'versions/v1/review/shard_{i//5:03d}.json'
        write(root / relative, {'sessions': sessions[i:i+5]})
        shards.append({'path': relative, 'sha256': sha(root / relative)})
    manifest_path = root / 'versions/v1/review/manifest.json'
    write(manifest_path, {'shards': shards, 'expected_sessions': 52})
    write(root / 'FINALIZED_v1.json', {'review_manifest_sha256': sha(manifest_path)})
    document = repo / 'review.md'
    document.write_text('Synthetic test report only.\n', encoding='utf-8')
    write(repo / 'coverage.json', {'sessions': coverage})
    git('add', '.'); git('commit', '-m', 'Synthetic review gate fixture')
    revision = git('rev-parse', 'HEAD')
    monkeypatch.setattr(gate, 'repository', lambda: repo)
    monkeypatch.setattr(gate, 'commit', lambda: revision)
    approval = {'previous_version': 'v1', 'next_version': 'v2',
        'finalized_sha256': sha(root / 'FINALIZED_v1.json'), 'manifest_sha256': sha(manifest_path),
        'review_commit': revision, 'skill_sha256': {k: sha(repo / p) for k, p in gate.SKILLS.items()},
        'review_document': 'review.md', 'review_document_sha256': sha(document),
        'coverage': [{'path': 'coverage.json', 'sha256': sha(repo / 'coverage.json')}]}
    write(root / 'versions/v2/review_approval.json', approval)
    return repo, root, sessions, approval


def test_complete_fifty_two_session_transition_passes(reviewed_transition):
    _, root, _, _ = reviewed_transition
    assert gate.review_check(root, 'v2')['sessions'] == 52


def test_empty_approval_coverage_cannot_start_next_version(reviewed_transition):
    _, root, _, approval = reviewed_transition
    approval['coverage'] = []
    write(root / 'versions/v2/review_approval.json', approval)
    with pytest.raises(ValueError, match='coverage incomplete'):
        gate.review_check(root, 'v2')


def test_changed_actual_prompt_invalidates_approval(reviewed_transition):
    _, root, sessions, _ = reviewed_transition
    request = next(iter(sessions[-1]['files_sha256']))
    write(root / request, {'changed': True})
    with pytest.raises(ValueError, match='prompt/tool evidence changed'):
        gate.review_check(root, 'v2')


def test_matching_local_hash_is_insufficient_without_commit(reviewed_transition):
    repo, root, _, approval = reviewed_transition
    path = repo / 'coverage.json'
    data = read(path); data['sessions'].pop(); write(path, data)
    approval['coverage'][0]['sha256'] = sha(path)
    write(root / 'versions/v2/review_approval.json', approval)
    with pytest.raises(ValueError, match='not committed'):
        gate.review_check(root, 'v2')


def test_twentieth_version_is_the_last_authorized_transition(reviewed_transition):
    _, root, _, _ = reviewed_transition
    with pytest.raises(ValueError, match='Invalid reviewed-version transition'):
        gate.review_check(root, 'v21')


def test_new_protocol_requires_binding_new_preparation_and_acceptance(reviewed_transition):
    _,old,_,approval=reviewed_transition
    new=old.parent/'new_protocol';new.mkdir()
    config=configuration();config.update(schema='legacy-review-fixture',versions=['v2','v3'],previous_protocol_root=str(old))
    write(new/'config.json',config)
    write(new/'PREPARED.json',{'new_protocol':True})
    write(new/'ACCEPTANCE_COMPLETE.json',{'new_native_checks':True})
    write(new/'versions/v2/review_approval.json',approval)
    with pytest.raises(ValueError,match='bind actual'):
        gate.review_check(new,'v2')
    approval.update(previous_protocol_root=str(old),new_prepared_sha256=sha(new/'PREPARED.json'),
        new_acceptance_sha256=sha(new/'ACCEPTANCE_COMPLETE.json'),objective_priority=config['agent_objective']['priority'])
    write(new/'versions/v2/review_approval.json',approval)
    assert gate.review_check(new,'v2')['sessions']==52
    write(new/'ACCEPTANCE_COMPLETE.json',{'changed':True})
    with pytest.raises(ValueError,match='bind actual'):
        gate.review_check(new,'v2')
