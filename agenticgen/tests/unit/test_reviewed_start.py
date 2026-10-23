"""A normal boundary must not overwrite/relaunch completed or staged versions."""
import importlib.util
from pathlib import Path
import pytest
from agenticgen.adaptive_schedule.io import write, sha

spec = importlib.util.spec_from_file_location('reviewed_start', Path(__file__).parents[2]/'scripts/start_reviewed_version.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def finalized(tmp_path):
    base = tmp_path/'versions/v8'
    write(base/'report.json', {'formal':True})
    write(base/'review/manifest.json', {'sources':'synthetic'})
    write(tmp_path/'FINALIZED_v8.json', {'outcomes':1000,'rounds':5,
        'report_sha256':sha(base/'report.json'), 'review_manifest_sha256':sha(base/'review/manifest.json')})
    return tmp_path


def test_completed_predecessor_accepts_new_boundary(finalized):
    assert module.previous_version(finalized, 'v9') == 'v8'
    assert not (finalized/'versions/v9').exists()


@pytest.mark.parametrize('kind', ['staged', 'finished', 'report_changed', 'missing_final'])
def test_unsafe_boundary_rejected(finalized, kind):
    if kind == 'staged': write(finalized/'versions/v9/review_approval.json', {})
    if kind == 'finished': write(finalized/'FINALIZED_v9.json', {})
    if kind == 'report_changed': write(finalized/'versions/v8/report.json', {'changed':True})
    if kind == 'missing_final': (finalized/'FINALIZED_v8.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        module.previous_version(finalized, 'v9')


@pytest.mark.parametrize('version', ['v8','v31','V9','v09','../v9'])
def test_out_of_scope_versions_rejected(finalized, version):
    with pytest.raises(ValueError): module.previous_version(finalized, version)


@pytest.mark.parametrize('attr', ['tests="0"','tests="1" failures="1"','tests="1" errors="1"','tests="1" skipped="1"'])
def test_remote_tests_must_be_complete(tmp_path, attr):
    p=tmp_path/'tests.xml';p.write_text('<testsuites><testsuite '+attr+'/></testsuites>')
    with pytest.raises(ValueError): module.passing_tests(p)


def test_skills_documents_do_not_schedule_a_test_suite():
    assert module.validation_tests(['agenticgen/docs/plan.zh-CN.md',
        'agenticgen/src/agenticgen/resources/skills/plip-group-analyst/SKILL.md']) == []


def test_deployment_change_gets_only_related_files():
    assert module.validation_tests(['agenticgen/scripts/start_reviewed_version.py',
        'agenticgen/scripts/deploy_reviewed_version.py',
        'agenticgen/tests/unit/test_reviewed_start.py']) == [
            'agenticgen/tests/unit/test_reviewed_deploy.py',
            'agenticgen/tests/unit/test_reviewed_start.py']


def test_other_code_requires_specific_validation_plan():
    with pytest.raises(ValueError, match='scoped remote'):
        module.validation_tests(['agenticgen/src/agenticgen/providers/deepseek_api.py'])


@pytest.mark.parametrize('kind', ['strategy_review_operation','teacher_restart_operation'])
def test_active_protocol_dispatch_preserves_current_gate(tmp_path, monkeypatch, kind):
    prepared = {'master_review_policy':{'mode':module.MODE,'full_review_prohibited':True},
                kind:'operations/accepted', 'previous_strategy_review_operation':'operations/historical'}
    monkeypatch.setattr(module,'prepared_path',lambda root:tmp_path/'PREPARED.json')
    write(tmp_path/'PREPARED.json',prepared)
    called=[]
    def verify(root,version):
        called.append((root,version));return {'phase':'accepted'}
    def wrong(*args):raise AssertionError('Inactive historical gate must not be called')
    monkeypatch.setattr(module,'verify_review_policy',verify if kind=='strategy_review_operation' else wrong)
    monkeypatch.setattr(module,'verify_teacher_restart',verify if kind=='teacher_restart_operation' else wrong)
    assert module.verify_active_protocol(tmp_path,'v11')=={'kind':kind,'operation':'operations/accepted'}
    assert called==[(tmp_path,'v11')]
    assert read_prepared(tmp_path)==prepared
    assert not (tmp_path/'versions').exists()


def read_prepared(root):
    import json
    return json.loads((root/'PREPARED.json').read_text())


@pytest.mark.parametrize('case',['missing','ambiguous','wrong_mode','full_review','failed_gate','missing_transaction'])
def test_active_protocol_cannot_bypass_readiness(tmp_path, monkeypatch, case):
    prepared={'master_review_policy':{'mode':module.MODE,'full_review_prohibited':True},
              'teacher_restart_operation':'operations/accepted'}
    if case=='missing':prepared.pop('teacher_restart_operation')
    if case=='ambiguous':prepared['strategy_review_operation']='operations/stale'
    if case=='wrong_mode':prepared['master_review_policy']['mode']='full_review'
    if case=='full_review':prepared['master_review_policy']['full_review_prohibited']=False
    monkeypatch.setattr(module,'prepared_path',lambda root:tmp_path/'PREPARED.json')
    write(tmp_path/'PREPARED.json',prepared)
    def verify(*args):
        if case=='failed_gate':raise ValueError('Existing native/hash verification failed')
        return None if case=='missing_transaction' else {'phase':'accepted'}
    monkeypatch.setattr(module,'verify_teacher_restart',verify)
    with pytest.raises(ValueError):module.verify_active_protocol(tmp_path,'v11')
    assert read_prepared(tmp_path)==prepared
    assert not (tmp_path/'versions').exists()
