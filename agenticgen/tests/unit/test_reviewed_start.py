"""A normal boundary must not overwrite/relaunch completed or staged versions."""
import importlib.util
from pathlib import Path
import pytest
from agenticgen.adaptive_schedule.io import write, sha, read, value_sha
from agenticgen.adaptive_schedule.search_policy import policy

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
            'agenticgen/tests/unit/test_control_gates.py',
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


@pytest.fixture
def rollback_boundary(tmp_path, monkeypatch):
    root=tmp_path/'results'; repo=tmp_path/'repo'; review='agenticgen/docs/experiments/v12_strategy_review/'
    old=root/'versions/v10'; restored=policy()
    binding={'skill_hashes':{role:'source-'+role for role in module.SKILLS},
             'source_fingerprint':{'native':'preserved'}}
    write(old/'binding.json',binding);write(old/'search_policy.json',restored)
    document=review+'rollback_compatibility.zh-CN.md'
    (repo/document).parent.mkdir(parents=True);(repo/document).write_text('Retain accepted native and interface compatibility.')
    plan={'previous_version':'v12','next_version':'v13','restored_version':'v10',
          'restored_binding_sha256':sha(old/'binding.json'),
          'source_binding_sha256':value_sha(binding['source_fingerprint']),
          'restored_policy':restored,'code_compatibility_document':document,
          'code_compatibility_sha256':sha(repo/document)}
    write(repo/review/'rollback_plan.json',plan)
    monkeypatch.setattr(module,'repository',lambda:repo)
    monkeypatch.setattr(module,'version_metrics',lambda root:[])
    monkeypatch.setattr(module,'rollback_advice',lambda rows:{'required':True,'restore_version':'v10'})
    monkeypatch.setattr(module,'load_version_policy',lambda root,version:read(root/'versions'/version/'search_policy.json'))
    calls=[]
    def committed(path,digest,revision):
        calls.append((path,digest,revision));return repo/path
    monkeypatch.setattr(module,'committed_file',committed)
    return root,repo,review,plan,calls


def test_required_rollback_binds_history_skills_and_document_before_staging(rollback_boundary):
    root,repo,review,plan,calls=rollback_boundary
    restored,record=module.reviewed_control_state(root,'v13',review,'fixed-source')
    assert restored==plan['restored_policy']
    for key in ('restored_version','restored_binding_sha256','source_binding_sha256','code_compatibility_sha256'):
        assert record[key]==plan[key]
    assert {path for path,_,_ in calls}=={review+'rollback_plan.json',review+'rollback_compatibility.zh-CN.md',*module.SKILLS.values()}
    assert all(rev=='fixed-source' for _,_,rev in calls)
    assert not (root/'versions/v13').exists()


@pytest.mark.parametrize('field',['previous_version','next_version','restored_version','restored_binding_sha256',
                                 'source_binding_sha256','restored_policy','code_compatibility_document','code_compatibility_sha256'])
def test_rollback_plan_must_match_exact_history(rollback_boundary,field):
    root,repo,review,plan,_=rollback_boundary
    plan[field]='wrong';write(repo/review/'rollback_plan.json',plan)
    with pytest.raises(ValueError):module.reviewed_control_state(root,'v13',review,'fixed-source')
    assert not (root/'versions/v13').exists()


def test_restored_skills_cannot_be_substituted(rollback_boundary,monkeypatch):
    root,repo,review,_,_=rollback_boundary
    def committed(path,digest,revision):
        if path in module.SKILLS.values():raise ValueError('Skill source differs')
        return repo/path
    monkeypatch.setattr(module,'committed_file',committed)
    with pytest.raises(ValueError,match='Skill source'):module.reviewed_control_state(root,'v13',review,'fixed-source')
    assert not (root/'versions/v13').exists()


def test_rollback_plan_cannot_reset_nonregressing_history(rollback_boundary,monkeypatch):
    root,_,review,_,_=rollback_boundary
    monkeypatch.setattr(module,'rollback_advice',lambda rows:{'required':False})
    with pytest.raises(ValueError,match='without two'):module.reviewed_control_state(root,'v13',review,'fixed-source')
    assert not (root/'versions/v13').exists()


def test_normal_boundary_keeps_policy_and_has_no_rollback(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'repository',lambda:tmp_path)
    monkeypatch.setattr(module,'version_metrics',lambda root:[])
    monkeypatch.setattr(module,'rollback_advice',lambda rows:{'required':False})
    monkeypatch.setattr(module,'load_version_policy',lambda root,version:policy())
    assert module.reviewed_control_state(tmp_path,'v14','review/','source')==(policy(),None)


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
