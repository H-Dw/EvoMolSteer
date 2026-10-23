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
