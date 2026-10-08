import importlib.util
from pathlib import Path

import pytest

_source = Path(__file__).resolve().parents[1] / 'scripts/attest_flowr_sources.py'
_spec = importlib.util.spec_from_file_location('flowr_attest_script', _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
attest = _module.attest


def inputs(tmp_path):
    root = tmp_path / 'upstream'
    root.mkdir()
    (root / 'model.py').write_bytes(b'implementation\n')
    return root, tmp_path / 'before.json', tmp_path / 'after.json'


def test_matching_before_after(tmp_path):
    root, before, after = inputs(tmp_path)
    a = attest(root, before, ['model.py'])
    b = attest(root, after, ['model.py'], before)
    assert a['files'] == b['files']
    assert b['source_unchanged'] is True
    assert b['reference_sha256']


def test_changed_source_rejects_and_preserves_previous(tmp_path):
    root, before, after = inputs(tmp_path)
    attest(root, before, ['model.py'])
    prior = before.read_bytes()
    (root / 'model.py').write_bytes(b'changed\n')
    with pytest.raises(ValueError, match='changed during'):
        attest(root, after, ['model.py'], before)
    assert before.read_bytes() == prior
    assert not after.exists()


def test_cannot_overwrite(tmp_path):
    root, before, _ = inputs(tmp_path)
    attest(root, before, ['model.py'])
    with pytest.raises(FileExistsError):
        attest(root, before, ['model.py'])


def test_experiment_side_record_inside_program_root(tmp_path):
    root, _, _ = inputs(tmp_path)
    output = root / 'experiments' / 'run01' / 'source.before.json'
    record = attest(root, output, ['model.py'])
    assert output.exists()
    assert record['files']['model.py']['bytes'] == len(b'implementation\n')


@pytest.mark.parametrize('names', [[], ['model.py', 'model.py'], ['../outside.py']])
def test_source_scope_rejected(tmp_path, names):
    root, before, _ = inputs(tmp_path)
    (tmp_path / 'outside.py').write_bytes(b'outside\n')
    with pytest.raises(ValueError):
        attest(root, before, names)
