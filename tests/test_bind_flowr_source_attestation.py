import importlib.util
from pathlib import Path

import pytest

from evomolsteer.io import digest, read_json, write_json

_source = Path(__file__).resolve().parents[1] / 'scripts/bind_flowr_source_attestation.py'
_spec = importlib.util.spec_from_file_location('flowr_bind_script', _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
bind = _module.bind


def inputs(tmp_path):
    root = tmp_path / 'generated'
    run = root / 'results' / 'trial'
    run.mkdir(parents=True)
    config = {'experiment': {'campaign': 'trial', 'seed': 42}, 'extension': {'code_commit': 'fixture'}}
    write_json(run / 'config.json', config)
    write_json(run / 'COMPLETE.json', {'status': 'complete'})
    before = tmp_path / 'before.json'
    after = tmp_path / 'after.json'
    a = {'schema_version': 'flowr-upstream-source-attestation-1.0', 'input_dataset': '/upstream',
         'files': {'model.py': {'sha256': 'fixture', 'bytes': 1}}}
    write_json(before, a)
    write_json(after, {**a, 'source_unchanged': True, 'reference_sha256': digest(before)})
    return root, run, before, after, config


def test_only_metadata_added_and_cannot_rebind(tmp_path):
    root, run, before, after, original = inputs(tmp_path)
    prior_hash = digest(run / 'config.json')
    assert bind(root, 'trial', before, after)['bound']
    result = read_json(run / 'config.json')
    binding = result.pop('upstream_source_attestation')
    assert result == original
    assert binding['generation_config_sha256_before_binding'] == prior_hash
    assert binding['before_record_sha256'] == digest(before)
    with pytest.raises(FileExistsError):
        bind(root, 'trial', before, after)


def test_mismatch_does_not_write_config(tmp_path):
    root, run, before, after, _ = inputs(tmp_path)
    original_bytes = (run / 'config.json').read_bytes()
    data = read_json(after)
    data['files']['model.py']['sha256'] = 'changed'
    write_json(after, data)
    with pytest.raises(ValueError):
        bind(root, 'trial', before, after)
    assert (run / 'config.json').read_bytes() == original_bytes


@pytest.mark.parametrize('campaign', ['../trial', '/trial', 'trial\\other', '.', '..'])
def test_campaign_path_rejected(tmp_path, campaign):
    root, _, before, after, _ = inputs(tmp_path)
    with pytest.raises(ValueError):
        bind(root, campaign, before, after)
