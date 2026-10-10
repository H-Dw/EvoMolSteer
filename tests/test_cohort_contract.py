import json
import pytest
from evomolsteer.io import digest
from evomolsteer.generation.cohort_contract import artifact_path, validate_artifacts


def test_portable_windows_paths_and_embedded_reference(tmp_path):
    data = tmp_path / 'configs'; data.mkdir()
    reference = data / 'reference.json.gz'; reference.write_bytes(b'correct')
    program = data / 'program.json'
    program.write_text(json.dumps({'reference_sha256': digest(reference)}))
    spec = {'program': 'configs\\program.json', 'reference': 'configs\\reference.json.gz',
            'program_sha256': digest(program), 'reference_sha256': digest(reference)}
    assert validate_artifacts(tmp_path, spec) == (program.resolve(), reference.resolve())
    other = data / 'wrong.json.gz'; other.write_bytes(b'wrong')
    spec.update(reference='configs/wrong.json.gz', reference_sha256=digest(other))
    with pytest.raises(ValueError, match='reward program reference'):
        validate_artifacts(tmp_path, spec)


@pytest.mark.parametrize('value', ['../outside.json', 'C:/outside.json', '/tmp/outside.json'])
def test_repository_escape_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='inside'):
        artifact_path(tmp_path, value)
