"""A study's explicit model must survive fresh and nested role requests."""
import json
import pytest
from evomolsteer.continuous.dependency_ablation import export, validate


@pytest.fixture
def repeat(tmp_path):
    (tmp_path / 'protocol.json').write_text(json.dumps({'agent_model': 'gpt-6-luna'}))
    folder = tmp_path / 'condition' / 'replicate_1'
    folder.mkdir(parents=True)
    (folder / 'evidence.json').write_text(json.dumps({
        'condition': 'fixture', 'formula_registry': {}, 'allowed_updates': {}, 'evidence': []}))
    (folder / 'Analyst.instructions.md').write_text('Use only supplied evidence.')
    return folder


def test_nested_role_inherits_luna_and_rejects_conflicting_model(repeat):
    request = export(repeat, 'Analyst')
    assert request['requested_model'] == 'gpt-6-luna'
    with pytest.raises(ValueError, match='conflicts with the study protocol'):
        export(repeat, 'Analyst', requested_model='gpt-6-sol')


def test_unpinned_response_is_rejected(repeat):
    request = export(repeat, 'Analyst')
    request.pop('requested_model')
    (repeat / 'Analyst.request.json').write_text(json.dumps(request))
    (repeat / 'Analyst.response.json').write_text('{}')
    with pytest.raises(ValueError, match='does not pin the study model'):
        validate(repeat, 'Analyst')
