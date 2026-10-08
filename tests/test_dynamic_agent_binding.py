import json,sys
from pathlib import Path
import pytest
from evomolsteer.io import digest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_dynamic_agent_inputs import verified_response


def test_actual_input_and_instruction_hashes_are_checked(tmp_path):
    (tmp_path/'Designer.input.json').write_text('{"evidence":1}')
    (tmp_path/'Designer.instructions.md').write_text('Use measured evidence.')
    binding={'input_sha256':digest(tmp_path/'Designer.input.json'),'instruction_sha256':digest(tmp_path/'Designer.instructions.md')}
    (tmp_path/'Designer.request.json').write_text(json.dumps(binding))
    (tmp_path/'Designer.response.json').write_text(json.dumps({**binding,'role':'Designer'}))
    assert verified_response(tmp_path,'Designer')['role']=='Designer'
    (tmp_path/'Designer.input.json').write_text('{"evidence":2}')
    with pytest.raises(ValueError):verified_response(tmp_path,'Designer')


def test_matching_request_and_response_do_not_allow_changed_instructions(tmp_path):
    for name,body in [('Analyst.input.json','{}'),('Analyst.instructions.md','original')]:
        (tmp_path/name).write_text(body)
    binding={'input_sha256':digest(tmp_path/'Analyst.input.json'),'instruction_sha256':digest(tmp_path/'Analyst.instructions.md')}
    (tmp_path/'Analyst.request.json').write_text(json.dumps(binding))
    (tmp_path/'Analyst.response.json').write_text(json.dumps({**binding,'role':'Analyst'}))
    (tmp_path/'Analyst.instructions.md').write_text('altered')
    with pytest.raises(ValueError):verified_response(tmp_path,'Analyst')
