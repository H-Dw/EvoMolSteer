from pathlib import Path
import copy
import pytest
from evomolsteer.io import read_json,digest
from evomolsteer.continuous.sparse_design import audit_sparse_behavior,compile_sparse
from evomolsteer.generation.window_reference import load_reference


def test_literal_agent_changed_design_in_response_to_node_counterevidence():
    root=Path(__file__).resolve().parents[1];e=root/'docs/experiments/ck2_affinity_geometry30_20261007'
    folder=e/'sparse_skill_test_v2';prior_path=e/'endpoint_mining/sparse_coordinate_prior_v2.json'
    skill=root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-sparse-coordinate.md'
    result=audit_sparse_behavior(skill,folder/'input.json',folder/'prompt.txt',folder/'response.json',prior_path)
    assert result['sparse_response_verified'] and result['direction_mode']=='linear_node_effect' and result['recorded_function_count']==24
    response=read_json(folder/'response.json');reference=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    prior=read_json(prior_path)
    program=compile_sparse(response,read_json(e/'endpoint_skill_test/compiled_design.json'),load_reference(reference),digest(reference),prior,digest(prior_path))
    assert program['native_rms_ratio']==.3 and program['derivative_path']=='flowr_endpoint_vjp'
    assert program['geometry_direction_functions']==prior['node_effect_functions']
    wrong=copy.deepcopy(response);wrong['reward_design']['direction_mode']='legendre_effect'
    wrong['reward_design']['geometry_direction_functions']=prior['direction_functions']
    with pytest.raises(ValueError,match='contradicts'):audit_sparse_behavior(skill,folder/'input.json',folder/'prompt.txt',wrong,prior_path)
    wrong=copy.deepcopy(response);wrong['reward_design']['reference_sha256']='wrong'
    with pytest.raises(ValueError,match='hash mismatch'):audit_sparse_behavior(skill,folder/'input.json',folder/'prompt.txt',wrong,prior_path)
