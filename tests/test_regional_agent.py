import copy
from pathlib import Path
import pytest
from evomolsteer.io import read_json,write_json
from evomolsteer.continuous.regional_design import audit_regional_behavior

def sources():
    root=Path(__file__).resolve().parents[1];e=root/'docs/experiments/ck2_affinity_geometry30_20261007';f=e/'regional_skill_test'
    return root,e,f

def test_literal_regional_agent_responds_to_numeric_contradictions():
    root,e,f=sources()
    result=audit_regional_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-regional-pointcloud.md',f/'input.json',f/'prompt.txt',f/'response.json',e/'regional_mining_v1/region_prior.json')
    assert result['selected_round']==7 and result['selected_region']==0
    assert result['node_sign_contradiction_verified'] and result['information_limitations_verified']

@pytest.mark.parametrize('change',['constant_attraction','causal_overlap','information_claim','ignore_regression','incorrect_dose','identity_gradient'])
def test_hash_correct_but_behavior_wrong_response_is_rejected(tmp_path,change):
    root,e,f=sources();bad=copy.deepcopy(read_json(f/'response.json'))
    if change=='constant_attraction':bad['regional_case']['constant_signed_attraction_allowed']=True
    elif change=='causal_overlap':bad['overlap_independent_causality']=True
    elif change=='information_claim':bad['information_case']['excess_RMS_proves_affinity']=True
    elif change=='ignore_regression':bad['regression_case']['intervention']='dose_escalation'
    elif change=='incorrect_dose':bad['reward_design']['native_rms_ratio']=.51
    else:bad['reward_design']['derivative_path']='identity'
    path=tmp_path/'response.json';write_json(path,bad)
    with pytest.raises(ValueError):audit_regional_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-regional-pointcloud.md',f/'input.json',f/'prompt.txt',path,e/'regional_mining_v1/region_prior.json')
