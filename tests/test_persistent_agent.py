import copy
from pathlib import Path
import pytest
from evomolsteer.io import read_json,digest,write_json
from evomolsteer.continuous.persistent_design import audit_persistent_behavior,compile_persistent
from evomolsteer.generation.window_reference import load_reference


def sources():
    root=Path(__file__).resolve().parents[1];e=root/'docs/experiments/ck2_affinity_geometry30_20261007';folder=e/'persistent_skill_test'
    return root,e,folder,e/'endpoint_mining/sparse_coordinate_prior_v2.json',e/'kinematics_mining_v3/kinematics_prior.json'


def test_literal_new_skill_response_and_registered_compilation():
    root,e,f,prior,h=sources()
    audit=audit_persistent_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-persistent-coordinate.md',f/'input.json',f/'prompt.txt',f/'response.json',prior,h)
    reference=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    p=compile_persistent(read_json(f/'response.json'),{},load_reference(reference),digest(reference),
        read_json(prior),digest(prior),read_json(h),digest(h),h.relative_to(root).as_posix())
    assert audit['selected_round']==7 and audit['trajectory_intervention_verified']
    assert p['reward_view']=='endpoint_supported_attractor' and p['history_strength']==0
    assert p['native_rms_ratio']==.3 and p['window']==[0,.5] and p['additional_per_step_affinity_calls']==0


def test_unchanged_diagnosis_or_fabricated_history_is_rejected(tmp_path):
    root,e,f,prior,h=sources();response=read_json(f/'response.json');path=tmp_path/'response.json'
    for change in ('dose_only','unsupported_history'):
        bad=copy.deepcopy(response)
        if change=='dose_only':bad['trajectory_case']['intervention']='dose_escalation'
        else:bad['history_feature_names'].append('shape_xx')
        write_json(path,bad)
        with pytest.raises(ValueError):audit_persistent_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-persistent-coordinate.md',f/'input.json',f/'prompt.txt',path,prior,h)
