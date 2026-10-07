"""Real Agent decisions, including rejection of hash-correct numeric conflicts."""
import copy
from pathlib import Path
import pytest
from evomolsteer.io import read_json, write_json
from evomolsteer.continuous.terminal_design import audit_terminal_behavior


def sources():
    root = Path(__file__).resolve().parents[1]
    e = root/'docs/experiments/ck2_affinity_geometry30_20261007'
    return root, e/'terminal_skill_test_v4', e/'terminal_lineage_v4/terminal_reference.json.gz'


def test_actual_agent_obeys_terminal_credit_missingness_and_capacity():
    root, folder, ref = sources()
    result = audit_terminal_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-terminal-lineage.md', folder/'input.json',
        folder/'prompt.txt', folder/'response.json', ref)
    assert result['selected_round'] == 7 and result['terminal_credit_response_verified']
    assert result['missing_extinction_not_low_affinity_verified'] and result['sparse_early_credit_not_filled_verified']


@pytest.mark.parametrize('conflict', ['extinct_zero', 'final_rescore', 'filled_credit', 'wrong_capacity',
    'descendant_bonus', 'causal_survival', 'wrong_base_prior', 'extra_head', 'identity_jacobian'])
def test_hash_correct_response_cannot_ignore_revised_skill(tmp_path, conflict):
    root, folder, ref = sources()
    bad = copy.deepcopy(read_json(folder/'response.json'))
    if conflict == 'extinct_zero': bad['lineage_case']['extinction_affinity_label'] = 0.
    elif conflict == 'final_rescore': bad['lineage_case']['decoded_final_rescore_label'] = True
    elif conflict == 'filled_credit': bad['lineage_case']['whole_window_credit_strata_allowed'] = True
    elif conflict == 'wrong_capacity': bad['lineage_case']['teacher_cap_per_batch'] = 10
    elif conflict == 'descendant_bonus': bad['descendant_count_prior_bonus'] = True
    elif conflict == 'causal_survival': bad['survival_is_affinity_causality'] = True
    elif conflict == 'wrong_base_prior': bad['lineage_case']['base_prior'] = 'equal_teacher'
    elif conflict == 'extra_head': bad['production_extra_forward_calls_per_step'] = 1
    else: bad['reward_design']['derivative_path'] = 'identity'
    path = tmp_path/'response.json'; write_json(path, bad)
    with pytest.raises(ValueError):
        audit_terminal_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-terminal-lineage.md', folder/'input.json',
            folder/'prompt.txt', path, ref)
