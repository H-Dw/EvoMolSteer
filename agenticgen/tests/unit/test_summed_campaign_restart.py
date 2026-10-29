"""A new objective can change its identity, but not the frozen molecular input."""

import copy
import importlib.util
from pathlib import Path

import pytest

path = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/20261029_chemical_interact_sum/prepare_run.py"
)
spec = importlib.util.spec_from_file_location("summed_campaign_restart", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def subjects():
    condition = {
        "policy": "old",
        "reward_policy": "old",
        "condition_sha256": "old",
        "distance_normalization": {"mode": "old"},
        "controlled_atom_ids": [str(i) for i in range(9)],
        "interact_atoms": [{"atom_id": str(i)} for i in range(9)],
        "atom_to_generated_slot": {str(i): i for i in range(9)},
        "reference_coordinates_angstrom": [[i, 0, 0] for i in range(9)],
        "interaction_pairs": [{"atom_id": "0", "target_atom_id": "protein"}],
    }
    before = [{"teacher": {"target": "teacher"}, "condition": condition}]
    after = copy.deepcopy(before)
    after[0]["condition"].update(
        policy="new", reward_policy="new", condition_sha256="new"
    )
    del after[0]["condition"]["distance_normalization"]
    return before, after


def test_new_config_preserves_teacher_local_search_and_budget_relevant_settings():
    old = {
        "protocol": "old",
        "root": "old",
        "rounds": 5,
        "chemical_inpainting": {"enabled": True},
        "agent_tools": {"skill_release": "2.2.0"},
        "calibration": {"search_center_by_teacher": {"first": 1, "second": 4}},
        "guidance_reward": {"mode": "old"},
    }
    new = module.configuration(old, Path("new"))
    assert new["calibration"] == old["calibration"]
    assert (
        new["protocol"] == module.NEW_PROTOCOL
        and new["agent_tools"]["skill_release"] == "2.3.0"
    )
    assert "guidance_reward" not in new and old["guidance_reward"] == {"mode": "old"}


def test_condition_objective_identity_may_change_without_input_geometry_changes():
    module.assert_same_subjects(*subjects())


@pytest.mark.parametrize(
    "field",
    ["reference_coordinates_angstrom", "atom_to_generated_slot", "interaction_pairs"],
)
def test_geometry_and_mapping_changes_are_rejected(field):
    before, after = subjects()
    after[0]["condition"][field] = []
    with pytest.raises(ValueError, match="geometry"):
        module.assert_same_subjects(before, after)


def test_old_condition_identity_cannot_be_reused_for_new_loss():
    before, after = subjects()
    after[0]["condition"]["condition_sha256"] = "old"
    with pytest.raises(ValueError, match="new frozen"):
        module.assert_same_subjects(before, after)
