import pytest

from evomolsteer.adaptive_schedule.comparisons import validity_transitions, shared_batch_bootstrap
from evomolsteer.adaptive_schedule.io import defaults


def test_validity_transitions_do_not_hide_failed_slots():
    good = {"pic50_on_rescore": 6, "build_success": True, "connected": True}
    bad = {"pic50_on_rescore": None, "build_success": False, "connected": False}
    assert validity_transitions([good, bad, good, bad], [good, good, bad, bad]) == {
        "attempts": 4, "both_valid": 1, "invalid_to_valid": 1, "valid_to_invalid": 1, "both_invalid": 1}
    with pytest.raises(ValueError): validity_transitions([good], [])


def test_shared_seed_batches_are_not_resampled_independently_across_teachers():
    rows = [{"teacher_id": t, "batch": b, "delta": sign * b}
            for t, sign in [("a", 1), ("b", -1)] for b in range(5)]
    result = shared_batch_bootstrap(rows, ["a", "b"], "delta")
    assert result["teacher_equal_mean_delta"] == 0
    assert result["exploratory_ci"] == [0, 0]
    rows.pop()
    assert shared_batch_bootstrap(rows, ["a", "b"], "delta")["exploratory_ci"] is None


def test_new_server_keeps_protocol_and_changes_only_deployment_identity():
    config = defaults()
    assert config["ssh"] == {"host": "ssh.zzai.scnet.cn", "port": 11762, "user": "root"}
    assert config["flowr"] == "/opt/MolSteer/flowr_root"
    assert config["initial_ticks"] == [5]*4 and config["rounds"] == 10 and config["steps"] == 100
    assert config["seeds"] == [42, 100045, 200048, 300051, 400054]
