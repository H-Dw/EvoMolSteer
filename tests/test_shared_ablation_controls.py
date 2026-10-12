import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evomolsteer.ablation.shared_controls import SharedControls, UNCHANGED_M2, batch_key, verify_attempts
from evomolsteer.continuous.m2_skill_api import effective_signature
from evomolsteer.io import digest, read_json, write_json


@pytest.fixture
def bank(tmp_path):
    batches = list(range(100, 108))
    reference = tmp_path / "reference.json"
    write_json(reference, {"geometry": [1, 2, 3]})
    program = {"program_id": "M2", "eta": .33, "reference_sha256": digest(reference)}
    write_json(tmp_path / "program.json", program)
    sha = effective_signature(program)
    f = pd.DataFrame({"batch": np.repeat(batches, 50), "slot": np.tile(np.arange(50), 8),
        "pic50_on_rescore": 7.8, "valid_connected": True, "pb_fast_pass": True,
        "energy_status": "converged", "mmff_relief_per_heavy": .5})
    f.loc[0, ["valid_connected", "pb_fast_pass"]] = False
    f.loc[0, "energy_status"] = "not_applicable"
    f.loc[0, "mmff_relief_per_heavy"] = np.nan
    f.to_csv(tmp_path / "metrics.csv", index=False)
    result = {"name": "completed_M2", "effective_signature": sha, "prior_seed_mode": "indexed_batch",
        "initial_state_signatures": {f"{arm}/batch_{b:03d}": str(b) for arm in ["gradient", "unguided"] for b in batches},
        "outcomes": {arm: {"metrics": "metrics.csv"} for arm in ["gradient", "unguided"]}}
    m = {"batches": batches, "batch_size": 50, "prior_seed_mode": "indexed_batch", "master_seed": 42,
        "steps": 100, "program": "program.json", "effective_signature": sha, "result": result,
        "files": {"metrics.csv": digest(tmp_path / "metrics.csv")}, "generator_sources": {}}
    write_json(tmp_path / "manifest.json", m)
    policy = {"schema_version": "shared-controls-1.0", "scope": "all_unfinished_evaluations",
        "protocol_sha256": "activation", "manifest": "manifest.json", "manifest_sha256": digest(tmp_path / "manifest.json")}
    write_json(tmp_path / "policy.json", policy)
    shared = SharedControls(tmp_path, "policy.json", "activation", {}, "indexed_batch")
    return tmp_path, shared, program, reference


def owner_for(repo, shared):
    calls = []
    def infer(path, sha, **kwargs):
        calls.append(kwargs)
        result = copy.deepcopy(shared.result)
        result.update(name="new_reward", effective_signature=sha)
        return result
    return SimpleNamespace(state={"measured": {}}, frames=lambda result, arm="gradient": pd.read_csv(repo / result["outcomes"][arm]["metrics"]),
        _budget=lambda: None, save=lambda: None, infer=infer), calls


def test_identical_m2_reuses_scores_and_failed_attempts_without_inference(bank):
    repo, shared, program, reference = bank
    owner, calls = owner_for(repo, shared)
    result = shared.measurement(owner, repo / "program.json", reference, effective_signature(program))
    assert result is shared.result
    assert not calls
    assert len(owner.frames(result)) == 400
    assert not owner.frames(result).iloc[0].valid_connected
    assert next(iter(owner.state["numerical_reuse"].values()))["note"] == UNCHANGED_M2


def test_new_program_generated_once_with_fixed_batches_then_reused(bank):
    repo, shared, program, reference = bank
    owner, calls = owner_for(repo, shared)
    program["eta"] = .5
    write_json(repo / "candidate.json", program)
    sha = effective_signature(program)
    a = shared.measurement(owner, repo / "candidate.json", reference, sha)
    b = shared.measurement(owner, repo / "candidate.json", reference, sha)
    assert a == b
    assert len(calls) == 1
    assert calls[0]["batches"] == shared.batches
    assert next(iter(owner.state["numerical_reuse"].values()))["first_action"] == "generate_new_numerical_program"


def test_cached_initial_state_mismatch_rejected(bank):
    repo, shared, program, reference = bank
    owner, calls = owner_for(repo, shared)
    program["eta"] = .5
    write_json(repo / "candidate.json", program)
    sha = effective_signature(program)
    wrong = copy.deepcopy(shared.result)
    wrong["effective_signature"] = sha
    wrong["initial_state_signatures"]["gradient/batch_100"] = "different"
    owner.state["measured"][sha + ":" + batch_key(shared.batches)] = wrong
    with pytest.raises(ValueError, match="initial state mismatch"):
        shared.measurement(owner, repo / "candidate.json", reference, sha)
    assert not calls


@pytest.mark.parametrize("change", ["metrics", "manifest", "policy", "rng"])
def test_immutable_bank_or_policy_changes_fail_closed(bank, change):
    repo, shared, _, _ = bank
    state = {"shared_control_policy_sha256": shared.policy_sha256}
    if change == "metrics":
        with (repo / "metrics.csv").open("a") as stream: stream.write("\nchanged")
    elif change == "manifest":
        with (repo / "manifest.json").open("a") as stream: stream.write(" ")
    elif change == "policy":
        with (repo / "policy.json").open("a") as stream: stream.write(" ")
    with pytest.raises(ValueError):
        SharedControls(repo, "policy.json", "activation", state, "legacy_stream" if change == "rng" else "indexed_batch")


def test_full_drop_identity_note_and_no_extra_generation_units(bank):
    _, shared, _, _ = bank
    condition = {"rows": [{"signature": shared.manifest["effective_signature"]}] * 6}
    record = shared.comparison(condition, condition)
    assert record["both_execute_M2"]
    assert record["note"] == UNCHANGED_M2
    assert record["shared_control_bank"]["attempts_per_arm"] == 400
    assert not record["independent_confirmation"]


def test_missing_failed_slot_and_missing_affinity_are_not_filtered(bank):
    repo, shared, _, _ = bank
    f = pd.read_csv(repo / "metrics.csv")
    with pytest.raises(ValueError, match="coverage"):
        verify_attempts(f.iloc[1:], shared.batches, 50)
    f.loc[0, "pic50_on_rescore"] = np.nan
    with pytest.raises(ValueError, match="missing affinity"):
        verify_attempts(f, shared.batches, 50)


def test_all_remaining_stages_resolve_to_same_bank_and_controls_never_infer(bank):
    repo, shared, _, _ = bank
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_scientific_ablation_study import ScientificStudy
    s = ScientificStudy.__new__(ScientificStudy)
    s.shared_controls = shared
    s.state = {"matched_controls": {}}
    s.infer = lambda *args, **kwargs: pytest.fail("Controls must never regenerate")
    for requested in [[6100,6101,6102,6103], range(7000,7008), range(8000,8008), range(8500,8508)]:
        assert s.evaluation_batches(requested) == shared.batches
        assert s.controls(requested) is shared.result
    assert not s.evaluation_metadata()["independent_confirmation"]


def test_program_metadata_and_explicit_default_do_not_change_signature(bank):
    _, _, program, _ = bank
    clone = dict(program, program_id="full", round=2, pointcloud_delta_A=1.)
    assert effective_signature(program) == effective_signature(clone)
