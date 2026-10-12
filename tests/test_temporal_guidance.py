import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from evomolsteer.io import read_json, digest
from evomolsteer.temporal_guidance.policy import make_policy, validate_policy, program_signature, clock_grid
from evomolsteer.temporal_guidance.mining import fit_stages
from evomolsteer.temporal_guidance.workflow import compile_recipe, require_execution, TemporalEvidenceTools
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.scheduled_endpoint_reward import ScheduledEndpointReward
from evomolsteer.generation.endpoint_controller import endpoint_pullback
from evomolsteer.generation.local_reward import bounded_local_step

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/experiments/temporal_guidance_20261013/study.json"


def fixture(window=(.2, .6), steps=10):
    from evomolsteer.continuous.affinity_geometry import names
    points = np.array([[0., 0., 0.]])
    static = np.array([[-1., 0., 0.], [1., 0., 0.]])
    dynamic = static + np.array([0., 1., .3])
    times = clock_grid(window, steps)
    ref = {"schema_version": "affinity-endpoint-library-1.0", "window": list(window), "times": times.tolist(),
           "features": names(points), "feature_scale": [1.] * len(names(points)), "landmarks_A": points.tolist(),
           "origin_A": [0., 0., 0.], "frames": [
               {"time": float(t), "teacher_endpoint_A": [static.tolist()], "teacher_scores": [0.],
                "trajectory_teacher_endpoint_A": [dynamic.tolist()], "trajectory_teacher_batches": [0]} for t in times]}
    p = {"window": list(window), "reward_view": "endpoint_pointcloud", "derivative_path": "flowr_endpoint_vjp",
         "mixture_temperature": .5, "pointcloud_delta_A": 1., "time_ramp_power": 0}
    return p, ref


@pytest.mark.parametrize("route", ["dose", "reward"])
def test_identity_scalar_and_gradient_are_bitwise_equal_to_original(route):
    p, ref = fixture()
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, route)
    original, scheduled = EndpointGeometryReward(p, ref), ScheduledEndpointReward(p, ref)
    x = torch.tensor([[[-.7, .1, -.2], [.6, .1, .2]]], dtype=torch.float64, requires_grad=True)
    atoms, mask = torch.zeros((1, 2), dtype=torch.int64), torch.ones((1, 2), dtype=torch.bool)
    a, da = original(x, atoms, mask, .2, x.detach())
    b, db = scheduled(x, atoms, mask, .2, x.detach())
    ga, = torch.autograd.grad(a.sum(), x)
    gb, = torch.autograd.grad(b.sum(), x)
    assert torch.equal(a, b) and torch.equal(ga, gb) and torch.equal(da["dose_gate"], db["dose_gate"])
    assert scheduled.active(.2, .3) and not scheduled.active(.6, .7)


def test_scalar_scaling_is_cancelled_but_controller_dose_factor_is_operative():
    g = torch.tensor([[[.3, .8, -.5], [.1, -.2, .6]]], dtype=torch.float64)
    native, mask = torch.ones_like(g) * .02, torch.ones((1, 2), dtype=torch.bool)
    apply = lambda gradient, gate: bounded_local_step(gradient, native, mask, torch.tensor([gate]), .33, 1., 10., torch.tensor([100.]))[0]
    torch.testing.assert_close(apply(g, 1.), apply(g * 15, 1.))
    torch.testing.assert_close(apply(g, .4), apply(g, 1.) * .4)


def test_blended_scalar_uses_one_model_forward_one_vjp_and_no_head_derivative():
    p, ref = fixture()
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "reward", blend=(.5, .5, .5))
    reward = ScheduledEndpointReward(p, ref)
    x = torch.tensor([[[-.7, .1, -.2], [.6, .1, .2]]], dtype=torch.float64)
    calls = []
    def predict(z):
        calls.append(1)
        affinity = z.square().sum((1, 2))
        affinity.register_hook(lambda g: (_ for _ in ()).throw(AssertionError("Head was differentiated")))
        return {"coords": torch.sin(z), "atomics": torch.ones((1, 2, 1)), "affinity": {"pic50": affinity}}, {}
    result = endpoint_pullback(predict, reward, x, torch.ones((1, 2), dtype=torch.bool), 1., x.new_zeros((1, 3)), .2)
    assert len(calls) == 1 and torch.isfinite(result[2]).all()
    assert float(result[4]["temporal_teacher_blend"][0]) == .5
    # This test uses a synthetic predictor; it is not a FLOWR performance result.


def test_temporal_target_changes_gradient_direction_at_the_same_dose():
    p, ref = fixture()
    mask, atoms = torch.ones((1, 2), dtype=torch.bool), torch.zeros((1, 2), dtype=torch.int64)
    x = torch.tensor([[[-.6, 0., 0.], [.6, 0., 0.]]], dtype=torch.float64, requires_grad=True)
    def gradient(blend):
        q = copy.deepcopy(p)
        q["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "reward", blend=(blend,) * 3)
        value, detail = ScheduledEndpointReward(q, ref)(x, atoms, mask, .2, x.detach())
        g, = torch.autograd.grad(value.sum(), x)
        assert detail["dose_gate"].item() == 1.
        return g
    a, b = gradient(0), gradient(1)
    assert a[..., 1].abs().sum() == 0 and b[..., 1].sum() > 0


def test_policy_is_dynamic_and_keeps_requested_ratio_budget_not_claimed_physical_dose():
    t = clock_grid([.1, .7], 100)
    p = make_policy([.1, .7], t, 100, "dose", dose=(.35, 1., 1.65))
    assert np.isclose(np.sum(p["dose_factors"]) / 100, .6)
    assert p["dose_factors"][0] < 1 < p["dose_factors"][-1]
    assert p["window"] == [.1, .7]
    for key, bad in [("teacher_blend", [1.] * len(t)), ("dose_factors", [float("nan")] * len(t))]:
        q = copy.deepcopy(p)
        q[key] = bad
        with pytest.raises(ValueError):
            validate_policy(q, [.1, .7], t)


def test_reward_reversal_and_constant_controls_have_equal_discrete_parameter_mean():
    t = clock_grid([0., .5], 100)
    a = make_policy([0., .5], t, 100, "reward", blend=(1., .5, 0.))
    b = make_policy([0., .5], t, 100, "reward", blend=(0., .5, 1.))
    np.testing.assert_allclose(a["teacher_blend"], b["teacher_blend"][::-1], atol=1e-15)
    assert np.isclose(np.mean(a["teacher_blend"]), .5)
    assert a["normalized_phase_support"] == [0., .49]
    assert a["teacher_blend"][-1] == 0 and b["teacher_blend"][0] == 0


def test_no_teacher_clock_extrapolation_or_incomplete_grid():
    p, ref = fixture()
    with pytest.raises(ValueError):
        make_policy(p["window"], ref["times"][:-1], 10, "dose")
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "reward")
    reward = ScheduledEndpointReward(p, ref)
    with pytest.raises(ValueError):
        reward(torch.zeros((1, 2, 3)), torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool), .55)


def test_data_spline_uses_whole_window_and_recovers_a_stable_observed_knot():
    t = np.arange(30) / 100
    truth = .2 + 2 * t + 5 * np.maximum(t - .15, 0)
    rng = np.random.default_rng(42)
    y = truth + rng.normal(0, .001, size=(8, len(t)))
    fit = fit_stages(t, y, minimum_nodes=5)
    assert fit["accepted_for_stage_hypothesis"] and abs(fit["knot_times"][0] - .15) < .021
    assert fit["observed_support"] == [0., .29] and not fit["extrapolation_allowed"]


def test_stationary_full_window_does_not_force_stages():
    t = np.arange(30) / 100
    y = np.ones((8, 30)) * .4
    fit = fit_stages(t, y, minimum_nodes=5)
    assert fit["method"] == "constant" and not fit["accepted_for_stage_hypothesis"]


def test_all_registered_recipes_preserve_temperatures_and_m2_sources():
    config = read_json(ROOT / CONFIG)
    base = read_json(ROOT / config["base_program"])
    for name, recipe in config["recipes"].items():
        p, ref = compile_recipe(ROOT, config, name)
        for key in ["mixture_temperature", "teacher_endpoint_temperature_A2", "teacher_neighbors", "teacher_score_beta", "native_rms_ratio"]:
            assert p[key] == base[key]
        assert p["reference_sha256"] == digest(ROOT / ref)
        if recipe["route"] == "dose":
            assert all(v == 0 for v in p["temporal_policy"]["teacher_blend"])
        else:
            assert all(v == 1 for v in p["temporal_policy"]["dose_factors"])


def test_identical_executed_arrays_ignore_recipe_metadata_and_collapse_identity():
    p, ref = fixture()
    signature = program_signature(p)
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "dose")
    assert program_signature(p) == signature
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "dose", dose=(.35, 1., 1.65))
    signature = program_signature(p)
    p["temporal_policy"]["knots_origin"] = "Another label for the same executed table"
    assert program_signature(p) == signature
    p["temporal_policy"]["dose_factors"][0] += .001
    assert program_signature(p) != signature


@pytest.mark.parametrize("execute", [False, True])
def test_disabled_plan_cannot_call_roles_or_flowr(execute):
    config = read_json(ROOT / CONFIG)
    with pytest.raises(ValueError, match="Planning configuration"):
        require_execution(config, execute)


def test_current_skills_are_not_modified_or_injected_twice():
    from evomolsteer.temporal_guidance.workflow import render_instructions
    config = read_json(ROOT / CONFIG)
    condition = next(c for c in config["conditions"] if c["id"] == "S11_both_extensions")
    text, receipts = render_instructions(ROOT, config, condition, "Designer")
    assert text.count("name: evomolsteer-designer\n") == 1
    assert "staged-reward-designer" in text
    assert "--proxy" not in text and "VPN" not in text
    assert "prior temperatures" in text
    assert len(receipts) == 4


def test_curve_source_handles_use_actual_immutable_observed_window():
    config = read_json(ROOT / CONFIG)
    tools = TemporalEvidenceTools(ROOT / config["evidence"])
    name = next(iter(tools.curves["functions"]))
    observed = tools.execute("get_curve", {"feature": name})
    assert observed["ok"] and observed["observation"]["source"]["kind"] == "curve"
    tools.verify_sources()
    other = TemporalEvidenceTools(ROOT / config["evidence"])
    other.import_references(tools.provenance())
    assert "curve_support" in other.resolve([observed["observation"]["ref"]])


def test_delivery_audit_rejects_suffix_injection_and_summarizes_without_time_bins(tmp_path):
    import json
    from evomolsteer.temporal_guidance.delivery import audit_delivery
    p, ref = fixture()
    p["temporal_policy"] = make_policy(p["window"], ref["times"], 10, "dose")
    path = tmp_path / "batch_6376" / "guidance_trace.jsonl"
    path.parent.mkdir()
    rows = []
    for i in range(10):
        active = 2 <= i < 6
        rows.append({"step": i, "score_time": i / 10, "state_time": (i + 1) / 10,
                     "production_target_forward_calls": 1, "particle_resampled": False,
                     "affinity_outputs_detached": True, "arm": "gradient", "active": active,
                     "reward_evaluated": active, "injection_l2_A": [.2 if active else 0.] * 2,
                     "temporal_dose_factor": [1., 1.], "temporal_teacher_blend": [0., 0.], "temporal_delta_A": [1., 1.],
                     "requested_rms_A": [.2, .2], "injection_rms_A": [.2, .2], "calibration_rms_A": [1., 1.],
                     "gradient_norm": [.3, .4], "cap_factor": [1., 1.], "backtrack_factor": [1., 1.],
                     "first_order_reward_change": [.01, .01]})
    write = lambda: path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    write()
    report = audit_delivery([path], p, tmp_path / "audit", 10)
    assert report["passed"] and report["no_time_bins"] and report["batches"][0]["controlled_steps"] == 4
    rows[7]["injection_l2_A"] = [.01, .01]
    write()
    with pytest.raises(ValueError, match="inactive-window"):
        audit_delivery([path], p, tmp_path / "bad_audit", 10)


def test_profile_tools_preserve_citations_across_role_handoff(tmp_path):
    from evomolsteer.io import write_json
    config = read_json(ROOT / CONFIG)
    tools = TemporalEvidenceTools(ROOT / config["evidence"])
    name = next(iter(tools.curves["functions"]))
    path = tmp_path / "stage_profiles.json"
    write_json(path, {"window": tools.curves["window"], "score_times": tools.curves["times"],
                      "input_sha256": {"packet": tools.files["packet.json"]},
                      "functions": {name: {"accepted_for_stage_hypothesis": False, "knot_times": []}}})
    a = TemporalEvidenceTools(ROOT / config["evidence"], path)
    response = a.execute("get_stage_profile", {"feature": name})
    assert response["ok"]
    d = TemporalEvidenceTools(ROOT / config["evidence"], path)
    d.import_references(a.provenance())
    d.verify_sources()
    assert d.resolve([response["observation"]["ref"]]) == ["temporal_profile/" + name]
