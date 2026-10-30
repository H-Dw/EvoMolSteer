"""Regression and analytic checks for the candidate-local coordinate objective."""

import itertools

import pytest
import torch

from agenticgen.uniguide.geometry_batch import CandidateControl
from agenticgen.uniguide.geometry_reward import KEYS, evaluate, weights
from test_geometry_reward_controls import spec, tensors


def control(tmp_path, population=20):
    result = object.__new__(CandidateControl)
    result.initial = {"population": population}
    result.failed_units = {}
    result.rows = []
    result.folder = tmp_path
    return result


def test_healthy_checks_preserve_tensor_identity_and_grad_fn(tmp_path):
    guard = control(tmp_path)
    leaf = torch.arange(120.0, dtype=torch.float32).reshape(10, 4, 3).requires_grad_()
    value = leaf.square()
    for field in (
        "distance endpoint",
        "internal distance errors",
        "interaction distance errors",
        "loss",
    ):
        actual = guard.check_rows(field, value, list(range(10)))
        assert actual is value
        assert actual.grad_fn is value.grad_fn
    # A failure in a different concurrent shard must not add guard nodes here.
    guard.failed_units[1] = {"fixture": "unrelated failed shard"}
    assert guard.check_rows("distance endpoint", value, list(range(10))) is value


def test_network_traversal_keeps_all_healthy_tensors_and_structural_fields(tmp_path):
    guard = control(tmp_path)
    leaf = torch.ones(20, 3, requires_grad=True)
    value = leaf * 2
    affinity = {"prediction": torch.full((20,), float("nan"))}
    mask = torch.ones(20, 3, dtype=torch.bool)
    result = guard.prepare_output(
        {
            "coords": value,
            "heads": [value, (value,)],
            "affinity": affinity,
            "mask": mask,
        },
        list(range(20)),
    )
    assert result["coords"] is value
    assert result["heads"][0] is value and result["heads"][1][0] is value
    assert result["affinity"] is affinity and result["mask"] is mask
    assert guard.failed_units == {}


def test_discovery_precedes_quarantine_and_preserves_failure_evidence(tmp_path):
    guard = control(tmp_path)
    healthy = torch.ones(20, 3, requires_grad=True)
    broken = healthy.detach().clone()
    broken[13, 1] = float("nan")
    result = guard.prepare_output({"first": healthy, "last": broken}, list(range(20)))
    assert torch.equal(result["first"][:10], healthy[:10])
    assert not result["first"][10:].any()
    gradient = torch.autograd.grad(result["first"].sum(), healthy)[0]
    assert torch.equal(gradient[:10], torch.ones(10, 3))
    assert not gradient[10:].any()
    assert (tmp_path / "nonfinite_unit_001.pt.gz").exists()
    assert guard.failed_units[1]["row"] == 13
    with pytest.raises(ValueError, match="assign numerical failure"):
        guard.check_rows("layout", healthy, [0])


@pytest.mark.parametrize("switches", list(itertools.product((0, 1), repeat=3)))
def test_all_component_switches_match_analytic_distance_gradient(switches):
    condition = spec(True)
    reference, valid = tensors(condition)
    endpoint = (
        reference * 1.3 + reference.new_tensor([0.2, 0.5, -0.3])
    ).requires_grad_()
    configured = dict(zip(KEYS, switches))
    result, _ = evaluate(endpoint, condition, valid, configured)
    gradient = torch.autograd.grad(result[0].sum(), endpoint)[0]

    expected_loss = endpoint.new_zeros(1)
    expected_gradient = torch.zeros_like(endpoint)
    positions = endpoint.detach()
    categories = ("controlled_internal", "controlled_neighborhood")
    for weight, category in zip(switches[:2], categories):
        pairs = [p for p in condition["pairs"] if p["category"] == category]
        for pair in pairs:
            left, right = pair["generated_slots"]
            vector = positions[:, left] - positions[:, right]
            length = vector.norm(dim=-1)
            error = length - pair["reference_distance_angstrom"]
            factor = weight
            expected_loss += factor * error.square() / 2
            force = factor * error[:, None] * vector / length[:, None]
            expected_gradient[:, left] += force
            expected_gradient[:, right] -= force
    pairs = condition["interaction_pairs"]
    for pair in pairs:
        slot = pair["generated_slot"]
        vector = positions[:, slot] - positions.new_tensor(
            pair["target_coordinates_angstrom"]
        )
        length = vector.norm(dim=-1)
        error = length - pair["reference_distance_angstrom"]
        factor = switches[2]
        expected_loss += factor * error.square() / 2
        expected_gradient[:, slot] += factor * error[:, None] * vector / length[:, None]
    torch.testing.assert_close(result[0], expected_loss, atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-14, rtol=1e-12)


def test_full_coordinate_vjp_guard_identity_and_no_cross_candidate_gradient(tmp_path):
    condition = spec(True)
    reference, valid = tensors(condition)
    leaf = (reference.repeat(10, 1, 1) * 1.2).requires_grad_()
    valid = valid.repeat(10, 1)
    matrix = leaf.new_tensor([[1.2, 0.3, 0], [0, 0.8, 0.1], [0.2, 0, 1.1]])
    endpoint = leaf @ matrix + 0.4
    controls = [dict(zip(KEYS, (1, i % 2, 1))) for i in range(10)]
    plain = evaluate(endpoint, condition, valid, controls)[0][0]
    guard = control(tmp_path, population=10)
    guarded = evaluate(endpoint, condition, valid, controls, check=guard.check_rows)[0][
        0
    ]
    first = torch.autograd.grad(plain.sum(), leaf, retain_graph=True)[0]
    second = torch.autograd.grad(guarded.sum(), leaf, retain_graph=True)[0]
    assert torch.equal(first, second)
    isolated = torch.autograd.grad(guarded[0], leaf, retain_graph=True)[0]
    assert not isolated[1:].any()
    assert torch.autograd.gradcheck(
        lambda x: evaluate(x @ matrix + 0.4, condition, valid, controls)[0][0],
        (leaf,),
    )


def test_weight_layout_and_removal_of_current_state_api():
    condition = spec(True)
    endpoint, valid = tensors(condition)
    with pytest.raises(ValueError, match="weight count"):
        evaluate(endpoint, condition, valid, [weights(condition)] * 2)
    with pytest.raises(TypeError, match="current"):
        evaluate(endpoint, condition, valid, current=endpoint)


@pytest.mark.parametrize("component", [False, True])
def test_summed_condition_loss_retains_graph_preserving_safety_callback(component):
    from agenticgen.uniguide.geometry_condition import loss, prepare
    from test_uniguide_geometry import features

    condition = prepare(features(), component_reward=component)
    endpoint, valid = tensors(condition)
    endpoint = (endpoint * 1.3).requires_grad_()
    calls = []

    def inspect(name, value):
        calls.append(name)
        return value

    checked = loss(endpoint, condition, valid, check=inspect)[0]
    reference = loss(endpoint, condition, valid)[0]
    assert calls[:2] == ["distance endpoint", "internal distance errors"]
    assert torch.equal(checked, reference)
    assert torch.equal(
        torch.autograd.grad(checked.sum(), endpoint, retain_graph=True)[0],
        torch.autograd.grad(reference.sum(), endpoint)[0],
    )


def test_increment_overflow_is_quarantined_before_global_finite_check(tmp_path):
    from agenticgen.uniguide.condition import guidance_delta

    guard = control(tmp_path)
    gradient = torch.ones(20, 4, 3)
    gradient[13] = 1e30
    scales = torch.ones(20)
    scales[13] = 1e30
    time = torch.full((20,), 0.1)
    valid = torch.ones(20, 4, dtype=torch.bool)
    reference, _ = guidance_delta(
        gradient[:10], time[:10], 0.00999, scales[:10], 1.0, valid[:10]
    )
    delta, _ = guidance_delta(
        gradient, time, 0.00999, scales, 1.0, valid, check=guard.check_rows
    )
    assert torch.equal(delta[:10], reference)
    assert not delta[10:].any()
    assert guard.failed_units[1]["field"] == "guidance increment"
    # Independent callers retain fail-fast behavior, not a silent zero gradient.
    with pytest.raises(FloatingPointError):
        guidance_delta(gradient, time, 0.00999, scales, 1.0, valid)
