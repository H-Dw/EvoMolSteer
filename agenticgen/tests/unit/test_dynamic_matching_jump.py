"""Dynamic matching, shared chemistry, and three-intensity jump contracts."""

import copy
from types import SimpleNamespace

import h5py
import jsonschema
import numpy as np
import pytest
import torch

from test_absolute_coordinate_protocol import condition as condition
from agenticgen.uniguide import absolute_coordinate as coordinates
from agenticgen.uniguide import linear_scale as scale
from agenticgen.uniguide import exploration
from agenticgen.uniguide.io import value_sha, read, write


@pytest.fixture
def dynamic_condition(condition):
    body = {
        k: copy.deepcopy(v) for k, v in condition.items() if k != "condition_sha256"
    }
    body["assignment_policy"] = coordinates.DYNAMIC_ASSIGNMENT
    return dict(body, condition_sha256=value_sha(body))


def control(condition, step=10, resume=None):
    return SimpleNamespace(
        geometry_condition=condition,
        active_step=step,
        resume=resume,
        check_rows=lambda name, value, ids: value,
    )


def endpoints():
    return torch.tensor(
        [[[9.0, 9.0, 9.0], [0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]], dtype=torch.float64
    )


def test_new_conditions_default_to_dynamic_without_changing_legacy(condition, tmp_path):
    features = {
        key: condition[key]
        for key in ("atoms", "bonds", "controlled_atom_ids", "coordinate_frame")
    }
    features["coordinates_angstrom"] = condition["reference_coordinates_angstrom"]
    fresh = coordinates.prepare(features, tmp_path / "protein.pdb")
    assert fresh["assignment_policy"] == coordinates.DYNAMIC_ASSIGNMENT
    assert fresh["condition_sha256"] != condition["condition_sha256"]
    assert condition["assignment_policy"] == coordinates.ASSIGNMENT


def test_dynamic_assignment_changes_each_step_without_rng_or_identity_claim(
    dynamic_condition,
):
    c = control(dynamic_condition)
    state = coordinates.AssignmentState(c)
    x = endpoints()
    valid = torch.ones(1, 3, dtype=torch.bool)
    before = torch.get_rng_state().clone()
    first = state.evaluate(x, valid, [0])
    assert first[3].tolist() == [[1, 2]]
    c.active_step = 11
    second = state.evaluate(x[:, [1, 0, 2]], valid, [0])
    assert second[3].tolist() == [[0, 2]] and second[0].item() == 0.0
    assert state.rows[0]["first_slots"] == [1, 2] and state.rows[0]["slots"] == [0, 2]
    assert state.rows[0]["last_step"] == 11
    assert torch.equal(before, torch.get_rng_state())


def test_dynamic_loss_keeps_full_input_vjp_and_matches_finite_difference(
    dynamic_condition,
):
    c = control(dynamic_condition)
    state = coordinates.AssignmentState(c)
    x = torch.tensor(
        [[[0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [0.8, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.ones(1, 3, dtype=torch.bool)
    result = state.evaluate(2 * x, valid, [0])
    gradient = torch.autograd.grad(result[0].sum(), x)[0]
    expected = torch.tensor(
        [[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=torch.float64
    )
    assert torch.allclose(gradient, expected, atol=1e-12, rtol=1e-12)
    direction = gradient / gradient.norm()
    epsilon = 1e-5
    positive = state.evaluate(2 * (x.detach() + epsilon * direction), valid, [0])[0]
    negative = state.evaluate(2 * (x.detach() - epsilon * direction), valid, [0])[0]
    assert ((positive - negative) / (2 * epsilon)).item() == pytest.approx(
        gradient.norm().item(), rel=1e-7
    )


def test_dynamic_mapping_mask_is_current_and_cache_remains_bounded(dynamic_condition):
    c = control(dynamic_condition)
    state = coordinates.AssignmentState(c)
    x = endpoints()
    valid = torch.ones(1, 3, dtype=torch.bool)
    for step in range(10, 100):
        c.active_step = step
        points = x if step % 2 == 0 else x[:, [1, 0, 2]]
        actual = state.evaluate(points, valid, [0])[3]
        assert actual.tolist() == ([[1, 2]] if step % 2 == 0 else [[0, 2]])
    assert len(state._assignments) == 1 and len(state.objective._tensors) == 1
    c.active_step = 100
    valid[0, 0] = False
    actual = state.evaluate(x[:, [1, 0, 2]], valid, [0])[3]
    assert 0 not in actual[0].tolist()


def test_dynamic_resume_checks_previous_mapping_then_rematches(dynamic_condition):
    c = control(dynamic_condition)
    state = coordinates.AssignmentState(c)
    valid = torch.ones(1, 3, dtype=torch.bool)
    x = endpoints()
    first = state.evaluate(x, valid, [0])
    saved = copy.deepcopy(state.rows)
    resume = dict(
        step=11,
        absolute_assignment_state=saved,
        absolute_assignment_sha256=value_sha(saved),
        previous_assignment=first[3].clone(),
    )
    recovered = coordinates.AssignmentState(control(dynamic_condition, 11, resume))
    recovered.restore()
    assert recovered.evaluate(x[:, [1, 0, 2]], valid, [0])[3].tolist() == [[0, 2]]
    corrupt = copy.deepcopy(resume)
    corrupt["previous_assignment"][0, 0] = 0
    with pytest.raises(ValueError, match="correspondence"):
        coordinates.AssignmentState(control(dynamic_condition, 11, corrupt)).restore()
    missing = dict(
        step=11,
        absolute_assignment_state={},
        absolute_assignment_sha256=value_sha({}),
        previous_assignment=first[3],
    )
    incomplete = coordinates.AssignmentState(control(dynamic_condition, 11, missing))
    incomplete.restore()
    with pytest.raises(ValueError, match="recovery"):
        incomplete.evaluate(x, valid, [0])


def test_chemical_state_uses_same_current_step_mapping_without_coordinates(
    dynamic_condition, tmp_path
):
    from test_geometry_chemical_state import fixture
    from agenticgen.uniguide.geometry_chemistry import match_fractions

    c, state, _ = fixture(tmp_path)
    c.folder = tmp_path / "dynamic"
    c.geometry_condition = dynamic_condition
    c.resume = None
    c.active_step = 10
    c.check_rows = lambda name, value, ids: value
    c.absolute_state = coordinates.AssignmentState(c)
    c.configure_chemistry()
    x = torch.cat(
        [
            endpoints(),
            torch.tensor([[[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]], dtype=torch.float64),
        ],
        dim=1,
    ).repeat(2, 1, 1)
    mask = state["mask"].bool()
    first = c.absolute_state.evaluate(x, mask, [0, 1])[3]
    before = torch.get_rng_state().clone()
    out = c.apply_chemical_inpainting(state, 10)
    assert all(
        values == [1.0, 1.0]
        for values in match_fractions(out, first, c.chemical_features).values()
    )
    c.active_step = 11
    c.rows.append(dict(step=11))
    second = c.absolute_state.evaluate(x[:, [1, 0, 2, 3, 4]], mask, [0, 1])[3]
    assert second.tolist() == [[0, 2], [0, 2]]
    final = c.apply_chemical_inpainting(out, 11)
    assert all(
        values == [1.0, 1.0]
        for values in match_fractions(final, second, c.chemical_features).values()
    )
    assert final["coords"] is state["coords"] and final["mask"] is state["mask"]
    assert final["self_condition"] == state["self_condition"]
    assert torch.equal(final["atomics"][:, 3], out["atomics"][:, 3])
    assert torch.equal(before, torch.get_rng_state())


def test_terminal_rematches_raw_prediction_and_keeps_identity_diagnostics(
    dynamic_condition, tmp_path
):
    from agenticgen.uniguide.geometry_sampling import (
        terminal_records,
        coordinate_population_statistics,
    )

    folder = tmp_path / "terminal"
    folder.mkdir()
    x = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.4, 0.0, 0.0]]]).repeat(
        10, 1, 1
    )
    output = dict(coords=x, mask=torch.ones(10, 3, dtype=torch.bool))
    with h5py.File(folder / "trace.h5", "w") as trace:
        trace["assignment"] = np.tile([1, 2], (100, 10, 1))
    model = SimpleNamespace(_generate_mols=lambda output, sanitise=True: [None] * 10)
    before = x.clone()
    rng = torch.get_rng_state().clone()
    rows = terminal_records(model, output, dynamic_condition, folder, 10)
    assert all(
        row["condition_generated_slots"] == [0, 2] and row[coordinates.RMSD] == 0
        for row in rows
    )
    assert all(
        row[
            "initial_assignment_condition_coordinate_root_mean_square_deviation_angstrom"
        ]
        == pytest.approx(2**0.5)
        for row in rows
    )
    receipt = read(folder / "COORDINATE_ASSIGNMENT.json")
    assert receipt["terminal_rematched"] and not receipt["no_terminal_rematching"]
    assert receipt["terminal_to_last_guidance_switch_fraction"] == [0.5] * 10
    assert torch.equal(output["coords"], before) and torch.equal(
        rng, torch.get_rng_state()
    )
    aggregate = {}
    coordinate_population_statistics(folder, rows, dynamic_condition, {}, aggregate)
    assert (
        aggregate["ranking_condition_coordinate_root_mean_square_deviation_angstrom"]
        == 0
    )
    assert aggregate["assignment_diagnostics_not_used_for_ranking"]["population"] == 10


def test_dynamic_trace_accepts_switching_and_rejects_stale_matches(
    dynamic_condition, tmp_path
):
    x = endpoints().float()
    values = torch.zeros(100, 1, 3, 3)
    values[10] = x
    values[11] = x[:, [1, 0, 2]]
    ids = np.zeros((100, 1, 2), dtype=np.int64)
    ids[10] = [[1, 2]]
    ids[11] = [[0, 2]]
    with h5py.File(tmp_path / "trace.h5", "w") as trace:
        trace["endpoint_A"] = values.numpy()
        trace["mask"] = np.ones((100, 1, 3), dtype=bool)
        trace["assignment"] = ids
    write(tmp_path / "steps.json", [dict(step=10), dict(step=11)])
    result = coordinates.verify_assignment_trace(tmp_path, dynamic_condition)
    assert (
        result["verified_candidate_steps"] == 2
        and result["stepwise_hungarian_verified"]
    )
    with h5py.File(tmp_path / "trace.h5", "a") as trace:
        trace["assignment"][11] = [[1, 2]]
    with pytest.raises(ValueError, match="Hungarian"):
        coordinates.verify_assignment_trace(tmp_path, dynamic_condition)


@pytest.mark.parametrize("candidate_batch", [False, True])
def test_step_hooks_accept_dynamic_switches_but_reject_frozen_switches(
    candidate_batch, monkeypatch
):
    from agenticgen.uniguide import sampler
    from agenticgen.uniguide.geometry_sampling import GeometryControl
    from agenticgen.uniguide.geometry_batch import CandidateControl

    monkeypatch.setattr(sampler.Control, "after", lambda self, state, step: state)
    kind = CandidateControl if candidate_batch else GeometryControl
    c = object.__new__(kind)
    c.absolute_state = SimpleNamespace(dynamic=True, save_cache=lambda: None)
    c.geometry_condition = dict(condition_sha256="test")
    c.scale_binding = {}
    c.rows = [dict(assignment_switch_fraction=[0.5])]
    c.apply_chemical_inpainting = lambda state, step: state
    c.step_scales = lambda step, population: [10] * population
    c.initial = dict(population=1)
    c.random_streams = SimpleNamespace(parts=[])
    c.failed_units = {}
    current = dict(coords=endpoints())
    assert c.after(current, 11) is current
    c.absolute_state.dynamic = False
    with pytest.raises(ValueError, match="[Ff]ixed|[Ff]rozen"):
        c.after(current, 11)


def test_fresh_jump_release_uses_three_complete_default_intensity_ranges(tmp_path):
    from agenticgen.uniguide.tool_runtime import configure

    c = configure(dict(root=str(tmp_path), protocol=coordinates.POLICY))
    settings = c["guidance_control"]
    assert settings["maximum_window_tick_change"] == 4
    assert settings["maximum_total_tick_change"] == 12
    descriptions = exploration.describe(c["guidance_exploration"], settings[scale.UNIT])
    assert [row["total_absolute_tick_change"] for row in descriptions] == [
        [1, 2],
        [3, 6],
        [7, 12],
    ]
    assert [row["total_absolute_strength_change"] for row in descriptions] == [
        [10, 20],
        [30, 60],
        [70, 120],
    ]
    assert all(
        row["allowed_directions"] == ["increase", "decrease", "mixed"]
        for row in descriptions
    )


def jump_domain(tmp_path):
    from test_linear_guidance import config, domain

    c = exploration.configure(config(tmp_path))
    old = domain(tmp_path)
    body = {k: v for k, v in old.items() if k != "domain_sha256"}
    body.update(
        exploration_policy=c["guidance_exploration"],
        confirmed_effective_levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    return c, scale.seal(body)


@pytest.mark.parametrize(
    "updates,tier",
    [
        (((0, 1),), "focused"),
        (((0, -1), (1, 1)), "focused"),
        (((0, 2), (1, -2)), "expanded"),
        (((0, -2), (1, 2)), "expanded"),
        (((0, 2), (1, -2), (2, 1), (3, -1)), "bold"),
    ],
)
def test_three_branch_hypotheses_are_jump_intensity_not_direction(
    tmp_path, updates, tier
):
    from test_linear_guidance import row, action

    _, domain = jump_domain(tmp_path)
    base = row(50)
    proposal = dict(action(base, updates), exploration_tier=tier)
    resolved = scale.resolve_action(proposal, {base["id"]: base}, domain, [base])
    assert resolved["exploration_tier"] == tier
    assert resolved["task_type"] == "optimization"
    assert resolved["total_absolute_tick_change"] == sum(
        abs(step) for _, step in updates
    )


def test_jump_intensity_cannot_be_faked_and_known_large_jump_is_not_frontier(tmp_path):
    from test_linear_guidance import row, action

    _, domain = jump_domain(tmp_path)
    base = row(50)
    wrong = dict(action(base, ((0, 3), (1, -3))), exploration_tier="focused")
    with pytest.raises(ValueError, match="intensity"):
        scale.resolve_action(wrong, {base["id"]: base}, domain, [base])
    wrong["exploration_tier"] = "bold"
    assert (
        scale.resolve_action(wrong, {base["id"]: base}, domain, [base])["task_type"]
        == "optimization"
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            action(base), scale.action_schema({base["id"]: base}, domain)
        )


def test_new_release_has_dynamic_default_and_explicit_intensity_inputs(
    tmp_path, monkeypatch
):
    from agenticgen.uniguide.tool_runtime import configure, assets
    from agenticgen.uniguide import linear_council
    from test_linear_guidance import prepared, action

    c, subject, old, base, executor, calls = prepared(tmp_path, monkeypatch)
    c = configure(
        dict(
            c,
            protocol=coordinates.POLICY,
            agent_tools=dict(c["agent_tools"], skill_release="2.7.0"),
        )
    )
    assert c["atom_assignment_policy"] == coordinates.DYNAMIC_ASSIGNMENT
    body = {k: v for k, v in old.items() if k != "domain_sha256"}
    body.update(
        exploration_policy=c["guidance_exploration"],
        confirmed_effective_levels=[10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    domain = scale.seal(body)
    skills = tmp_path / "skills"
    for name, path in assets(c).items():
        (skills / name).parent.mkdir(parents=True, exist_ok=True)
        (skills / name).write_bytes(path.read_bytes())
    proposals = [
        dict(action(base, updates, tier), exploration_tier=tier)
        for updates, tier in [
            (((0, 1),), "focused"),
            (((0, 2), (1, -2)), "expanded"),
            (((0, 3), (1, -3)), "bold"),
        ]
    ]
    invocations = []

    def provider(messages, schema, output, **kwargs):
        invocations.append(messages)
        if "proposals" in schema["properties"]:
            answer = dict(
                decision="propose",
                reason="Potential improvements at three jump intensities.",
                proposals=proposals,
            )
        else:
            answer = dict(
                decision="propose",
                reason="Choose optimization opportunities, no directional quota.",
                selected=["bold", "expanded"],
            )
        kwargs["response_validator"](answer)
        write(output, answer)
        return answer

    result = linear_council.advise(
        c, [base], domain, tmp_path / "advice", skills, provider=provider
    )
    assert [p["exploration_tier"] for p in result["actions"]] == ["bold", "expanded"]
    assert len(invocations) == 2
    import json

    payload = json.loads(invocations[0][1]["content"])
    assert len(payload["exploration_hypotheses"]) == 3
    assert payload["working_neighbourhood_is_a_preference_not_an_admission_limit"]
    assert "falsifier" not in str(
        read(tmp_path / "advice/proposal.prompt_source.json")["response_schema"]
    )
