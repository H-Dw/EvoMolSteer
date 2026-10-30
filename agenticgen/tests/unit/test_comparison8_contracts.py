"""Optional native prefix, variable windows, checkpoint learning and queue isolation."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace, MethodType
from typing import Optional

import jsonschema
import pytest
import torch

from agenticgen.uniguide import guidance_windows as bounds
from agenticgen.uniguide import linear_scale as scale
from agenticgen.uniguide import checkpoint_learning as learning
from agenticgen.uniguide import prefix_inpainting as prefix
from agenticgen.uniguide.geometry_sampling import GeometryControl, scale_binding
from agenticgen.uniguide.geometry_dynamics import policy as dynamics
from agenticgen.uniguide.io import value_sha
from test_absolute_coordinate_protocol import condition as condition
from test_dynamic_matching_jump import dynamic_condition as dynamic_condition


@pytest.mark.parametrize("windows", [bounds.DEFAULT, bounds.FIVE])
def test_windows_cover_same_guidance_steps_and_execute_exact_boundaries(windows):
    schedule = list(range(10, 10 + len(windows)))
    config = dict(windows=[list(w) for w in windows])
    expanded = bounds.expand(schedule, config)
    assert expanded[:10] == [0.0] * 10 and len(expanded) == 100
    for strength, (lo, hi) in zip(schedule, windows):
        assert expanded[lo : hi + 1] == [strength] * (hi - lo + 1)
    actual = scale_binding(schedule, window_bounds=config["windows"])
    c = object.__new__(GeometryControl)
    c.scale_binding = actual
    assert c.expand_schedule(schedule) == expanded


@pytest.mark.parametrize(
    "windows", [[], [[10, 20], [20, 99]], [[11, 99]], [[10, 98]], [[10.0, 99]]]
)
def test_invalid_window_interfaces_stop_before_sampling(windows):
    with pytest.raises(ValueError):
        bounds.windows(dict(windows=windows))


def test_five_window_domain_schema_action_and_disabled_output(tmp_path):
    from test_dynamic_matching_jump import jump_domain
    from test_linear_guidance import row, action

    _, domain = jump_domain(tmp_path)
    body = {k: v for k, v in domain.items() if k != "domain_sha256"}
    body["guidance_windows"] = bounds.windows(dict(windows=bounds.FIVE))
    body["initial_guidance_scale_tick_schedule"] = [0] * 5
    domain = scale.seal(body)
    base = row(50)
    base[scale.EFFECTIVE] = [50] * 5
    proposal = dict(action(base, ((4, 1),)), exploration_tier="focused")
    jsonschema.validate(proposal, scale.action_schema({base["id"]: base}, domain))
    output = scale.resolve_action(proposal, {base["id"]: base}, domain, [base])
    assert output[scale.EFFECTIVE] == [50, 50, 50, 50, 60]
    assert scale.effective([0] * 5, domain, False) == [0] * 5
    assert len(output["guidance_windows"]) == 5
    with pytest.raises(ValueError):
        scale.encode([50] * 4, domain)


def measured(name, error):
    from test_linear_guidance import row
    from agenticgen.uniguide.coordinate_statistics import RANKING
    from agenticgen.uniguide.absolute_coordinate import RMSD

    result = row(50)
    result.update(id=name, **{RANKING: error, RMSD: error})
    return result


def test_two_round_declines_rebase_best_with_original_bad_paths():
    config = dict(checkpoint_learning=dict(enabled=True))
    baseline = measured("baseline", 0.3)
    a, b = measured("a", 0.4), measured("b", 0.5)
    rounds = [
        dict(iteration=i, groups=[r], checkpoint=learning.checkpoint([r], baseline))
        for i, r in enumerate((a, b), 1)
    ]
    context = learning.context(config, [baseline, a, b], rounds)
    assert (
        context["rebase_from_best_checkpoint"]
        and context["best_checkpoint_id"] == "baseline"
    )
    assert len(context["unsuccessful_recent_paths"]) == 2
    assert "performed poorly" in context["instruction"]
    rounds.append(
        dict(iteration=3, groups=[], checkpoint=learning.checkpoint([], baseline))
    )
    assert not learning.context(config, [baseline, a, b], rounds)[
        "rebase_from_best_checkpoint"
    ]


def test_successful_checkpoint_breaks_decline_sequence_and_historical_best_is_not_score():
    baseline = measured("baseline", 0.2)
    groups = [measured("a", 0.4), measured("b", 0.3), measured("c", 0.35)]
    rounds = [
        dict(iteration=i, groups=[r], checkpoint=learning.checkpoint([r], baseline))
        for i, r in enumerate(groups, 1)
    ]
    assert not learning.context(
        dict(checkpoint_learning=dict(enabled=True)), [baseline, *groups], rounds
    )["rebase_from_best_checkpoint"]
    assert rounds[0]["checkpoint"]["performance"] == 0.4
    assert rounds[0]["checkpoint"]["best_compatible_performance"] == 0.2


def native_function(path, name, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = dict(scope)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def flowr_reference():
    import importlib.util

    return Path(importlib.util.find_spec("flowr").submodule_search_locations[0])


def native_model():
    root = flowr_reference()
    scope = dict(torch=torch, Optional=Optional, _T=torch.Tensor)
    functional = root / "util/functional.py"
    scope["one_hot_encode_tensor"] = native_function(
        functional, "one_hot_encode_tensor", scope
    )
    symmetry = native_function(functional, "symmetrize_bonds", scope)
    scope["smolF"] = SimpleNamespace(symmetrize_bonds=symmetry)
    builder = SimpleNamespace()
    path = root / "models/mol_builder.py"
    builder.inpaint_molecule = MethodType(
        native_function(path, "inpaint_molecule", scope), builder
    )
    builder._inpaint_times = MethodType(
        native_function(path, "_inpaint_times", scope), builder
    )
    model = SimpleNamespace(
        builder=builder, inpainting_mode=False, graph_inpainting=False, coord_scale=2.0
    )
    model._get_times = MethodType(
        native_function(root / "models/fm_pocket.py", "_get_times", scope), model
    )
    return model


def test_native_joint_prefix_and_exact_step_10_cutoff(dynamic_condition, tmp_path):
    from test_geometry_chemical_state import fixture
    from agenticgen.uniguide.geometry_chemistry import match_fractions

    c, state, _ = fixture(tmp_path)
    c.model.builder = native_model().builder
    c.model._get_times = MethodType(native_model()._get_times.__func__, c.model)
    c.model.inpainting_mode = c.model.graph_inpainting = False
    c.model.coord_scale = 2.0
    c.model._adaptive = c
    c.absolute_config = dict(prefix_inpainting=dict(enabled=True))
    c.folder = tmp_path / "prefix"
    c.geometry_condition = dynamic_condition
    c.absolute_state = SimpleNamespace(dynamic=True)
    c.com = torch.zeros(2, 3)
    c.resume = None
    c.configure_chemistry()
    c.configure_prefix()
    before = copy.deepcopy(state)
    rng = torch.get_rng_state().clone()
    c.prepare_prefix_forward(state, 0)
    assert "fragment_mask" in state and not torch.equal(
        state["coords"], before["coords"]
    )
    c.active_step = 0
    times = [torch.zeros(2), torch.zeros(2), torch.zeros(2)]
    packed = c.model._get_times(times, state["mask"], state["fragment_mask"])
    assert all((v.squeeze(-1)[state["fragment_mask"]] == 1).all() for v in packed)
    assert not c.model.inpainting_mode  # No native loop-wide flag is left enabled.
    c.last_assignment = torch.tensor([[1, 2]] * 2)
    projected = c.apply_prefix_inpainting(state, 9)
    assert all(
        v == [1.0, 1.0]
        for v in match_fractions(
            projected, c.last_assignment, c.chemical_features
        ).values()
    )
    c.prepare_prefix_forward(projected, 10)
    assert "fragment_mask" not in projected and "fragment_mode" not in projected
    unchanged = projected["coords"].clone()
    assert c.apply_prefix_inpainting(projected, 10) is projected
    assert torch.equal(unchanged, projected["coords"])
    assert projected["self_condition"] == before["self_condition"]
    assert torch.equal(rng, torch.get_rng_state())
    assert prefix.policy({})["enabled"] is False


def test_joint_prefix_is_bound_separately_from_baseline_and_cannot_enable_after_cutoff():
    from agenticgen.uniguide.absolute_coordinate import POLICY

    base = dict(protocol=POLICY, output_policy="raw999")
    a, b = dynamics(base), dynamics(dict(base, prefix_inpainting=dict(enabled=True)))
    assert value_sha(a) != value_sha(b)
    assert not a["coordinate_projection"] and b[
        "coordinate_projection_step_bounds"
    ] == [0, 9]


def test_eight_round_budget_scales_with_teachers_and_selected_actions(tmp_path):
    from agenticgen.uniguide.linear_calibration import ensure_limits

    c = dict(root=str(tmp_path), rounds=8, guidance_control=dict(maximum_selected=3))
    limits = ensure_limits(c, 2)
    assert limits["stage_caps"] == dict(calibration=700, formal=2400)


def test_failed_teacher_task_does_not_cancel_later_experiments():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/20261031_guidance_comparison8/run_suite.py"
    )
    spec = importlib.util.spec_from_file_location("comparison_suite_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed, saved = [], []

    def run(job):
        observed.append(job["id"])
        if job["id"] == "baseline_failed":
            raise RuntimeError("engineering fixture")
        return dict(completed=True)

    jobs = [dict(id=name) for name in ("baseline_failed", "prefix", "five_windows")]
    result = module.execute_sequence(jobs, run, lambda job, row: saved.append(row))
    assert observed == [job["id"] for job in jobs]
    assert [r["status"] for r in result] == ["failed_skipped", "completed", "completed"]
    assert len(saved) == 3 and result[0]["source_artifacts_preserved"]


def test_decline_warning_and_best_anchor_reach_the_actual_provider_prompt(
    tmp_path, monkeypatch
):
    import json
    from test_linear_guidance import prepared
    from agenticgen.uniguide.tool_runtime import configure, assets
    from agenticgen.uniguide.absolute_coordinate import POLICY
    from agenticgen.uniguide import linear_council
    from agenticgen.uniguide.io import write

    c, subject, old, base, executor, _ = prepared(tmp_path, monkeypatch)
    bad = executor(
        None,
        c,
        subject["teacher"],
        subject["condition"],
        [60] * 4,
        tmp_path / "bad",
        "formal",
        50,
    )
    c = configure(
        dict(
            c,
            protocol=POLICY,
            agent_tools=dict(c["agent_tools"], skill_release="2.8.0"),
        )
    )
    domain = scale.seal(
        {
            **{k: v for k, v in old.items() if k != "domain_sha256"},
            "exploration_policy": c["guidance_exploration"],
        }
    )
    guidance = dict(
        rebase_from_best_checkpoint=True,
        best_checkpoint_id=base["id"],
        instruction="Previous optimization directions performed poorly; restart from the best checkpoint.",
    )
    c["guidance_checkpoint_context"] = guidance
    skills = tmp_path / "skills"
    for name, path in assets(c).items():
        (skills / name).parent.mkdir(parents=True, exist_ok=True)
        (skills / name).write_bytes(path.read_bytes())
    calls = []

    def provider(messages, schema, output, **options):
        payload = json.loads(messages[1]["content"])
        calls.append(payload)
        if "proposals" in schema["properties"]:
            assert schema["properties"]["proposals"]["items"]["properties"][
                "anchor_id"
            ]["enum"] == [base["id"]]
            answer = dict(decision="hold", reason="No new path selected.", proposals=[])
        else:
            answer = dict(decision="hold", reason="Keep measured best.", selected=[])
        options["response_validator"](answer)
        write(output, answer)
        return answer

    linear_council.advise(
        c, [base, bad], domain, tmp_path / "advice", skills, provider=provider
    )
    assert len(calls) == 2
    assert all(p["checkpoint_learning"] == guidance for p in calls)
    assert all(p["allowed_anchor_ids"] == [base["id"]] for p in calls)
