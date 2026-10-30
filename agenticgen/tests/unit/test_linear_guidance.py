"""Control-contract tests: real disk receipts and budgets, no API or model sampling."""

import pytest

from agenticgen.uniguide import linear_scale as scale
from agenticgen.uniguide import linear_calibration as calibration
from agenticgen.uniguide import linear_runtime as runtime
from agenticgen.uniguide.absolute_coordinate import RMSD, MAX_ERROR
from agenticgen.uniguide.coordinate_statistics import RANKING, aggregate_fields
from agenticgen.uniguide.io import Budget, frozen, read, write, sha, value_sha


def config(root, **settings):
    return scale.configure(
        dict(
            root=str(root),
            rounds=3,
            llm={},
            agent_tools=dict(
                skill_release="2.6.0",
                max_model_calls_per_round=30,
                max_output_tokens_per_round=491520,
            ),
            guidance_control=settings,
        )
    )


def row(
    strength, count=50, error=None, target="teacher", complete=True, displacement=None
):
    error = 1.0 / strength if error is None else error
    displacement = strength if displacement is None else displacement
    actual = [strength] * 4 if not isinstance(strength, list) else strength
    return dict(
        id=f"measure/{target}/{actual[0]}/{count}",
        teacher_id=target,
        condition_sha256=target + "condition",
        compatibility_sha256="sampling",
        complete=complete,
        attempts=count,
        **{scale.EFFECTIVE: actual, RMSD: error, MAX_ERROR: error * 2, RANKING: error},
        coordinate_loss_statistics=dict(ranking_eligible=True),
        window_diagnostics=[
            dict(
                max_atom_increment_A=displacement,
                increment_vector_rms_A=displacement / 10,
            )
            for _ in range(4)
        ],
    )


def domain(root, target="teacher"):
    settings = config(root)["guidance_control"]
    return calibration.derive_domain(
        [row(s, target=target) for s in [30, 40, 50]],
        target,
        target + "condition",
        "sampling",
        settings,
    )


def action(anchor, windows=((0, 1),), identifier="candidate"):
    return dict(
        id=identifier,
        anchor_id=anchor["id"],
        updates=[dict(window_id=i, step=s) for i, s in windows],
        evidence_ids=[anchor["id"]],
        improvement_reason="A new nearby allocation may improve the measured objective.",
        uncertainty="Unmeasured allocation; no outcome claimed.",
    )


def test_linear_ticks_switch_no_exponent_and_no_artificial_tick_interval(tmp_path):
    d = domain(tmp_path)
    assert scale.effective([-1, 0, 1, 2], d) == [40, 50, 60, 70]
    assert scale.effective([0] * 4, d) == [50] * 4
    assert scale.effective([0] * 4, d, False) == [0.0] * 4
    assert scale.effective([15] * 4, d) == [200] * 4
    assert scale.encode([10, 50, 100, 200], d) == [-4, 0, 5, 15]
    from agenticgen.uniguide.geometry_sampling import scale_binding

    binding = scale_binding([-1, 0, 1, 2], d)
    assert binding[scale.EFFECTIVE] == [40, 50, 60, 70]
    assert "normalized_guidance_scale_schedule" not in binding
    assert "calibrated_guidance_multiplier" not in binding


@pytest.mark.parametrize(
    "ticks", [[0.5, 0, 0, 0], [True, 0, 0, 0], [-5, 0, 0, 0], [16, 0, 0, 0]]
)
def test_invalid_linear_values_are_rejected_not_clamped(tmp_path, ticks):
    with pytest.raises(ValueError):
        scale.effective(ticks, domain(tmp_path))


def test_teacher_reference_and_protocol_binding_are_isolated(tmp_path):
    d = domain(tmp_path)
    with pytest.raises(ValueError, match="Another"):
        scale.validate_domain(d, "other")
    with pytest.raises(ValueError, match="Another"):
        scale.validate_domain(d, sampling_protocol_sha256="old")
    broken = {k: v for k, v in d.items() if k != "domain_sha256"}
    broken["calibrated_guidance_multiplier"] = 50
    with pytest.raises(ValueError, match="Exponent"):
        scale.validate_domain(scale.seal(broken))


def test_teacher_ranges_are_not_averaged(tmp_path):
    c = config(
        tmp_path,
        teacher_overrides={
            "left": dict(search_maximum=80, initial_scales=[20, 40, 60]),
            "right": dict(search_minimum=30, initial_scales=[40, 60, 80]),
        },
    )
    assert calibration.plan(c, "left")["control_settings"]["search_maximum"] == 80
    assert calibration.plan(c, "right")["control_settings"]["search_minimum"] == 30
    with pytest.raises(ValueError):
        config(tmp_path, initial_scales=[45])


def test_upper_neighbours_are_prioritized_over_lower_support(tmp_path):
    settings = config(tmp_path)["guidance_control"]
    rows = [row(s, 10, error=abs(s - 60) / 100 + 0.1) for s in [20, 40, 60, 80, 100]]
    assert calibration.next_screen(rows, settings) == 70
    assert calibration.promotion_levels(rows + [row(70, 10, error=0.2)], settings)[
        :3
    ] == [60, 70, 80]


def test_boundary_best_does_not_terminate_or_define_the_search_cap(tmp_path):
    d = domain(tmp_path)
    assert d[scale.REFERENCE] == 50
    assert d["boundary_status"] == "coverage_limited"
    assert d["control_settings"]["search_maximum"] == 200
    assert scale.boundary_status(d, row(50))[0]["upper_status"] == "coverage_limited"


def test_integer_moves_joint_windows_and_frontier_classification(tmp_path):
    d = domain(tmp_path)
    anchor = row(50)
    anchors = {anchor["id"]: anchor}
    resolved = scale.resolve_action(
        action(anchor, ((0, 1), (1, 2))), anchors, d, [anchor]
    )
    assert resolved[scale.EFFECTIVE] == [60, 70, 50, 50]
    assert resolved["task_type"] == "range_evidence"
    assert resolved["direction"] == "increase"
    mixed = scale.resolve_action(
        action(anchor, ((0, -1), (1, 1))), anchors, d, [anchor]
    )
    assert mixed["direction"] == "mixed"
    assert mixed[scale.EFFECTIVE] == [40, 60, 50, 50]
    far = scale.resolve_action(action(anchor, ((0, 4),)), anchors, d, [anchor])
    assert far[scale.EFFECTIVE] == [90, 50, 50, 50]


@pytest.mark.parametrize(
    "updates",
    [((0, 0.5),), ((0, True),), ((0, 5),), ((0, 4), (1, 4)), ((0, 1), (0, 1))],
)
def test_invalid_action_changes(tmp_path, updates):
    d = domain(tmp_path)
    anchor = row(50)
    with pytest.raises(ValueError):
        scale.resolve_action(
            action(anchor, updates), {anchor["id"]: anchor}, d, [anchor]
        )


def test_duplicates_failures_and_profile_scope(tmp_path):
    d = domain(tmp_path)
    anchor = row(50)
    a = action(anchor)
    proposal = scale.resolve_action(a, {anchor["id"]: anchor}, d, [anchor])
    with pytest.raises(ValueError, match="already"):
        scale.resolve_action(a, {anchor["id"]: anchor}, d, [anchor, proposal])
    with pytest.raises(ValueError, match="failed"):
        scale.resolve_action(
            a,
            {anchor["id"]: anchor},
            d,
            [anchor],
            [dict(proposal, failure={"error": "nonfinite"})],
        )
    observed = dict(anchor, id="mixed", **{scale.EFFECTIVE: [60, 50, 50, 50]})
    revised = scale.revise(d, observed)
    assert revised[scale.REFERENCE] == d[scale.REFERENCE]
    assert revised["confirmed_effective_levels"] == [30, 40, 50]
    moved = scale.resolve_action(
        action(observed, ((1, 1),)), {"mixed": observed}, revised, [observed]
    )
    assert (
        moved["task_type"] == "range_evidence"
    )  # 60 in window zero does not confirm window one.


def install_fake(monkeypatch, root, *, failed=None, errors=None):
    from agenticgen.uniguide import geometry_sampling

    def binding(config, teacher, condition):
        from agenticgen.uniguide.coordinate_statistics import policy

        return dict(
            teacher_id=teacher["target"],
            condition_sha256=condition["condition_sha256"],
            linear=True,
            coordinate_statistics_policy=policy(config),
        )

    monkeypatch.setattr(geometry_sampling, "compatibility", binding)
    monkeypatch.setattr(runtime, "compatibility", binding)
    measured = {}
    calls = []

    def executor(
        model, c, teacher, condition, schedule, folder, stage, count=50, domain=None
    ):
        actual = scale.effective(schedule, domain) if domain else schedule
        key = (teacher["target"], tuple(actual))
        prior = measured.get(key, 0)
        calls.append((teacher["target"], actual, count, stage))
        if count > prior:
            Budget(root).reserve(
                folder.relative_to(root).as_posix(),
                stage,
                count - prior,
                value_sha([key, count]),
            )
        measured[key] = count
        if failed and failed(teacher["target"], actual, count):
            raise FloatingPointError("injected numeric failure")
        error = errors(actual, count) if errors else (actual[0] - 60) ** 2 / 10000 + 0.2
        records = [
            dict(
                node_id=f"{seed}_{slot}",
                logical_seed=seed,
                logical_slot=slot,
                tensor_row=seed * (count // 5) + slot,
                **{RMSD: error, MAX_ERROR: error * 2},
            )
            for seed in range(5)
            for slot in range(count // 5)
        ]
        result = row(actual[0], count, error, target=teacher["target"])
        identity = dict(
            compatibility=binding(c, teacher, condition),
            **{scale.EFFECTIVE: actual},
            population=count,
        )
        result.update(
            id=folder.relative_to(root).as_posix(),
            compatibility_sha256=value_sha(identity["compatibility"]),
            fingerprint=value_sha(identity),
            **{scale.EFFECTIVE: actual},
        )
        result.update(aggregate_fields(records, c))
        write(folder / "records.json", records)
        write(folder / "aggregate.json", result)
        write(folder / "manifest.json", dict(identity=identity))
        write(folder / "steps.json", [])
        write(
            folder / "COMPLETE.json",
            dict(
                files={
                    name: sha(folder / name)
                    for name in [
                        "records.json",
                        "aggregate.json",
                        "manifest.json",
                        "steps.json",
                    ]
                }
            ),
        )
        write(
            root / "measurement_registry" / (result["fingerprint"] + ".json"),
            dict(identity=identity, source_group=result["id"]),
        )
        return result

    return executor, calls, binding


def test_linear_search_freezes_once_reuses_slots_and_reserves_range_budget(
    tmp_path, monkeypatch
):
    executor, calls, _ = install_fake(monkeypatch, tmp_path)
    c = config(tmp_path)
    subjects = [
        dict(teacher=dict(target=t), condition=dict(condition_sha256=t + "condition"))
        for t in ["left", "right"]
    ]
    domains = calibration.run(c, subjects, lambda *a: None, executor)
    assert set(domains) == {"left", "right"}
    assert all(d[scale.REFERENCE] == 60 for d in domains.values())
    assert all(
        70 in d["confirmed_effective_levels"] and 80 in d["confirmed_effective_levels"]
        for d in domains.values()
    )
    assert all(calibration.used(tmp_path, t) <= 250 for t in domains)
    before = len(calls)
    assert (
        calibration.run(
            c, subjects, lambda *a: pytest.fail("reference reloaded"), executor
        )
        == domains
    )
    assert len(calls) == before
    # Explicit calibration reuse writes the generic completion envelope.
    write(
        tmp_path / "calibration/COMPLETE.json",
        dict(
            domain_sha256=value_sha(domains),
            calibration_keys={
                target: d["calibration_key"] for target, d in domains.items()
            },
        ),
    )
    assert (
        calibration.run(
            c, subjects, lambda *a: pytest.fail("imported reference reloaded"), executor
        )
        == domains
    )
    assert len(calls) == before
    for target in domains:
        rows = [call for call in calls if call[0] == target]
        assert len({(tuple(r[1]), r[2]) for r in rows}) == len(rows)


def prepared(tmp_path, monkeypatch, errors=lambda actual, count: 0.4, failed=None):
    executor, calls, binding = install_fake(
        monkeypatch, tmp_path, errors=errors, failed=failed
    )
    c = config(tmp_path)
    subject = dict(
        teacher=dict(target="teacher"),
        condition=dict(condition_sha256="teachercondition"),
    )
    bind = binding(c, subject["teacher"], subject["condition"])
    frozen(
        tmp_path / "ATTEMPT_LIMITS.json",
        dict(
            stage_caps=dict(calibration=350, formal=450),
            per_teacher_stage_caps=dict(calibration=350),
        ),
    )
    write(
        tmp_path / "BUDGET_SCOPE.json", dict(stage="calibration", teacher_id="teacher")
    )
    base = executor(
        None,
        c,
        subject["teacher"],
        subject["condition"],
        [50] * 4,
        tmp_path / "baseline",
        "calibration",
        50,
    )
    d = calibration.derive_domain(
        [base], "teacher", "teachercondition", value_sha(bind), c["guidance_control"]
    )
    return c, subject, d, base, executor, calls


def test_range_evidence_promotes_nested_slots_once_and_replays_no_sampling(
    tmp_path, monkeypatch
):
    c, s, d, base, executor, calls = prepared(
        tmp_path, monkeypatch, errors=lambda a, n: 0.4 if a == [50] * 4 else 0.3
    )
    resolved = scale.resolve_action(action(base), {base["id"]: base}, d, [base])
    folder = tmp_path / "range_evidence/teacher/one"
    result = runtime.range_evidence(c, s, d, resolved, base, None, executor, folder)
    assert result["status"] == "full_population_complete"
    assert result["charged_attempts"] == 50
    assert [r[2] for r in calls[-2:]] == [20, 50]
    revised = scale.revise(d, result["row"])
    assert revised[scale.REFERENCE] == 50 and revised["evidence_revision"] == 1
    before = len(calls)
    assert (
        runtime.range_evidence(c, s, d, resolved, base, None, executor, folder)
        == result
    )
    assert len(calls) == before
    assert runtime.range_allowance(c, "teacher") == dict(
        remaining_attempts=50, remaining_configurations=1
    )


def test_measured_poor_frontier_stops_at_20_without_replacement(tmp_path, monkeypatch):
    c, s, d, base, executor, calls = prepared(
        tmp_path, monkeypatch, errors=lambda a, n: 0.4 if a == [50] * 4 else 0.8
    )
    resolved = scale.resolve_action(action(base), {base["id"]: base}, d, [base])
    result = runtime.range_evidence(
        c, s, d, resolved, base, None, executor, tmp_path / "range_evidence/teacher/one"
    )
    assert result["row"] is None and result["charged_attempts"] == 20
    assert [r[2] for r in calls] == [50, 20]
    assert runtime.range_allowance(c, "teacher")["remaining_attempts"] == 80


def test_no_budget_and_interrupted_claim_do_not_start_sampling(tmp_path, monkeypatch):
    c, s, d, base, executor, calls = prepared(tmp_path, monkeypatch)
    Budget(tmp_path).reserve("spent", "calibration", 260, "spent")
    resolved = scale.resolve_action(action(base), {base["id"]: base}, d, [base])
    result = runtime.range_evidence(
        c,
        s,
        d,
        resolved,
        base,
        None,
        executor,
        tmp_path / "range_evidence/teacher/no_budget",
    )
    assert result["status"] == "budget_limited" and len(calls) == 1
    write(tmp_path / "range_evidence/teacher/interrupted/CLAIM.json", {})
    with pytest.raises(RuntimeError, match="Interrupted"):
        runtime.range_allowance(c, "teacher")


def test_population_promotion_uses_same_raw_slots_and_retains_low_values(tmp_path):
    c = config(tmp_path)
    reference = [dict(logical_seed=0, logical_slot=i, **{RMSD: 1.0}) for i in range(50)]
    candidate = [
        dict(logical_seed=0, logical_slot=i, **{RMSD: 0.8 if i else 0.001})
        for i in range(20)
    ]
    write(tmp_path / "ref/records.json", reference)
    write(tmp_path / "candidate/records.json", candidate)
    result = runtime.promotion_evidence(c, tmp_path / "ref", tmp_path / "candidate")
    assert (
        result["population"] == 20
        and result["retained_intersection"] == 20
        and result["promote"]
    )
    candidate[0]["logical_slot"] = 99
    write(tmp_path / "candidate/records.json", candidate)
    with pytest.raises(ValueError, match="nested"):
        runtime.promotion_evidence(c, tmp_path / "ref", tmp_path / "candidate")


def test_unified_council_can_select_three_increases_and_has_no_mechanism_fields(
    tmp_path, monkeypatch
):
    from agenticgen.uniguide import linear_council
    from agenticgen.uniguide.tool_runtime import assets
    from agenticgen.uniguide.absolute_coordinate import POLICY
    import jsonschema

    c, s, d, base, executor, calls = prepared(tmp_path, monkeypatch)
    write(
        tmp_path / "conditions/teacher.json",
        dict(policy=POLICY, condition_sha256="teachercondition"),
    )
    skills = tmp_path / "skills"
    for name, path in assets(c).items():
        write_text = path.read_bytes()
        (skills / name).parent.mkdir(parents=True, exist_ok=True)
        (skills / name).write_bytes(write_text)
    invoked = []
    proposals = [action(base, ((i, 1),), f"up_{i}") for i in range(3)]

    def provider(messages, schema, output, **options):
        invoked.append(messages)
        if "proposals" in schema["properties"]:
            answer = dict(
                decision="propose",
                reason="Three distinct paths may improve the incumbent.",
                proposals=proposals,
            )
        else:
            answer = dict(
                decision="propose",
                reason="Select by improvement opportunity, without direction quotas.",
                selected=[p["id"] for p in proposals],
            )
        jsonschema.validate(answer, schema)
        options["response_validator"](answer)
        write(output, answer)
        return answer

    result = linear_council.advise(
        c, [base], d, tmp_path / "advice", skills, provider=provider
    )
    assert len(result["actions"]) == 3 and len(invoked) == 2
    assert all(
        scale.resolve_action(p, {base["id"]: base}, d, [base])["direction"]
        == "increase"
        for p in result["actions"]
    )
    source = read(tmp_path / "advice/proposal.prompt_source.json")
    assert "mechanism_uncertainty" not in str(source["response_schema"])
    assert "falsifier" not in str(source["response_schema"])
    assert len(source["messages"][0]["content"]) < 12000


def test_variable_rounds_hold_keeps_calibration_incumbent_and_replay_is_idle(
    tmp_path, monkeypatch
):
    from agenticgen.uniguide import absolute_runtime

    c, s, d, base, executor, calls = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime, "source_binding", lambda: {"control": value_sha(c["guidance_control"])}
    )
    write(
        tmp_path / "ATTEMPT_LIMITS.json",
        dict(
            policy=scale.POLICY,
            stage_caps=dict(calibration=350, formal=450),
            per_teacher_stage_caps=dict(calibration=350),
        ),
    )
    key = d["calibration_key"]
    folder = tmp_path / "calibration/frozen" / key
    write(folder / "domain.json", d)
    write(
        folder / "COMPLETE.json",
        dict(
            domain_sha256=d["domain_sha256"],
            measurement_receipts={
                base["id"]: sha(tmp_path / base["id"] / "COMPLETE.json")
            },
        ),
    )
    write(tmp_path / "calibration/domains.json", {"teacher": d})
    monkeypatch.setattr(absolute_runtime, "calibrated", lambda c: [s])
    advice_calls = []
    retained = []

    def advisor(c, h, domain, folder, skills, failures=(), final=False):
        advice_calls.append(final)
        for role in ("proposal", "synthesis"):
            write(
                folder / (role + ".json"),
                dict(decision="hold", reason="Keep the best measured allocation."),
            )
        return dict(decision="hold", reason="No better candidate selected.", actions=[])

    result = runtime.run_version(
        c,
        "GD01",
        lambda *a: None,
        executor,
        advisor,
        lambda root, groups, winner, folder: retained.append(
            (len(groups), winner["id"])
        ),
    )
    assert result["current"]["completed_rounds"] == 3
    assert (
        result["current"]["held_rounds"] == 3
        and result["current"]["requested_groups"] == 0
    )
    assert (
        result["current"]["complete"]
        and result["current"]["teachers"]["teacher"]["representative"] == base["id"]
    )
    assert result["current"]["teachers"]["teacher"]["best_formal_group"] is None
    assert retained == [(1, base["id"])] and advice_calls == [False, False, False, True]
    before = len(calls)
    assert (
        runtime.run_version(
            c,
            "GD01",
            lambda *a: pytest.fail("reload"),
            executor,
            advisor,
            lambda *a: None,
        )
        == result
    )
    assert len(calls) == before and len(advice_calls) == 4


def test_linear_initial_selection_and_old_exponent_interface_are_separate(tmp_path):
    from agenticgen.uniguide.geometry_reuse import select_initial, verify_initial

    d = domain(tmp_path)
    write(tmp_path / "calibration/domains.json", {"teacher": d})
    payload = select_initial(config(tmp_path), {"teacher": [-2, -1, 0, 0]})
    assert payload["scale_units"] == "linear_reference_ticks"
    assert payload["teachers"]["teacher"][scale.EFFECTIVE] == [30, 40, 50, 50]
    assert verify_initial(tmp_path, {"teacher": d}) == payload["teachers"]
    assert (
        select_initial(config(tmp_path), {"teacher": [30, 40, 50, 50]}, "actual")
        == payload
    )
    with pytest.raises(ValueError, match="constant confirmation"):
        select_initial(config(tmp_path), {"teacher": [1, 0, 0, 0]})


def test_partial_complete_profile_promotes_with_only_30_new_attempts(
    tmp_path, monkeypatch
):
    c, s, d, base, executor, calls = prepared(
        tmp_path, monkeypatch, errors=lambda a, n: 0.4 if a == [50] * 4 else 0.3
    )
    resolved = scale.resolve_action(action(base), {base["id"]: base}, d, [base])
    executor(
        None,
        c,
        s["teacher"],
        s["condition"],
        resolved[scale.EFFECTIVE],
        tmp_path / "partial",
        "calibration",
        20,
    )
    assert (
        runtime.remaining_profile_cost(tmp_path, "teacher", resolved[scale.EFFECTIVE])
        == 30
    )
    evidence = runtime.range_evidence(
        c,
        s,
        d,
        resolved,
        base,
        None,
        executor,
        tmp_path / "range_evidence/teacher/upgrade",
    )
    assert evidence["charged_attempts"] == 30
    assert runtime.range_allowance(c, "teacher")["remaining_attempts"] == 70


def test_runtime_executes_same_direction_portfolio_and_uses_full_history(
    tmp_path, monkeypatch
):
    from agenticgen.uniguide import absolute_runtime
    from agenticgen.uniguide.geometry_protocol import choose

    executor, calls, binding = install_fake(
        monkeypatch,
        tmp_path,
        errors=lambda a, n: 0.4 if a == [50] * 4 else 0.5 if len(set(a)) == 1 else 0.3,
    )
    c = config(tmp_path)
    c["rounds"] = 2
    s = dict(
        teacher=dict(target="teacher"),
        condition=dict(condition_sha256="teachercondition"),
    )
    calibration.ensure_limits(c, 1)
    write(
        tmp_path / "BUDGET_SCOPE.json", dict(stage="calibration", teacher_id="teacher")
    )
    rows = [
        executor(
            None,
            c,
            s["teacher"],
            s["condition"],
            [x] * 4,
            tmp_path / f"base{x}",
            "calibration",
            50,
        )
        for x in (40, 50, 60)
    ]
    base = choose(rows)
    d = calibration.derive_domain(
        rows,
        "teacher",
        "teachercondition",
        value_sha(binding(c, s["teacher"], s["condition"])),
        c["guidance_control"],
    )
    folder = tmp_path / "calibration/frozen" / d["calibration_key"]
    write(folder / "domain.json", d)
    write(
        folder / "COMPLETE.json",
        dict(
            domain_sha256=d["domain_sha256"],
            measurement_receipts={
                r["id"]: sha(tmp_path / r["id"] / "COMPLETE.json") for r in rows
            },
        ),
    )
    write(tmp_path / "calibration/domains.json", {"teacher": d})
    monkeypatch.setattr(absolute_runtime, "calibrated", lambda config: [s])
    monkeypatch.setattr(
        runtime, "source_binding", lambda: {"control": value_sha(c["guidance_control"])}
    )
    invocations = []

    def advisor(c, h, domain, folder, skills, failures=(), final=False):
        invocations.append(final)
        for role in ("proposal", "synthesis"):
            write(folder / (role + ".json"), dict(reason="Offline controller fixture."))
        actions = (
            [action(base, ((i, 1),), f"up_{i}") for i in range(3)]
            if len(invocations) == 1
            else []
        )
        return dict(decision="propose" if actions else "hold", actions=actions)

    result = runtime.run_version(
        c, "GD01", lambda *a: None, executor, advisor, lambda *a: None
    )["current"]
    assert (
        result["completed_rounds"] == 2
        and result["requested_groups"] == 3
        and len(result["groups"]) == 3
    )
    assert {r["selected_proposal"]["direction"] for r in result["groups"]} == {
        "increase"
    }
    assert all(r["task_type"] == "optimization" for r in result["groups"])
    assert result["teachers"]["teacher"][RANKING] == pytest.approx(0.3)
    assert (
        sum(
            r["count"]
            for r in read(tmp_path / "attempts.json").values()
            if r["stage"] == "formal"
        )
        == 150
    )


def test_linear_bindings_reach_the_existing_serial_scheduler_and_reuse(
    tmp_path, monkeypatch
):
    from test_uniguide_geometry_batch import setup_scheduler
    from agenticgen.uniguide import geometry_batch

    c, teacher, condition, model, calls, sample = setup_scheduler(tmp_path, monkeypatch)
    c = scale.configure(c)
    binding = geometry_batch.compatibility(c, teacher, condition)
    rows = [
        dict(
            row(x),
            condition_sha256=condition["condition_sha256"],
            compatibility_sha256=value_sha(binding),
        )
        for x in (30, 40, 50)
    ]
    d = calibration.derive_domain(
        rows,
        "teacher",
        condition["condition_sha256"],
        value_sha(binding),
        c["guidance_control"],
    )
    jobs = [
        dict(schedule=ticks, folder=tmp_path / f"group{i}", stage="formal", domain=d)
        for i, ticks in enumerate(([-1, 0, 0, 0], [0, -1, 0, 0]))
    ]
    measured = geometry_batch.execute_many(
        model, c, teacher, condition, jobs, sample_fn=sample
    )
    assert calls == [50, 50]
    assert [r[scale.EFFECTIVE] for r in measured] == [
        [40, 50, 50, 50],
        [50, 40, 50, 50],
    ]
    assert all(r[scale.REFERENCE] == 50 and r[scale.UNIT] == 10 for r in measured)
    geometry_batch.execute_many(model, c, teacher, condition, jobs, sample_fn=sample)
    assert calls == [50, 50]
    assert sum(r["count"] for r in read(tmp_path / "attempts.json").values()) == 100


def test_new_absolute_defaults_and_explicit_legacy_release_are_distinct(tmp_path):
    from agenticgen.uniguide.absolute_coordinate import POLICY
    from agenticgen.uniguide.tool_runtime import configure

    fresh = configure(dict(root=str(tmp_path / "fresh"), protocol=POLICY))
    assert fresh["agent_tools"]["skill_release"] == "2.7.0" and scale.enabled(fresh)
    old = configure(
        dict(
            root=str(tmp_path / "old"),
            protocol=POLICY,
            agent_tools=dict(skill_release="2.5.0"),
        )
    )
    assert old["agent_tools"]["skill_release"] == "2.5.0" and not scale.enabled(old)
    assert config(tmp_path, initial_scales=[20.0, 40.0, 60.0])["guidance_control"][
        "initial_scales"
    ] == [20, 40, 60]


def test_nonfinite_frontier_remains_charged_and_is_never_retried(tmp_path, monkeypatch):
    c, s, d, base, executor, calls = prepared(
        tmp_path, monkeypatch, failed=lambda t, a, n: a != [50] * 4
    )
    resolved = scale.resolve_action(action(base), {base["id"]: base}, d, [base])
    folder = tmp_path / "range_evidence/teacher/nonfinite"
    result = runtime.range_evidence(c, s, d, resolved, base, None, executor, folder)
    assert (
        result["status"] == "nonfinite_measurement" and result["charged_attempts"] == 20
    )
    assert result["failure_files"] and len(calls) == 2
    assert (
        runtime.range_evidence(c, s, d, resolved, base, None, executor, folder)
        == result
    )
    assert len(calls) == 2
    write(folder / "population_20/CALIBRATION_FAILURE.json", dict(error="overwritten"))
    with pytest.raises(ValueError, match="failure evidence"):
        runtime.range_evidence(c, s, d, resolved, base, None, executor, folder)
