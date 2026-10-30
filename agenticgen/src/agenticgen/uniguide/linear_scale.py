"""Linear, teacher-local strength coordinates and bounded allocation actions.

The reference is immutable. Evidence revisions and moving anchor neighbourhoods
are separate objects; neither a zero tick nor a panel edge disables guidance.
"""

import math

from .io import value_sha
from .guidance_windows import windows

POLICY = "linear-envelope-frontier-v1"
TICKS = "guidance_scale_tick_schedule"
REFERENCE = "calibrated_guidance_reference_scale"
UNIT = "guidance_scale_grid_unit"
EFFECTIVE = "effective_guidance_scale_schedule"
DEFAULTS = dict(
    policy=POLICY,
    guidance_scale_grid_unit=10,
    search_minimum=10,
    search_maximum=200,
    initial_scales=[20, 40, 60, 80, 100],
    teacher_overrides={},
    maximum_screening_levels=8,
    maximum_confirmation_levels=4,
    maximum_attempts_per_teacher=350,
    range_evidence_attempts=100,
    maximum_range_evidence_configurations=2,
    working_radius_ticks=2,
    maximum_window_tick_change=4,
    maximum_total_tick_change=6,
    maximum_proposals=6,
    maximum_selected=3,
    improvement_tolerance_angstrom=0.0,
)


def enabled(value):
    return (
        value.get("scale_policy", value.get("guidance_control", {}).get("policy"))
        == POLICY
    )


def integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def configure(config):
    calibration = config.get("calibration", {})
    if (
        set(calibration) - {"search_policy"}
        or calibration.get("search_policy", POLICY) != POLICY
    ):
        raise ValueError(
            "Linear control cannot reuse exponent calibration settings or measurements"
        )
    settings = dict(DEFAULTS, **config.get("guidance_control", {}))
    if set(settings) != set(DEFAULTS) or settings["policy"] != POLICY:
        raise ValueError("Unknown linear guidance control settings")
    for name in (
        UNIT,
        "search_minimum",
        "search_maximum",
        "maximum_screening_levels",
        "maximum_confirmation_levels",
        "maximum_attempts_per_teacher",
        "range_evidence_attempts",
        "maximum_range_evidence_configurations",
        "working_radius_ticks",
        "maximum_window_tick_change",
        "maximum_total_tick_change",
        "maximum_proposals",
        "maximum_selected",
    ):
        integer(settings[name], name)
    if (
        settings["maximum_attempts_per_teacher"] > 350
        or settings["maximum_selected"] > 3
    ):
        raise ValueError("Linear guidance exceeds the declared scientific budget")
    if settings["maximum_confirmation_levels"] > settings["maximum_screening_levels"]:
        raise ValueError("More confirmation levels than screening levels")
    if settings["maximum_window_tick_change"] > 4 or settings["maximum_proposals"] > 6:
        raise ValueError("Action limits exceed the linear JSON interface")
    if settings["maximum_selected"] > settings["maximum_proposals"]:
        raise ValueError("More selections than proposals")
    if settings["range_evidence_attempts"] >= settings["maximum_attempts_per_teacher"]:
        raise ValueError("No budget left for reference calibration")
    tolerance = settings["improvement_tolerance_angstrom"]
    if (
        type(tolerance) not in (int, float)
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("Invalid coordinate improvement tolerance")
    overrides = settings["teacher_overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("Teacher overrides must be a mapping")
    for target in [None, *overrides]:
        local = teacher_settings(settings, target)
        grid(local)
        starts = local["initial_scales"]
        if (
            not isinstance(starts, list)
            or not starts
            or len(set(starts)) != len(starts)
            or len(starts) > settings["maximum_screening_levels"]
        ):
            raise ValueError("Invalid initial linear search grid")
        for scale in starts:
            check_strength(scale, local)
    settings["initial_scales"] = [int(x) for x in settings["initial_scales"]]
    settings["teacher_overrides"] = {
        target: dict(
            values,
            **(
                {"initial_scales": [int(x) for x in values["initial_scales"]]}
                if "initial_scales" in values
                else {}
            ),
        )
        for target, values in overrides.items()
    }
    return dict(
        config, guidance_control=settings, calibration=dict(search_policy=POLICY)
    )


def teacher_settings(settings, target):
    override = settings.get("teacher_overrides", {}).get(target, {})
    if not isinstance(override, dict) or set(override) - {
        "search_minimum",
        "search_maximum",
        "initial_scales",
    }:
        raise ValueError(
            "Only teacher-local search bounds and starts may be overridden"
        )
    return dict(settings, **override, teacher_overrides={})


def grid(settings):
    unit = settings[UNIT]
    lower, upper = settings["search_minimum"], settings["search_maximum"]
    if (
        type(lower) is not int
        or type(upper) is not int
        or lower <= 0
        or upper < lower
        or lower % unit
        or upper % unit
    ):
        raise ValueError("Search bounds must be positive grid multiples")
    return list(range(lower, upper + 1, unit))


def check_strength(value, settings):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value not in grid(settings)
    ):
        raise ValueError("Strength is outside the declared linear grid")


def seal(body):
    return dict(body, domain_sha256=value_sha(body))


def validate_domain(
    domain, teacher_id=None, condition_hash=None, sampling_protocol_sha256=None
):
    if not enabled(domain) or value_sha(
        {k: v for k, v in domain.items() if k != "domain_sha256"}
    ) != domain.get("domain_sha256"):
        raise ValueError("Invalid linear scale domain binding")
    for key, value in (
        ("teacher_id", teacher_id),
        ("condition_sha256", condition_hash),
        ("sampling_protocol_sha256", sampling_protocol_sha256),
    ):
        if value is not None and domain[key] != value:
            raise ValueError(
                "Another teacher, condition or sampling protocol's reference"
            )
    if domain["calibration_key"] != value_sha(
        [
            domain["teacher_id"],
            domain["condition_sha256"],
            domain["sampling_protocol_sha256"],
        ]
    ):
        raise ValueError("Invalid calibration identity")
    if any(
        k in domain
        for k in (
            "calibrated_guidance_multiplier",
            "normalized_minimum",
            "normalized_maximum",
        )
    ):
        raise ValueError("Exponent protocol cannot enter the linear interface")
    settings = domain["control_settings"]
    check_strength(domain[REFERENCE], settings)
    if domain[UNIT] != settings[UNIT] or not domain["confirmed_effective_levels"]:
        raise ValueError("Missing linear confirmation evidence")
    for level in domain["confirmed_effective_levels"]:
        check_strength(level, settings)
    return domain


def encode(actual, domain):
    if not isinstance(actual, (list, tuple)) or len(actual) != len(windows(domain)):
        raise ValueError("Actual strengths must match the configured windows")
    for value in actual:
        check_strength(value, domain["control_settings"])
    return [int(round((value - domain[REFERENCE]) / domain[UNIT])) for value in actual]


def effective(ticks, domain, guidance_enabled=True):
    if (
        not isinstance(ticks, (list, tuple))
        or len(ticks) != len(windows(domain))
        or any(type(t) is not int for t in ticks)
        or type(guidance_enabled) is not bool
    ):
        raise ValueError(
            "Integer linear ticks per window and an independent boolean switch required"
        )
    actual = [domain[REFERENCE] + domain[UNIT] * tick for tick in ticks]
    encode(actual, domain)
    actual = [int(value) for value in actual]
    return actual if guidance_enabled else [0.0] * len(ticks)


def binding(ticks, domain, guidance_enabled=True):
    return {
        TICKS: list(ticks),
        REFERENCE: domain[REFERENCE],
        UNIT: domain[UNIT],
        EFFECTIVE: effective(ticks, domain, guidance_enabled),
        "scale_policy": POLICY,
        "scale_mode": "linear_reference_ticks",
        "guidance_enabled": guidance_enabled,
        "guidance_windows": windows(domain),
    }


def validate_schedule(ticks, domain):
    validate_domain(domain)
    effective(
        ticks, domain
    )  # Missing scalar confirmation is handled by the action/task route.


def action_schema(anchor_ids, domain=None):
    count = len(windows(domain))

    def obj(properties):
        return dict(
            type="object",
            properties=properties,
            required=list(properties),
            additionalProperties=False,
        )

    def text(limit):
        return dict(type="string", minLength=1, maxLength=limit)

    schema = obj(
        dict(
            id=text(80),
            anchor_id=dict(enum=list(anchor_ids)),
            updates=dict(
                type="array",
                minItems=1,
                maxItems=count,
                items=obj(
                    dict(
                        window_id=dict(type="integer", minimum=0, maximum=count - 1),
                        step=dict(type="integer", enum=[-4, -3, -2, -1, 1, 2, 3, 4]),
                    )
                ),
            ),
            evidence_ids=dict(
                type="array", maxItems=4, uniqueItems=True, items=text(200)
            ),
            improvement_reason=text(400),
            uncertainty=text(300),
        )
    )
    if domain and domain.get("exploration_policy"):
        from .exploration import TIERS

        schema["properties"]["exploration_tier"] = dict(enum=list(TIERS))
        schema["required"].append("exploration_tier")
    return schema


def resolve_action(action, anchors, domain, seen=(), failures=()):
    validate_domain(domain)
    if action.get("anchor_id") not in anchors:
        raise ValueError("Unknown completed measured anchor")
    if any(
        k in action
        for k in (
            EFFECTIVE,
            TICKS,
            REFERENCE,
            "normalized_guidance_scale_schedule",
            "calibrated_guidance_multiplier",
            "loss_weights",
            "branch",
        )
    ):
        raise ValueError(
            "Executor-owned fields or obsolete branch/exponent controls in action"
        )
    settings = domain["control_settings"]
    source = anchors[action["anchor_id"]]
    actual = list(source[EFFECTIVE])
    updates = action.get("updates")
    count = len(windows(domain))
    if not isinstance(updates, list) or not 1 <= len(updates) <= count:
        raise ValueError("Updates must fit the configured window count")
    changed_windows, changes = set(), []
    for update in updates:
        if not isinstance(update, dict) or set(update) != {"window_id", "step"}:
            raise ValueError("Invalid window update")
        window, step = update["window_id"], update["step"]
        if (
            type(window) is not int
            or window not in range(count)
            or window in changed_windows
            or type(step) is not int
            or not 1 <= abs(step) <= settings["maximum_window_tick_change"]
        ):
            raise ValueError("Invalid or repeated window / integer change")
        changed_windows.add(window)
        changes.append(step)
        actual[window] += domain[UNIT] * step
    if sum(map(abs, changes)) > settings["maximum_total_tick_change"]:
        raise ValueError("Action exceeds the declared total integer change")
    if domain.get("exploration_policy"):
        from .exploration import classify

        if action.get("exploration_tier") != classify(
            updates, domain["exploration_policy"]
        ):
            raise ValueError(
                "Declared exploration intensity does not match the actual parameter jump"
            )
    ticks = encode(actual, domain)
    profiles = [r.get(EFFECTIVE, r) if isinstance(r, dict) else r for r in seen]
    if actual in profiles:
        raise ValueError("Configuration already measured or reserved")
    blocked = [
        r
        for r in failures
        if r.get("failure")
        or r.get("error")
        or r.get("evidence_status")
        in ("statistics_ineligible", "measured_displacement_growth")
    ]
    if actual in [r[EFFECTIVE] for r in blocked if EFFECTIVE in r]:
        raise ValueError("Exact failed/interrupted configuration cannot be retried")
    known = set(domain["confirmed_effective_levels"])
    needs_evidence = any(
        x not in known and x != source[EFFECTIVE][i] for i, x in enumerate(actual)
    ) or (
        not domain.get("exploration_policy")
        and max(map(abs, changes)) > settings["working_radius_ticks"]
    )
    direction = (
        "increase" if min(changes) > 0 else "decrease" if max(changes) < 0 else "mixed"
    )
    return dict(
        action,
        **binding(ticks, domain),
        direction=direction,
        total_absolute_tick_change=sum(map(abs, changes)),
        task_type="range_evidence" if needs_evidence else "optimization",
        domain_sha256=domain["domain_sha256"],
        calibration_key=domain["calibration_key"],
        executor_resolved=True,
        agent_action_sha256=value_sha(action),
    )


def boundary_status(domain, incumbent):
    settings = domain["control_settings"]
    levels = set(domain["confirmed_effective_levels"])
    return [
        dict(
            window_id=i,
            current_strength=value,
            working_interval=[
                max(
                    settings["search_minimum"],
                    value - domain[UNIT] * settings["working_radius_ticks"],
                ),
                min(
                    settings["search_maximum"],
                    value + domain[UNIT] * settings["working_radius_ticks"],
                ),
            ],
            upper_status=(
                "declared_limit"
                if value == settings["search_maximum"]
                else (
                    "interior_supported"
                    if value + domain[UNIT] in levels
                    else "coverage_limited"
                )
            ),
        )
        for i, value in enumerate(incumbent[EFFECTIVE])
    ]


def revise(domain, row):
    """Append a profile-scoped receipt, never promote a mixed scalar to a universal limit."""
    if not row.get("complete") or row["attempts"] != 50:
        raise ValueError("Only complete full-population evidence can revise a domain")
    if (
        any(row[k] != domain[k] for k in ("teacher_id", "condition_sha256"))
        or row["compatibility_sha256"] != domain["sampling_protocol_sha256"]
    ):
        raise ValueError("Incompatible range evidence")
    from .coordinate_statistics import ranking_eligible

    if (
        not ranking_eligible(row)
        or row.get("stability_status") == "large_finite_displacement_growth"
    ):
        raise ValueError("Ineligible range evidence cannot expand the domain")
    encode(row[EFFECTIVE], domain)
    body = {k: v for k, v in domain.items() if k != "domain_sha256"}
    body["parent_domain_sha256"] = domain["domain_sha256"]
    body["evidence_revision"] = domain.get("evidence_revision", 0) + 1
    body["confirmed_profiles"] = [
        *domain.get("confirmed_profiles", []),
        dict(
            source_id=row["id"],
            **{EFFECTIVE: row[EFFECTIVE]},
            scope="exact_configuration_only",
        ),
    ]
    if len(set(row[EFFECTIVE])) == 1:
        body["confirmed_effective_levels"] = sorted(
            set(body["confirmed_effective_levels"]) | {row[EFFECTIVE][0]}
        )
    return seal(body)
