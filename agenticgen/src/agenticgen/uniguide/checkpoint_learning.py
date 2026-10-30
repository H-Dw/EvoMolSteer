"""Measured round performance and an explicit best-checkpoint restart context."""

from .geometry_protocol import choose, primary_score
from .coordinate_statistics import ranking_eligible
from .linear_scale import EFFECTIVE

POLICY = "two-consecutive-round-declines-rebase-best-v1"


def configure(config):
    settings = config.get("checkpoint_learning", {"enabled": False})
    if (
        not isinstance(settings, dict)
        or set(settings) != {"enabled"}
        or type(settings["enabled"]) is not bool
    ):
        raise ValueError("checkpoint_learning requires a boolean enabled field")
    return settings


def checkpoint(groups, incumbent):
    eligible = [
        r
        for r in groups
        if r.get("attempts") == 50
        and ranking_eligible(r)
        and r.get("stability_status") != "large_finite_displacement_growth"
    ]
    winner = choose(eligible) if eligible else None
    best = choose([incumbent, *eligible])
    return dict(
        status="measured" if winner else "no_new_complete_measurement",
        measured_source_id=winner["id"] if winner else None,
        performance=primary_score(winner) if winner else None,
        input_incumbent_performance=primary_score(incumbent),
        best_compatible_source_id=best["id"],
        best_compatible_performance=primary_score(best),
    )


def context(config, history, completed_rounds):
    if not configure(config)["enabled"]:
        return None
    best = choose(history)
    recent = completed_rounds[-3:]
    points = [r.get("checkpoint", {}) for r in recent]
    if (
        len(completed_rounds) == 2
        and points
        and "input_incumbent_performance" in points[0]
    ):
        points.insert(
            0,
            dict(
                status="measured", performance=points[0]["input_incumbent_performance"]
            ),
        )
    if len(points) < 3 or any(p.get("status") != "measured" for p in points):
        rebase = False
    else:
        rebase = (
            points[0]["performance"]
            < points[1]["performance"]
            < points[2]["performance"]
        )
    # Do not repeat the same restart when intervening rounds have no measurements.
    poor = (
        [
            dict(
                iteration=r["iteration"],
                checkpoint=r.get("checkpoint"),
                directions=[
                    dict(
                        source_id=g["id"],
                        actual_schedule=g[EFFECTIVE],
                        proposal=g.get("selected_proposal"),
                    )
                    for g in r["groups"]
                ],
            )
            for r in recent[-2:]
        ]
        if rebase
        else []
    )
    return dict(
        policy=POLICY,
        rebase_from_best_checkpoint=rebase,
        best_checkpoint_id=best["id"],
        best_checkpoint_actual_schedule=best[EFFECTIVE],
        best_checkpoint_performance=primary_score(best),
        consecutive_declines=2 if rebase else 0,
        unsuccessful_recent_paths=poor,
        instruction=(
            (
                "The previous two adjacent checkpoint performances declined. "
                "Those optimization directions performed poorly. Start new hypotheses "
                "from the best compatible checkpoint and explore different promising paths; "
                "do not repeat its generation or design ablations."
            )
            if rebase
            else "Learn all completed results and seek promising paths beyond the current best."
        ),
    )
