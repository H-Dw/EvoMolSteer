"""Measured round performance and an explicit best-checkpoint restart context."""

import math

from .geometry_protocol import choose, primary_score
from .coordinate_statistics import ranking_eligible
from .linear_scale import EFFECTIVE, open_domain
from .io import value_sha

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


def checkpoint(groups, incumbent, baseline=None):
    eligible = [
        r
        for r in groups
        if r.get("attempts") == 50
        and ranking_eligible(r)
        and r.get("stability_status") != "large_finite_displacement_growth"
    ]
    winner = choose(eligible) if eligible else None
    best = choose([incumbent, *eligible])
    if baseline is not None and primary_score(best) >= primary_score(incumbent):
        best = incumbent  # A tie is not a new best node.
    result = dict(
        status="measured" if winner else "no_new_complete_measurement",
        measured_source_id=winner["id"] if winner else None,
        performance=primary_score(winner) if winner else None,
        input_incumbent_performance=primary_score(incumbent),
        best_compatible_source_id=best["id"],
        best_compatible_performance=primary_score(best),
    )
    if baseline is not None:
        if primary_score(best) > primary_score(baseline):
            raise ValueError("Global best regressed below its calibration root")
        result.update(calibration_source_id=baseline["id"],
                      calibration_performance=primary_score(baseline),
                      improved_global_best=primary_score(best)<primary_score(incumbent))
    return result


def context(config, history, completed_rounds, domain=None):
    if domain is not None and open_domain(domain):
        return rooted_context(config, history, completed_rounds, domain)
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


def calibration_root(history, domain):
    rows = {r["id"]:r for r in history}
    if len(rows) != len(history) or domain["initial_source"] not in rows:
        raise ValueError("The original calibration root must remain in compatible history")
    root = rows[domain["initial_source"]]
    check_node_scope(root,domain)
    if (not root.get("complete") or root.get("attempts") != 50 or not ranking_eligible(root)
        or root.get("stability_status") == "large_finite_displacement_growth"
        or not math.isfinite(primary_score(root))):
        raise ValueError("The calibration root is not a complete eligible population50")
    return root


def check_node_scope(row,domain):
    if (row.get("teacher_id")!=domain["teacher_id"] or
        row.get("condition_sha256")!=domain["condition_sha256"] or
        row.get("compatibility_sha256")!=domain["sampling_protocol_sha256"]):
        raise ValueError("Checkpoint belongs to another teacher/condition/sampling protocol")


def best_from_calibration(history, domain, completed_rounds=()):
    baseline = calibration_root(history, domain)
    candidates = [r for r in history if r.get("complete") and r.get("attempts")==50
                  and ranking_eligible(r) and r.get("stability_status")!="large_finite_displacement_growth"]
    best = baseline
    if completed_rounds:
        prior = completed_rounds[-1].get("checkpoint", {}).get("best_compatible_source_id")
        for row in candidates:
            if row["id"] == prior and primary_score(row)<primary_score(best):
                best = row
    for row in candidates:
        if primary_score(row)<primary_score(best):
            best = row
    return best


def node_summary(row):
    return dict(source_id=row["id"],actual_schedule=row[EFFECTIVE],performance=primary_score(row),
                population=row["attempts"], fingerprint=row.get("fingerprint"),
                evidence_kind="saved_complete_measurement")


def decision_path(history, domain, best, completed_rounds=()):
    """Only persisted decision summaries, actions and outcomes; no hidden reasoning."""
    rows = {r["id"]:r for r in history}
    decisions = {g["id"]:r.get("decision_summary") for r in completed_rounds for g in r["groups"]}
    path, seen, current = [], set(), best
    while True:
        check_node_scope(current,domain)
        if current["id"] in seen:
            raise ValueError("Cyclic checkpoint decision ancestry")
        seen.add(current["id"])
        proposal = current.get("proposal",current.get("selected_proposal"))
        recorded = next((g.get("selected_proposal") for r in completed_rounds for g in r["groups"]
                         if g["id"]==current["id"]),None)
        if recorded is not None and value_sha(proposal)!=value_sha(recorded):
            raise ValueError("Checkpoint proposal differs from its completed round decision")
        parent = proposal.get("anchor_id") if proposal else None
        node = dict(node_summary(current),parent_source_id=parent,
                    decision_summary=decisions.get(current["id"]),
                    saved_proposal_summary=({k:proposal[k] for k in
                        ("anchor_id","updates","target_scale_schedule","exploration_tier","improvement_reason","uncertainty","evidence_ids")
                        if k in proposal} if proposal else None))
        path.append(node)
        if parent is None:
            if current["id"] not in domain["bootstrap_sources"] and "calibration" not in current["id"].split("/"):
                raise ValueError("Missing measured parent of a non-calibration checkpoint")
            node["path_start"] = "initial_calibration_root" if current["id"]==domain["initial_source"] else "independent_calibration_peer"
            break
        if parent not in rows:
            raise ValueError("Checkpoint parent is absent from compatible measured history")
        node["change_from_parent"] = primary_score(current)-primary_score(rows[parent])
        current = rows[parent]
    return list(reversed(path))


def best_state(learning):
    return dict(calibration_baseline=learning["calibration_baseline"],
                representative=learning["best_checkpoint_id"],
                performance=learning["best_checkpoint_performance"],
                change_from_calibration=learning["change_from_calibration"],
                best_decision_path=learning["best_decision_path"],
                best_path_sha256=learning["best_path_sha256"])


def rooted_context(config, history, completed_rounds, domain):
    root = calibration_root(history,domain)
    best = best_from_calibration(history,domain,completed_rounds)
    path = decision_path(history,domain,best,completed_rounds)
    prior = context(config,history,completed_rounds) if configure(config)["enabled"] else None
    recent = completed_rounds[-2:]
    stalled = (len(recent)==2 and all(r.get("checkpoint",{}).get("status")=="measured"
               and r["checkpoint"]["performance"]>=r["checkpoint"]["input_incumbent_performance"] for r in recent))
    rebase = bool(configure(config)["enabled"] and (stalled or (prior or {}).get("rebase_from_best_checkpoint")))
    bad_paths = [dict(iteration=r["iteration"], checkpoint=r.get("checkpoint"),
                      decision_summary=r.get("decision_summary"),
                      directions=[dict(source_id=g["id"],actual_schedule=g[EFFECTIVE],
                                       proposal=g.get("selected_proposal",g.get("proposal"))) for g in r["groups"]])
                 for r in recent] if rebase else []
    return dict(policy="calibration-rooted-measured-path-v2",rebase_from_best_checkpoint=rebase,
                calibration_baseline=node_summary(root),
                calibration_selection=dict(origin="deterministic_complete_population_calibration_not_agent_round",
                    criterion="lowest eligible complete-population primary score",
                    measured_candidates=[node_summary(r) for r in history if r["id"] in domain["bootstrap_sources"]]),
                best_checkpoint_id=best["id"],best_checkpoint_actual_schedule=best[EFFECTIVE],
                best_checkpoint_performance=primary_score(best),
                change_from_calibration=primary_score(best)-primary_score(root),
                best_decision_path=path,best_path_sha256=value_sha(path),
                rollback_reason=("two_adjacent_measured_rounds_without_global_improvement" if stalled and rebase
                                 else "two_adjacent_round_declines" if rebase else None),
                unsuccessful_recent_paths=bad_paths,
                path_review_required=True,
                instruction=("Review the measured best node and its real predecessor decisions, parameter changes and outcomes. "
                             "Summarize the saved decision path and failed side paths; do not request hidden chain-of-thought. "
                             "Start from the best node and redirect toward a different potentially better complete configuration. "
                             "Do not resample the best, explain window mechanisms or design ablations." if rebase else
                             "Compare every opportunity with the calibration-rooted global best, including round one. "
                             "Use saved decisions and outcomes to seek potentially better complete paths, not mechanism explanations."))
