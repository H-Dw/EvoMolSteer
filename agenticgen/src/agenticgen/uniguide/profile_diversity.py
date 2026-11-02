"""Executor-owned absolute profiles and observable portfolio diversity, no inference."""
import itertools
from . import linear_scale as scale
from .io import value_sha


def distance(actual, best):
    return sum(abs(a-b) // 10 for a, b in zip(actual, best))


def classify(actual, best):
    amount = distance(actual, best)
    if not 1 <= amount <= 12:
        raise ValueError("Absolute profile must be 1..12 ticks from the current best")
    return "focused" if amount <= 2 else "expanded" if amount <= 6 else "bold"


def resolve(action, anchors, domain, seen, failures):
    from .box_search import enabled
    if enabled(domain):
        from .diverse_profiles import resolve as expanded_resolve
        return expanded_resolve(action,anchors,domain,seen,failures)
    source = anchors.get(action.get("anchor_id"))
    if source is None:
        raise ValueError("Unknown compatible measured anchor")
    actual = action.get("target_scale_schedule")
    if not isinstance(actual, list) or len(actual) != 6 or any(type(v) is not int for v in actual):
        raise ValueError("Six explicit integer target scales required")
    for value in actual:
        scale.check_strength(value, domain["control_settings"])
    if any(v for v, on in zip(actual, domain["guidance_box_mask"]) if not on):
        raise ValueError("An excluded box cannot be re-enabled")
    if any(actual == r.get(scale.EFFECTIVE) for r in seen):
        raise ValueError("Profile is already measured or reserved")
    if any(actual == r.get(scale.EFFECTIVE) for r in failures if r.get('failure') or r.get('error') or
           r.get('evidence_status') in ('statistics_ineligible','measured_displacement_growth')):
        raise ValueError("Failed or nonrankable exact profile cannot be resubmitted")
    # Each advisor receives an executor-derived incumbent, not a model-declared distance.
    incumbent = anchors.get(action.get("optimization_target", {}).get("incumbent_source_id"))
    if incumbent is None:
        raise ValueError("Absolute profile requires the actual measured global best")
    best = incumbent[scale.EFFECTIVE]
    target=action['optimization_target']
    if target.get('calibration_source_id')!=domain['initial_source'] or target.get('purpose')!='seek_potential_global_improvement':
        raise ValueError('Absolute action must bind the unchanged calibration root and optimization purpose')
    tier = classify(actual, best)
    if action.get("exploration_tier") != tier:
        raise ValueError("Exploration tier differs from actual distance to global best")
    return dict(action, **scale.binding(scale.encode(actual, domain), domain),
                direction="mixed", total_absolute_tick_change=distance(actual, best),
                task_type="optimization", domain_sha256=domain["domain_sha256"],
                calibration_key=domain["calibration_key"], executor_resolved=True,
                agent_action_sha256=value_sha(action), stability_guarantee=False)


def bank(history, domain, best, failures=()):
    """Finite, deterministic coverage suggestions; never claim global exhaustion."""
    from .box_search import enabled
    if enabled(domain):
        from .diverse_profiles import bank as expanded_bank
        return expanded_bank(history,domain,best,failures)
    active = [i for i, on in enumerate(domain["guidance_box_mask"]) if on]
    bad=[r for r in failures if r.get('failure') or r.get('error') or
         r.get('evidence_status') in ('statistics_ineligible','measured_displacement_growth')]
    blocked = {tuple(r[scale.EFFECTIVE]) for r in [*history, *bad] if scale.EFFECTIVE in r}
    candidates = []
    for count in (1, 2):
        for ids in itertools.combinations(active, count):
            for values in itertools.product(range(0, 201, 10), repeat=count):
                actual = list(best)
                for i, value in zip(ids, values): actual[i] = value
                change = distance(actual, best)
                if not 1 <= change <= 12 or tuple(actual) in blocked: continue
                changed = sum(a != b for a, b in zip(actual, best))
                if change >= 7 and len(active) >= 2 and changed < 2: continue
                candidates.append(actual)
    # Include multi-box redistributions without making a mechanism hypothesis.
    for offsets in itertools.product((-2, -1, 0, 1, 2), repeat=len(active)):
        if sum(x != 0 for x in offsets) < 3 or not 1 <= sum(map(abs, offsets)) <= 12: continue
        actual = list(best)
        for i, offset in zip(active, offsets): actual[i] += 10*offset
        if any(v < 0 or v > 200 for v in actual) or tuple(actual) in blocked: continue
        candidates.append(actual)
    unique = {tuple(p): p for p in candidates}
    result = []
    for tier in ("focused", "expanded", "bold"):
        options = [p for p in unique.values() if classify(p, best) == tier]
        for _ in range(4):
            if not options: break
            chosen = max(options, key=lambda p: (min((distance(p, q) for q in result), default=0), value_sha(p)))
            result.append(chosen); options.remove(chosen)
    return [dict(target_scale_schedule=p, exploration_tier=classify(p, best),
                 distance_from_global_best_ticks=distance(p, best), performance=None,
                 evidence_status="unmeasured", new_attempts=0) for p in result]


def require_portfolio(actions, domain, best, suggestions, remaining, *, pairwise=True):
    from .box_search import enabled
    if enabled(domain):
        from .diverse_profiles import require_portfolio as expanded_require
        return expanded_require(actions,domain,best,suggestions,remaining,pairwise=pairwise)
    profiles = [a["target_scale_schedule"] for a in actions]
    active = sum(domain["guidance_box_mask"])
    if remaining >= 50 and any(p["exploration_tier"] == "bold" for p in suggestions):
        if not any(classify(p, best) == "bold" and (active < 2 or sum(a != b for a, b in zip(p, best)) >= 2) for p in profiles):
            raise ValueError("Executable coverage candidates exist: select a genuine multi-box bold profile")
    for left, right in itertools.combinations(profiles if pairwise else [], 2):
        if distance(left, right) < 3 or (active >= 2 and sum(a != b for a, b in zip(left, right)) < 2):
            raise ValueError("Selected profiles are near duplicates")
