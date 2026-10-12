"""Declarative, observed-clock policies for dose and endpoint reward shape.

No model, network, dataset mutation or learned predictor is involved. Numerical
tables, rather than recipe names or prompt wording, define execution identity.
"""
import copy
import hashlib
import json

import numpy as np

VERSION = "endpoint-temporal-policy-1.0"
ROUTES = {"dose", "reward", "combined"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def clock_grid(window, steps):
    a, b = np.asarray(window, float)
    if not np.isfinite([a, b]).all() or not 0 <= a < b <= 1:
        raise ValueError("Finite learning window within the native horizon required")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("Integer integration grid required")
    if not np.allclose(np.array([a, b]) * steps, np.rint(np.array([a, b]) * steps), atol=1e-7, rtol=0):
        raise ValueError("Learning window boundaries must align with integration")
    times = np.arange(steps, dtype=float) / steps
    times = times[(times >= a - 1e-8) & (times + 1 / steps <= b + 1e-8)]
    if len(times) < 2:
        raise ValueError("At least two controlled nodes required")
    return times


def sample_knots(phase, knots, values):
    """Smoothstep between declared knots, never interpolate teacher coordinates."""
    u, k, v = np.asarray(phase, float), np.asarray(knots, float), np.asarray(values, float)
    if k.ndim != 1 or len(k) < 2 or v.shape != k.shape or k[0] != 0 or k[-1] != 1:
        raise ValueError("One value per increasing normalized knot, endpoints 0 and 1")
    if not np.isfinite(k).all() or not np.isfinite(v).all() or np.any(np.diff(k) <= 0):
        raise ValueError("Finite increasing knots required")
    if not np.isfinite(u).all() or np.any(u < 0) or np.any(u > 1):
        raise ValueError("No phase extrapolation")
    j = np.minimum(np.searchsorted(k, u, side="right") - 1, len(k) - 2).clip(0)
    f = (u - k[j]) / (k[j + 1] - k[j])
    f = f * f * (3 - 2 * f)
    return v[j] + f * (v[j + 1] - v[j])


def make_policy(window, times, steps, route, *, knots=(0., .5, 1.),
                dose=(1., 1., 1.), blend=(0., 0., 0.), delta=(1., 1., 1.)):
    """Equalize time-integrated ratio budget, not realized physical displacements.

The latter depend on each seed's native flow, gradient and safety caps. They
must be measured; this normalization does not pretend to match realized dose.
"""
    t = np.asarray(times, float)
    if t.shape != clock_grid(window, steps).shape or not np.allclose(t, clock_grid(window, steps), atol=2e-6, rtol=0):
        raise ValueError("Complete exact historical score nodes required")
    # Normalize over observed SCORE nodes, not the unobserved next-state endpoint.
    # Forward/reversed profiles then have the same discrete parameter average.
    phase = (t - t[0]) / (t[-1] - t[0])
    w = sample_knots(phase, knots, dose)
    if np.any(w <= 0):
        raise ValueError("Positive guidance throughout the learned window required")
    # Equal dt on the frozen native integration grid. Left-node Riemann sum.
    w /= w.mean()
    policy = {"schema_version": VERSION, "route": route, "window": list(window),
              "steps": steps, "score_times": t.tolist(), "dose_factors": w.tolist(),
              "teacher_blend": sample_knots(phase, knots, blend).tolist(),
              "delta_A": sample_knots(phase, knots, delta).tolist(),
              "normalized_phase_support": [float(t[0]), float(t[-1])],
              "interpolation": "smoothstep_parameters_only", "minimum_dose_factor": .25,
              "maximum_dose_factor": 2., "ratio_budget": "sum(dt*w)=learning_duration",
              "knots_origin": "predeclared_intervention; not a discovered biological stage"}
    validate_policy(policy, window, t, steps)
    return policy


def validate_policy(policy, window, times, steps=None):
    expected_keys = {"schema_version", "route", "window", "steps", "score_times", "dose_factors",
                     "teacher_blend", "delta_A", "interpolation", "minimum_dose_factor",
                     "maximum_dose_factor", "ratio_budget", "knots_origin", "normalized_phase_support"}
    if set(policy) != expected_keys or policy["schema_version"] != VERSION or policy["route"] not in ROUTES:
        raise ValueError("Unregistered temporal policy")
    if list(window) != policy["window"] or steps is not None and steps != policy["steps"]:
        raise ValueError("Policy/learning/integration support mismatch")
    t = np.asarray(times, float)
    if t.shape != (len(policy["score_times"]),) or not np.allclose(t, policy["score_times"], atol=2e-6, rtol=0):
        raise ValueError("Policy must cover precisely the observed score nodes")
    grid = clock_grid(window, policy["steps"])
    if t.shape != grid.shape or not np.allclose(t, grid, atol=2e-6, rtol=0):
        raise ValueError("Observed nodes must cover all controlled integration steps")
    if policy["normalized_phase_support"] != [float(t[0]), float(t[-1])]:
        raise ValueError("Normalized timing must refer to observed score support")
    w, b, d = [np.asarray(policy[k], float) for k in ("dose_factors", "teacher_blend", "delta_A")]
    if any(v.shape != t.shape or not np.isfinite(v).all() for v in [w, b, d]):
        raise ValueError("One finite parameter value per observed node")
    if policy["minimum_dose_factor"] != .25 or policy["maximum_dose_factor"] != 2.:
        raise ValueError("Unregistered dose bounds")
    if np.any(w < .25) or np.any(w > 2.) or not np.isclose(w.mean(), 1., atol=1e-12, rtol=0):
        raise ValueError("Bounded equal ratio budget required")
    if np.any(b < 0) or np.any(b > 1) or np.any(d < .5) or np.any(d > 2.):
        raise ValueError("Registered blend/curvature bounds required")
    if policy["route"] == "dose" and (np.any(b != 0) or np.any(d != 1)):
        raise ValueError("Dose exploration must preserve M2 reward shape")
    if policy["route"] == "reward" and np.any(w != 1):
        raise ValueError("Reward exploration must preserve the dose policy")
    if policy["interpolation"] != "smoothstep_parameters_only" or policy["ratio_budget"] != "sum(dt*w)=learning_duration":
        raise ValueError("Unknown schedule interpolation/budget convention")
    return policy


def policy_is_identity(policy):
    return all(all(x == expected for x in policy[key]) for key, expected in
               [("dose_factors", 1.), ("teacher_blend", 0.), ("delta_A", 1.)])


def program_signature(program):
    """Deduplicate equivalent executed tables, independent of prompts/recipe IDs.

Identity policies collapse to the historical effective signature so existing
M2 results can be reused. Reference bytes remain part of non-identity identity.
"""
    from evomolsteer.continuous.m2_skill_api import effective_signature
    p = copy.deepcopy(program)
    policy = p.pop("temporal_policy", None)
    p.pop("temporal_policy_sha256", None)
    p.pop("temporal_design_provenance", None)
    if policy is None or policy_is_identity(policy):
        return effective_signature(p)
    numerical = {k: policy[k] for k in ["schema_version", "steps", "window", "score_times",
                                       "dose_factors", "teacher_blend", "delta_A"]}
    return canonical_hash({"base": effective_signature(p), "executed_policy": numerical})
