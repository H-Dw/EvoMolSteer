"""Compact whole-window temporal evidence from independent batch-node moments.

Continuous spline candidates see every observed node; no [0,.1) style strata
are introduced. Knots are exploratory policy candidates, not physical forces.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..io import digest, read_json, write_json


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.)
    return result


def _basis(times, indices, constant=False):
    u = (times - times[0]) / (times[-1] - times[0])
    columns = [np.ones(len(u))]
    if not constant:
        columns += [u, *[np.maximum(u - u[i], 0) for i in indices]]
    return np.stack(columns, axis=1)


def fit_stages(times, values, max_knots=2, minimum_nodes=6, stability_floor=.7):
    """Nested leave-one-batch-out fit; one-SE preference for simpler functions.

Within each held-out fold, choose knots using only the other batches. Constant,
linear, one-knot and two-knot models use the same full support. The resulting
CV and knot stability remain exploratory, not a causal guidance validation.
"""
    t, y = np.asarray(times, float), np.asarray(values, float)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Increasing observed times required")
    if y.ndim != 2 or y.shape[1] != len(t) or len(y) < 4 or not np.isfinite(y).all():
        raise ValueError("Complete independent batch-node contrast matrix required")
    if max_knots not in (0, 1, 2) or not isinstance(minimum_nodes, int) or minimum_nodes < 2:
        raise ValueError("Bounded spline complexity required")
    if not 0 < stability_floor <= 1:
        raise ValueError("Invalid knot stability threshold")
    pool = range(minimum_nodes, len(t) - minimum_nodes + 1)
    specs = [("constant", [()]), ("linear", [()])]
    for count in range(1, max_knots + 1):
        knots = [c for c in combinations(pool, count)
                 if all(b - a >= minimum_nodes for a, b in zip((0,) + c, c + (len(t),)))]
        if knots:
            specs.append((f"spline_{count}", knots))
    losses, fits, fold_knots = [], [], []
    total = y.sum(0)
    mean = y.mean(0)
    for name, knots in specs:
        bases = [_basis(t, k, name == "constant") for k in knots]
        projections = np.stack([b @ np.linalg.pinv(b) for b in bases])
        select = lambda v: int(np.argmin(((np.einsum("kij,j->ki", projections, v) - v) ** 2).mean(1)))
        fold_loss, chosen = [], []
        for i, observation in enumerate(y):
            training = (total - observation) / (len(y) - 1)
            j = select(training)
            fold_loss.append(float(np.mean((projections[j] @ training - observation) ** 2)))
            chosen.append(knots[j])
        j = select(mean)
        fits.append((knots[j], bases[j] @ np.linalg.lstsq(bases[j], mean, rcond=None)[0]))
        losses.append(fold_loss)
        fold_knots.append(chosen)
    loss = np.asarray(losses)
    average = loss.mean(1)
    best = int(average.argmin())
    numerical_floor = 1e-12 * max(float(np.mean(y ** 2)), 1e-12)
    cutoff = average[best] + loss[best].std(ddof=1) / np.sqrt(len(y)) + numerical_floor
    chosen = int(np.flatnonzero(average <= cutoff)[0])
    knots, prediction = fits[chosen]
    stability = [float(np.mean([any(abs(i - j) <= 2 for j in k) for k in fold_knots[chosen]])) for i in knots]
    accepted = bool(knots) and min(stability) >= stability_floor
    return {"method": specs[chosen][0], "CV_MSE_by_model": dict(zip([s[0] for s in specs], average.tolist())),
            "selection_rule": "nested batch CV, one standard error with numerical floor, then knot stability",
            "knot_times": [float(t[i]) for i in knots], "knot_stability": stability,
            "accepted_for_stage_hypothesis": accepted, "fitted_values_z": prediction.tolist(),
            "fitted_rate_z_per_t": np.gradient(prediction, t).tolist(),
            "independent_batches": len(y), "observed_support": [float(t[0]), float(t[-1])],
            "extrapolation_allowed": False}


def mine_profiles(moments_path, prior_path, packet_path, output, *, max_knots=2, minimum_nodes=6):
    """Input: existing wide moments + scales. Output: one compact profile JSON.

    Save batch-level CIs, rates and proposed knots, without re-expanding seeds,
    duplicating coordinates or retaining per-fold raw rows.
    """
    prior, packet = read_json(prior_path), read_json(packet_path)
    t = np.asarray(prior["times"], float)
    if prior["window"] != packet["clocks"]["learning_window"] or prior["times"] != packet["clocks"]["observed_score_times"]:
        raise ValueError("Source moments and selection metadata clocks disagree")
    fields, scales = prior["features"], np.asarray(prior["feature_scales"], float)
    if scales.shape != (len(fields),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Measured positive coordinate scales required")
    columns = ["batch", "score_time"] + [prefix + "__" + f for prefix in ["high", "low"] for f in fields]
    df = pd.read_parquet(moments_path, columns=columns).sort_values(["batch", "score_time"])
    batch_ids = sorted(df.batch.unique())
    if df.duplicated(["batch", "score_time"]).any() or not np.array_equal(df.score_time, np.tile(t, len(batch_ids))):
        raise ValueError("Missing or duplicate independent batch nodes")
    result, pvalues = {}, []
    for feature, scale in zip(fields, scales):
        y = ((df["high__" + feature] - df["low__" + feature]) / scale).to_numpy(float).reshape(len(batch_ids), len(t))
        fitted = fit_stages(t, y, max_knots, minimum_nodes)
        rate = np.gradient(y, t, axis=1)
        half = lambda a: student_t.ppf(.975, len(y) - 1) * a.std(0, ddof=1) / np.sqrt(len(y))
        mean, se = y.mean(0), half(y)
        integrated = np.trapezoid(y, t, axis=1) / (t[-1] - t[0])
        standard_error = integrated.std(ddof=1) / np.sqrt(len(y))
        p = float(2 * student_t.sf(abs(integrated.mean() / standard_error), len(y) - 1)) if standard_error > 0 else (1. if integrated.mean() == 0 else 0.)
        result[feature] = {**fitted, "score_times": t.tolist(), "empirical_values_z": mean.tolist(),
                           "pointwise_CI95_low": (mean - se).tolist(), "pointwise_CI95_high": (mean + se).tolist(),
                           "empirical_rate_z_per_t": rate.mean(0).tolist(),
                           "rate_pointwise_CI95_halfwidth": half(rate).tolist(),
                           "whole_window_effect_z": float(integrated.mean()), "whole_window_p": p,
                           "whole_window_CI95_halfwidth": float(student_t.ppf(.975, len(y) - 1) * standard_error)}
        pvalues.append(p)
    for item, q in zip(result.values(), bh(pvalues)):
        item["whole_window_q"] = float(q)
    profile = {"schema_version": "whole-window-stage-profiles-1.0", "window": prior["window"],
               "score_times": t.tolist(), "representation": "predicted_endpoint_world_A_features",
               "functions": result, "batch_ids": batch_ids,
               "decoded_terminal_labels_available": packet["decoded_terminal_learning_labels_available"],
               "input_sha256": {"moments": digest(moments_path), "prior": digest(prior_path), "packet": digest(packet_path)},
               "tool_sha256": digest(__file__),
               "interpretation": "Selected-vs-control coordinate association. Pointwise intervals are descriptive, not simultaneous. Stage stability is exploratory. No physical energy, causal guidance strength or final-affinity trajectory is inferred."}
    write_json(output, profile)
    return profile
