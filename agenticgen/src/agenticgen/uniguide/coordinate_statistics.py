"""Upper-tail-only population statistics; never alter candidate gradients."""
import math
import numpy as np

POLICY = "absolute-coordinate-upper-outer-fence-v1"
CONFIG = "rmsd_outlier_policy"
SUMMARY = "coordinate_loss_statistics"
RANKING = "ranking_condition_coordinate_root_mean_square_deviation_angstrom"
DEFAULTS = dict(method="upper_outer_fence", interquartile_multiplier=3.0,
                minimum_upper_excess_angstrom=1.0, maximum_excluded_fraction=0.1,
                minimum_population=10)


def policy(config=None):
    supplied = (config or {}).get(CONFIG, {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown upper-RMSD outlier policy fields")
    settings = dict(DEFAULTS, **supplied)
    if settings["method"] != "upper_outer_fence":
        raise ValueError("Only the declared upper outer fence is supported")
    for key in ("interquartile_multiplier", "minimum_upper_excess_angstrom"):
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Positive finite outlier threshold parameters required")
    fraction = settings["maximum_excluded_fraction"]
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction < .25:
        raise ValueError("Only a minority below one quarter may be excluded")
    if type(settings["minimum_population"]) is not int or settings["minimum_population"] < 4:
        raise ValueError("At least four observations required to estimate quartiles")
    return settings


def summarize(values, settings=None, identities=None):
    settings = policy({CONFIG: settings or {}})
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise FloatingPointError("Missing, negative or nonfinite RMSD is not a removable high outlier")
    if identities is not None and len(identities) != len(values):
        raise ValueError("Outlier identities and measurements differ")
    x = np.asarray(values, dtype=np.float64)
    lower, upper = np.quantile(x, [.25, .75], method="linear")
    threshold = float(upper + max(settings["interquartile_multiplier"] * (upper-lower),
                                  settings["minimum_upper_excess_angstrom"]))
    excluded = (x > threshold) if len(x) >= settings["minimum_population"] else np.zeros(len(x), dtype=bool)
    retained = x[~excluded]
    count = int(excluded.sum())
    eligible = bool(len(retained) and count / len(x) <= settings["maximum_excluded_fraction"] + 1e-12)
    with np.errstate(over="raise", invalid="raise"):
        try:
            raw_loss = float(np.mean(x*x/2))
            kept_loss = float(np.mean(retained*retained/2)) if len(retained) else None
        except FloatingPointError as error:
            raise FloatingPointError("Nonfinite population coordinate loss; no replacement or clipping") from error
    return dict(policy_id=POLICY, scope="complete_teacher_configuration_population_only",
        policy=settings, expected=len(x), measured=len(x), retained_count=len(retained),
        excluded_high_count=count, excluded_high_fraction=count/len(x),
        threshold_angstrom=threshold, quartile_25_angstrom=float(lower), quartile_75_angstrom=float(upper),
        minimum=float(x.min()), maximum=float(x.max()), arithmetic_mean=float(x.mean()),
        upper_tail_filtered_mean=float(retained.mean()) if len(retained) else None,
        raw_mean_coordinate_loss_angstrom_squared=raw_loss,
        upper_tail_filtered_mean_coordinate_loss_angstrom_squared=kept_loss,
        excluded_high_indices=np.flatnonzero(excluded).tolist(),
        excluded_high_samples=[dict(index=i, rmsd_angstrom=values[i],
            **({"identity": identities[i]} if identities is not None else {})) for i in np.flatnonzero(excluded).tolist()],
        lower_tail_excluded_count=0, threshold_comparison="strictly_greater_than",
        ranking_eligible=eligible, status="complete" if eligible else "too_many_high_outliers",
        gradient_modified=False, source_samples_deleted=False, unit="angstrom")


def aggregate_fields(records, config):
    from .absolute_coordinate import RMSD
    identities = [{k:r.get(k) for k in ("node_id", "logical_seed", "logical_slot")} for r in records]
    result = summarize([r.get(RMSD) for r in records], policy(config), identities)
    return {RANKING: result["upper_tail_filtered_mean"] if result["ranking_eligible"] else None,
            SUMMARY: result}


def ranking_eligible(row):
    return row.get(SUMMARY, {}).get("ranking_eligible", True)


def descriptive(values):
    finite = [float(v) for v in values if type(v) in (int, float) and math.isfinite(v)]
    if not finite:
        return dict(status="unmeasured", measured=0, expected=len(values))
    x = np.asarray(finite)
    return dict(status="complete" if len(x)==len(values) else "incomplete_descriptive_only",
        measured=len(x), expected=len(values), arithmetic_mean=float(x.mean()),
        median=float(np.median(x)), minimum=float(x.min()), maximum=float(x.max()),
        percentile_90=float(np.quantile(x,.9)), percentile_95=float(np.quantile(x,.95)),
        observations_retained=len(x), filtering_applied=False, unit="angstrom")
