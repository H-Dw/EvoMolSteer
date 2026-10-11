"""Deterministic whole-window tools over existing compact Steer statistics.

No trajectory expansion, neural fitting, terminal-label invention, or time bins.
The API packet contains all scalar effects and compact curve diagnostics; the
full reproducible functions and float64 summaries are separate small artifacts.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import legvander, legval, legder
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

from evomolsteer.continuous.m2_skill_api import BASE, REF, SOURCE, compact_tables
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.statistics import bh

REGIONAL = "docs/experiments/ck2_affinity_geometry30_20261007/regional_mining_v1"
DYNAMIC = "configs/experiments/skill_ablation_v1/endpoint_reference.json.gz"
BRANCH = "docs/experiments/flowcompat30_20261009/mining/branch_mutation_v2"
JOINT = "docs/experiments/flowcompat30_20261009/mining/joint_contrast_v2"
HISTORICAL = "docs/experiments/guidance_vs_steer_20261007/steer_full1000/candidate_metrics.csv"


def fit_continuous(times, batch_values, *, maximum_degree=3, fidelity_limit=.25):
    """Choose low-order curves by held-out independent-batch prediction.

    Scores are descriptive, not new significance tests. A poor polynomial is
    explicitly replaced by observed-node linear interpolation, never extended.
    """
    t, y = np.asarray(times, float), np.asarray(batch_values, float)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Finite increasing observed times required")
    if y.ndim != 2 or y.shape[1] != len(t) or y.shape[0] < 3 or not np.isfinite(y).all():
        raise ValueError("Complete independent-batch time curves required")
    u = 2 * (t-t[0]) / (t[-1]-t[0]) - 1
    designs = [legvander(u, degree) for degree in range(min(maximum_degree, len(t)-2)+1)]
    errors = []
    for design in designs:
        error = []
        for b in range(len(y)):
            other = np.delete(y, b, axis=0).mean(0)
            pred = design @ np.linalg.lstsq(design, other, rcond=None)[0]
            error.append(float(np.mean((pred-y[b])**2)))
        errors.append(float(np.mean(error)))
    # Stable tie handling prefers the smaller model.
    minimum = min(errors)
    tolerance = 1e-12*max(1., max(errors))
    degree = next(i for i, error in enumerate(errors) if error <= minimum+tolerance)
    design = designs[degree]
    coefficients = np.linalg.lstsq(design, y.mean(0), rcond=None)[0]
    predicted = design @ coefficients
    observed = y.mean(0)
    rmse = float(np.sqrt(np.mean((predicted-observed)**2)))
    relative = rmse / max(float(np.sqrt(np.mean(observed**2))), 1e-12)
    strong = np.abs(observed) > .1
    conflicts = int(np.sum(strong & (predicted*observed < 0)))
    accepted = relative <= fidelity_limit and conflicts == 0
    derivative = legval(u, legder(coefficients)) * 2/(t[-1]-t[0])
    result = {"observed_support": [float(t[0]), float(t[-1])],
        "representation": "predicted_endpoint_world_A_features",
        "method": "legendre" if accepted else "observed_node_linear",
        "degree": degree, "coefficients": coefficients.tolist(),
        "held_out_batch_MSE_by_degree": errors, "relative_node_RMSE": relative,
        "strong_sign_conflicts": conflicts, "polynomial_accepted": accepted,
        "independent_batches": len(y), "sign_changes": int(np.sum(observed[1:]*observed[:-1] < 0)),
        "first_z": float(observed[0]), "last_z": float(observed[-1]),
        "whole_window_z": float(np.trapezoid(observed, t)/(t[-1]-t[0])),
        "first_rate_z": float(derivative[0]) if accepted else float((observed[1]-observed[0])/(t[1]-t[0])),
        "last_rate_z": float(derivative[-1]) if accepted else float((observed[-1]-observed[-2])/(t[-1]-t[-2])),
        "extrapolation_allowed": False}
    if not accepted:
        result.update(times=t.tolist(), values_z=observed.tolist())
    return result


def evaluate_curve(function, times):
    x = np.asarray(times, float)
    lo, hi = function["observed_support"]
    if not np.isfinite(x).all() or np.any(x < lo-1e-10) or np.any(x > hi+1e-10):
        raise ValueError("No extrapolation outside observed score-clock support")
    if function["method"] == "legendre":
        return legval(2*(x-lo)/(hi-lo)-1, function["coefficients"])
    return np.interp(x, function["times"], function["values_z"])


def turnover_accounting(times, contrast, edge_contrast):
    """Exact accounting; residual includes reclassification and parent weights."""
    t = np.asarray(times, float)
    c, e = np.asarray(contrast, float), np.asarray(edge_contrast, float)
    if c.shape != e.shape or c.ndim != 2 or c.shape[1] != len(t) or not np.isfinite(c).all() or not np.isfinite(e[:, 1:]).all():
        raise ValueError("Complete equal-clock population and edge contrasts required")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Increasing time support required")
    change = np.diff(c, axis=1)
    within = e[:, 1:]
    residual = change-within
    if not np.allclose(change, within+residual, rtol=1e-12, atol=1e-12):
        raise ValueError("Turnover identity failed")
    duration = t[-1]-t[0]
    return {"population_rate": change.sum(1)/duration,
            "edge_rate": within.sum(1)/duration,
            "turnover_residual_rate": residual.sum(1)/duration}


def build(repo, output):
    repo, out = Path(repo).resolve(), Path(output).resolve()
    if (out / "tool_receipt.json").exists():
        receipt = read_json(out / "tool_receipt.json")
        if receipt["tool_source_sha256"] != digest(__file__):
            raise ValueError("Tool implementation changed; use a new versioned output")
        for name, sha in receipt["input_sha256"].items():
            if digest(repo/name) != sha:
                raise ValueError("Frozen scientific input changed: " + name)
        for name, sha in receipt["output_sha256"].items():
            if digest(out/name) != sha:
                raise ValueError("Scientific tool output changed: " + name)
        return read_json(out/"packet.json")
    source = read_json(repo/SOURCE)
    inputs = {SOURCE: digest(repo/SOURCE), BASE: digest(repo/BASE), REF: digest(repo/REF), DYNAMIC: digest(repo/DYNAMIC)}
    # Validate all used historical source identities, not just their new paths.
    for name, sha in source["source_hashes"].items():
        if digest(repo/name) != sha:
            raise ValueError("Original derived evidence changed: " + name)
        inputs[name] = sha
    for name in [REGIONAL+"/batch_node_moments.parquet", REGIONAL+"/region_prior.json",
                 BRANCH+"/whole_window_effects.parquet", JOINT+"/whole_window_effects.parquet"]:
        inputs[name] = digest(repo/name)
    for folder in [BRANCH, JOINT]:
        identity=read_json(repo/folder/"manifest.json")["files"]["whole_window_effects.parquet"]["sha256"]
        if digest(repo/folder/"whole_window_effects.parquet")!=identity:
            raise ValueError("Supplemental original tool receipt mismatch")
    static = json.loads(gzip.decompress((repo/REF).read_bytes()))
    dynamic = json.loads(gzip.decompress((repo/DYNAMIC).read_bytes()))
    for key in ["window", "times", "landmarks_A", "origin_A", "representation"]:
        if static[key] != dynamic[key]:
            raise ValueError("Static/dynamic reference coordinate or clock mismatch")
    moments = pd.read_parquet(repo/REGIONAL/"batch_node_moments.parquet")
    prior = read_json(repo/REGIONAL/"region_prior.json")
    times = np.asarray(static["times"], float)
    if prior["times"] != times.tolist() or prior["window"] != static["window"]:
        raise ValueError("Regional observed support differs from teachers")
    batch_ids = sorted(moments.batch.unique())
    moments = moments.sort_values(["batch", "score_time"])
    if moments.duplicated(["batch", "score_time"]).any() or not np.array_equal(moments.score_time.to_numpy(), np.tile(times, len(batch_ids))):
        raise ValueError("Missing or duplicate independent batch nodes")
    features = prior["features"]
    scale = np.asarray(prior["feature_scales"], float)
    if scale.shape != (len(features),) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Finite positive original feature scales required")
    fits, summaries, turnover, salience = {}, [], [], {}
    for j, field in enumerate(features):
        arr = lambda kind: moments[kind+"__"+field].to_numpy(float).reshape(len(batch_ids), len(times))
        contrast = (arr("high")-arr("low"))/scale[j]
        edges = (arr("edge_high")-arr("edge_low"))/scale[j]
        fit = fit_continuous(times, contrast)
        fits[field] = fit
        summaries.append({"feature": field, **{k:v for k,v in fit.items() if not isinstance(v, (dict, list))}})
        pieces = turnover_accounting(times, contrast, edges)
        for measure, value in pieces.items():
            mean = float(value.mean())
            se = float(value.std(ddof=1)/np.sqrt(len(value)))
            half = float(student_t.ppf(.975, len(value)-1)*se)
            p = float(2*student_t.sf(abs(mean/se), len(value)-1)) if se > 0 else (1. if mean == 0 else 0.)
            turnover.append({"feature": field, "measure": measure, "mean_rate_z": float(value.mean()),
                "batch_std": float(value.std(ddof=1)), "positive_batch_fraction": float((value>0).mean()),
                "independent_batches": len(value), "CI95_low": mean-half, "CI95_high": mean+half,
                "p_batch_t": p, "inferential_scope": "descriptive whole-window rate, independent batches; not a causal selection test"})
        if field.endswith("_soft_mass"):
            region = int(field.split("_")[1])
            # An attention heuristic, not an inferred causal gradient or target.
            values = np.clip(np.abs(evaluate_curve(fit, times)), .1, 2.)
            salience[str(region)] = values.tolist()
    for row, q in zip(turnover, bh([r["p_batch_t"] for r in turnover])):
        row["q_BH_600_rates"] = float(q)
    clouds = np.asarray(static["frames"][-1]["teacher_endpoint_A"], float)
    rms = []
    for i, cloud in enumerate(clouds):
        for other in clouds[:i]:
            d = ((cloud[:,None]-other[None])**2).sum(-1)
            rows, cols = linear_sum_assignment(d)
            rms.append(float(np.sqrt(d[rows,cols].mean())))
    kept = [x for x in source["evidence"] if x["kind"] not in {"discovery_quality", "regional_curves"}]
    for kind, name in [("branch_adjusted", BRANCH), ("joint_diagnostics", JOINT)]:
        for i, row in enumerate(pd.read_parquet(repo/name/"whole_window_effects.parquet").to_dict("records")):
            kept.append({"evidence_id": f"{kind}:{i}", "kind": kind, "data": {
                k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k,v in row.items()}})
    for i, row in enumerate(summaries):
        kept.append({"evidence_id": f"continuous:{i}", "kind": "continuous_trends", "data": row})
    for i, row in enumerate(turnover):
        kept.append({"evidence_id": f"turnover:{i}", "kind": "turnover_accounting", "data": row})
    packet = {"schema_version": "scientific-steer-evidence-3.0", "tables": compact_tables(kept),
        "objective": "Final predicted target affinity primary; validity and strain secondary; coordinate control only",
        "clocks": {"learning_window": static["window"], "observed_score_times": static["times"],
            "curve_support": [float(times[0]), float(times[-1])],
            "representation": static["representation"], "label_semantics": static["label_semantics"]},
        "teacher_modes": {"count": len(clouds), "assignment_rms_A_quantiles": np.quantile(rms,[.1,.5,.9]).tolist(),
            "landmarks_A": static["landmarks_A"], "source": REF,
            "dynamic_source": DYNAMIC, "temperature_edits": False},
        "prerequisites": {key: True for key in ["selection_contrasts", "descendant_proxy", "uncertainty", "batch_support",
            "spatial_diagnostics", "joint_geometry", "continuous_trends", "turnover_accounting", "formula_registry",
            "temporal_recipes", "geometric_scales", "teacher_modes", "counterevidence"]},
        "decoded_terminal_learning_labels_available": False,
        "limitations": ["Descendant credit is recorded joint-head credit, not decoded terminal affinity",
            "Turnover residual includes score reclassification and extant-parent weighting, not only resampling",
            "Adjusted branch covariance is supplied evidence; region moments themselves are not rigid-motion residuals",
            "Curve fitting does not establish causal benefit or a gradient; no time bins or outside-window analysis",
            "Static teachers are last-learning-node forecasts, not actual final structures",
            "All weak and null scalar effects retained; current study outcomes are withheld from design roles"],
        "source_sha256": inputs, "tool_source_sha256": digest(__file__)}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/"packet.json", packet)
    write_json(out/"continuous_functions.json", {"times": times.tolist(), "window": static["window"], "functions": fits})
    write_json(out/"regional_salience.json", {"times": times.tolist(), "window": static["window"], "regions": salience,
        "semantics": "Bounded absolute validated contrast used only as detached regional attention, not affinity force"})
    pd.DataFrame(summaries).to_parquet(out/"continuous_summary.parquet", compression=None, index=False)
    pd.DataFrame(turnover).to_parquet(out/"turnover_summary.parquet", compression=None, index=False)
    augmented = copy.deepcopy(static)
    for target, observed in zip(augmented["frames"], dynamic["frames"]):
        target["trajectory_teacher_endpoint_A"] = observed["teacher_endpoint_A"]
        target["trajectory_teacher_batches"] = observed["teacher_batches"]
    augmented["scientific_sources"] = {"static": digest(repo/REF), "dynamic": digest(repo/DYNAMIC)}
    (out/"temporal_reference.json.gz").write_bytes(gzip.compress(
        json.dumps(augmented, separators=(",", ":")).encode(), mtime=0))
    regions = {x["evidence_id"]: int(x["data"]["region"]) for x in kept if "region" in x["data"]}
    for i, row in enumerate(summaries):
        if row["feature"].startswith("landmark_"):
            regions[f"continuous:{i}"] = int(row["feature"].split("_")[1])
    write_json(out/"index.json", {"ids": [x["evidence_id"] for x in kept], "regions": regions})
    artifacts = ["packet.json", "continuous_functions.json", "regional_salience.json", "continuous_summary.parquet",
                 "turnover_summary.parquet", "temporal_reference.json.gz", "index.json"]
    if any(digest(repo/name) != sha for name,sha in inputs.items()):
        raise ValueError("Scientific inputs changed during calculation")
    write_json(out/"tool_receipt.json", {"tool": "scientific_whole_window_tools", "input_sha256": inputs,
        "output_sha256": {n:digest(out/n) for n in artifacts}, "tool_source_sha256": digest(__file__),
        "float_storage": "float64 uncompressed Parquet; content-addressed JSON inputs; one teacher reference",
        "time_binning": False, "particle_feature_cache_created": False,
        "bytes": {n:(out/n).stat().st_size for n in artifacts}})
    return packet


def historical_control(repo, output):
    """Evaluation only: never place historical/current performance in API data."""
    repo,out=Path(repo),Path(output)
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(repo/HISTORICAL)
    if "arm" in frame:
        arms=frame.arm.unique()
        if len(arms)!=1:raise ValueError("Ambiguous historical Steer arm")
    cols=["batch","slot","pic50_on_rescore","valid_connected","pb_fast_pass","energy_status","mmff_relief_per_heavy","smiles"]
    if not set(cols)<=set(frame) or frame.duplicated(["batch","slot"]).any() or not np.isfinite(frame.pic50_on_rescore).all():
        raise ValueError("Historical comparison coverage mismatch")
    destination=out/"historical_steer.metrics.parquet"
    frame[cols].to_parquet(destination,compression=None,index=False)
    write_json(out/"historical_steer.source.json",{"path":HISTORICAL,"sha256":digest(repo/HISTORICAL),
        "compact_sha256":digest(destination),"n":len(frame),"batches":sorted(map(int,frame.batch.unique())),
        "mean_pic50":float(frame.pic50_on_rescore.mean()),"comparison":"unpaired; unequal online resampling budget",
        "visible_to_design_roles":False})
    return destination
