"""Synthesize a joint regional coordinate hypothesis from frozen observations.

This is a deterministic data transformation, not an affinity estimator. Frozen
whole-window covariance fits supply directions; original same-ancestor mutation
evidence continues to determine eligibility and physical displacement amplitude.
"""
from __future__ import annotations

import copy
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial import Legendre

from ..io import digest, read_json

REGION_PATTERN = re.compile(r"^region_(\d+)_internal_displacement_([xyz])$")
METRIC = "adjusted_covariance"
REQUIRED_FILES = (
    "whole_window_effects.parquet", "batch_window_statistics.parquet",
    "fitted_trends.json", "feature_catalog.json", "manifest.json",
    "regional_direction_coverage.json", "evidence.json",
)


def validate_live_receipt(path, tool_id):
    """Verify an existing actual-process receipt without rewriting it."""
    path = Path(path).resolve()
    receipt = read_json(path)
    if receipt.get("tool_id") != tool_id:
        raise ValueError("Receipt tool identity does not match")
    unchanged = receipt.get("inputs_and_code_unchanged")
    if receipt.get("schema_version") == "flowcompat-supplemental-tool-receipt-2.0":
        unchanged = receipt.get("inputs_unchanged") and receipt.get("source_unchanged")
    if receipt.get("returncode") != 0 or not unchanged:
        raise ValueError("Successful unchanged live execution is required")
    if not receipt.get("complete_outputs") or receipt.get("error"):
        raise ValueError("Receipt is incomplete or records an execution error")
    command = receipt.get("command")
    if not isinstance(command, list) or len(command) < 2:
        raise ValueError("Actual argv is missing")
    bound = {str(path): digest(path)}
    for category in ("input_files", "code_files", "output_files"):
        files = receipt.get(category)
        if not isinstance(files, dict) or not files:
            raise ValueError("Receipt must bind input, source and output files")
        for name, sha in files.items():
            source = Path(name).resolve()
            if not source.is_file() or digest(source) != sha:
                raise ValueError(f"Bound {category} file changed: {source}")
            if str(source) in bound and bound[str(source)] != sha:
                raise ValueError("Inconsistent upstream file hash")
            bound[str(source)] = sha
        # The supplemental runner uses slash-normalized absolute keys; the live
        # tool itself uses platform paths. Identity must not depend on spelling.
        receipt[category] = {str(Path(name).resolve()): sha for name, sha in files.items()}
    if receipt.get("plan_path"):
        plan = Path(receipt["plan_path"]).resolve()
        if digest(plan) != receipt.get("plan_sha256"):
            raise ValueError("Actual source tool plan changed")
        bound[str(plan)] = digest(plan)
    if str(Path(command[1]).resolve()) not in receipt["code_files"]:
        raise ValueError("Executed script is not source-bound")
    return receipt, bound


def _fit(fit):
    if not isinstance(fit, dict):
        raise ValueError("A complete XYZ frozen fit is required")
    coefficients = np.asarray(fit.get("coefficients"), dtype=float)
    degree = fit.get("degree")
    start = float(fit.get("observed_time_start", np.nan))
    end = float(fit.get("observed_time_end", np.nan))
    if not isinstance(degree, int) or degree < 0 or coefficients.shape != (degree + 1,):
        raise ValueError("Frozen Legendre coefficient shape does not match degree")
    if not np.isfinite(coefficients).all() or not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("Finite fitted coefficients and a nonempty observed interval are required")
    return coefficients, start, end


def evaluate_xyz(functions, time):
    """Return None outside the common observed interval, never extrapolate."""
    if set(functions) != set("xyz"):
        raise ValueError("All three XYZ components must be present")
    fits = [_fit(functions[axis]) for axis in "xyz"]
    start = max(value[1] for value in fits)
    end = min(value[2] for value in fits)
    if start > end:
        raise ValueError("XYZ observed intervals do not overlap")
    # Node bounds permit only arithmetic roundoff, not missing-depth imputation.
    tolerance = 8 * np.finfo(float).eps * max(1., abs(start), abs(end))
    if time < start - tolerance or time > end + tolerance:
        return None
    return np.array([float(Legendre(c, domain=[s, e])(min(max(time, s), e))) for c, s, e in fits])


def batch_direction_support(vectors, absolute_floor=1e-12):
    """Each batch is compared with the mean of the other independent batches.

    These directional summaries are descriptive; they do not add another test or
    turn the dependent batch-pair comparisons into independent observations.
    """
    values = np.asarray(vectors, float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Finite independent-batch XYZ vectors required")
    if len(values) < 2:
        raise ValueError("At least two independent batches are required")
    leave_out = (values.sum(axis=0) - values) / (len(values) - 1)
    norm = np.linalg.norm(values, axis=1)
    other_norm = np.linalg.norm(leave_out, axis=1)
    observed = (norm > absolute_floor) & (other_norm > absolute_floor)
    cosines = np.full(len(values), np.nan)
    cosines[observed] = np.sum(values[observed] * leave_out[observed], axis=1) / (norm[observed] * other_norm[observed])
    observed_cosines = cosines[observed]
    return {
        "independent_batches": int(len(values)),
        "loo_observed_batches": int(observed.sum()),
        "loo_direction_observed_fraction": float(observed.mean()),
        "loo_positive_fraction": float(np.mean(observed_cosines > 0)) if observed.any() else 0.,
        "loo_mean_cosine": float(observed_cosines.mean()) if observed.any() else None,
        "batch_window_xyz": values.tolist(),
        "loo_cosines": [float(v) if np.isfinite(v) else None for v in cosines],
        "inference_unit": "Independent generation batches; LOO summaries are descriptive, not new p-values",
    }


def learn_rules(effects, batches, fitted, catalog, window, *, q_threshold=.05,
                minimum_batches=10, minimum_loo_support=.70, absolute_floor=1e-12):
    """One pre-specified ancestry depth per region; no generated final labels."""
    if not 0 < q_threshold < 1 or minimum_batches < 2 or not .5 <= minimum_loo_support <= 1 or absolute_floor <= 0:
        raise ValueError("Invalid generic discovery eligibility thresholds")
    points = np.asarray(catalog["landmarks_A"], float)
    width = float(catalog["region_width_A"])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or width <= 0:
        raise ValueError("Finite receptor-frame region landmarks and positive width required")
    for table, columns in ((effects, {"depth", "metric", "feature", "whole_window_mean", "CI_low", "CI_high", "q", "independent_batches_available"}),
                           (batches, {"batch", "depth", "metric", "feature", "whole_window_value"})):
        if not columns.issubset(table.columns):
            raise ValueError("Required frozen statistics columns are missing")
    regions = sorted({int(match[1]) for feature in effects.feature for match in [REGION_PATTERN.match(feature)] if match})
    candidates, rejected = [], []
    for region in regions:
        if region >= len(points):
            raise ValueError("Regional feature has no bound landmark")
        names = [f"region_{region:02d}_internal_displacement_{axis}" for axis in "xyz"]
        for depth in sorted(int(v) for v in effects.depth.unique()):
            rows = effects[(effects.depth == depth) & (effects.metric == METRIC) & effects.feature.isin(names)]
            if len(rows) != 3 or rows.feature.nunique() != 3:
                raise ValueError("Frozen covariance must contain exactly one complete XYZ triplet")
            rows = rows.set_index("feature").loc[names]
            q = rows.q.to_numpy(float)
            ci_low, ci_high = rows.CI_low.to_numpy(float), rows.CI_high.to_numpy(float)
            significant = np.isfinite(q) & (q < q_threshold) & ((ci_low > 0) | (ci_high < 0))
            if not significant.any():
                continue
            if rows.independent_batches_available.min() < minimum_batches:
                rejected.append({"region": region, "depth": depth, "reason": "insufficient independent batches"})
                continue
            records = batches[(batches.depth == depth) & (batches.metric == METRIC) & batches.feature.isin(names)]
            if records.duplicated(["batch", "feature"]).any():
                raise ValueError("Duplicate batch covariance observations are not independent")
            matrix = records.pivot(index="batch", columns="feature", values="whole_window_value").reindex(columns=names).dropna()
            if len(matrix) < minimum_batches:
                rejected.append({"region": region, "depth": depth, "reason": "missing complete batch XYZ vectors"})
                continue
            support = batch_direction_support(matrix.to_numpy(), absolute_floor)
            mean = rows.whole_window_mean.to_numpy(float)
            if not np.isfinite(mean).all():
                raise ValueError("Finite absolute covariance effect required")
            amplitude = float(np.linalg.norm(mean))
            if amplitude <= absolute_floor or support["loo_observed_batches"] < minimum_batches or support["loo_positive_fraction"] < minimum_loo_support:
                rejected.append({"region": region, "depth": depth, "reason": "absolute effect or LOO direction support below threshold", **support})
                continue
            depth_fits = fitted.get("depths", {}).get(str(depth), {}).get("functions", {})
            xyz = {axis: depth_fits.get(f"{METRIC}/{name}") for axis, name in zip("xyz", names)}
            for fit in xyz.values():
                _fit(fit)
            if any(_fit(fit)[1] < window[0] or _fit(fit)[2] > window[1] for fit in xyz.values()):
                raise ValueError("Fit scope exceeds the learned selection window")
            start = max(_fit(fit)[1] for fit in xyz.values())
            end = min(_fit(fit)[2] for fit in xyz.values())
            evaluate_xyz(xyz, start)
            candidates.append({
                "region": region, "depth": depth, "landmark_A": points[region].tolist(),
                "feature_names": names, "metric": METRIC, "component_q": q.tolist(),
                "component_CI_low": ci_low.tolist(), "component_CI_high": ci_high.tolist(),
                "significant_components": [axis for axis, good in zip("xyz", significant) if good],
                "whole_window_xyz": mean.tolist(), "absolute_covariance_norm": amplitude,
                "observed_time_start": start, "observed_time_end": end,
                "functions_xyz": copy.deepcopy(xyz), "batch_ids": matrix.index.astype(int).tolist(), **support,
            })
    chosen = []
    for region in sorted({v["region"] for v in candidates}):
        options = [v for v in candidates if v["region"] == region]
        # Larger depth has stronger conditioning; absolute magnitude must not
        # automatically privilege the longest ancestry lag.
        best = min(options, key=lambda v: (-v["loo_positive_fraction"], min(v["component_q"]), v["depth"]))
        chosen.append(best)
    return {
        "schema_version": "whole-window-regional-joint-direction-rules-1.0", "window": list(window),
        "region_width_A": width, "q_threshold": q_threshold, "minimum_independent_batches": minimum_batches,
        "minimum_loo_positive_fraction": minimum_loo_support, "absolute_covariance_floor": absolute_floor,
        "depth_selection": "Eligibility first; descending LOO positive fraction, ascending frozen minimum component q, then shallower depth",
        "selection_uses_final_generated_labels": False, "new_significance_tests": False,
        "selected_regions": chosen, "eligible_region_depths": len(candidates), "rejected_candidates": rejected,
        "field_formula": "b_ki=sum_r C_r(t)*(exp(-||T_ki-landmark_r||^2/(2*width^2))-mean_i h_ri); d_k=b_k/RMS(b_k)",
        "units": "C_r: coordinate-Angstrom times recorded score; d_k: unit atom-RMS joint displacement",
        "limitations": [
            "Observed adjusted covariance-to-coordinate direction is an unvalidated intervention hypothesis, not an affinity gradient.",
            "Whole-window support does not establish vector consistency at every step, nor causal attribution to any model module.",
            "Depth selection uses the same discovery batches; frozen q-values are not post-selection confirmatory significance.",
            "Only batches are independent; pairwise batch comparisons and resampled copies do not increase N.",
            "Missing early ancestry nodes are not extrapolated; no invented phases beyond the frozen fitted degree.",
            "Conditional selection, surviving-branch censoring and nuisance adjustment leave observational bias.",
            "Natural shift amplitude, original local eligibility and atom weights stay fixed; positive, negative and null directions require paired generation tests.",
        ],
    }


def joint_direction(teacher, time, rules, absolute_floor=1e-12):
    """One complete teacher joint mode with exactly zero translation component."""
    coordinates = np.asarray(teacher, float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0 or not np.isfinite(coordinates).all():
        raise ValueError("Finite atom-by-XYZ teacher coordinates required")
    width = float(rules["region_width_A"])
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Finite positive region width required")
    field = np.zeros_like(coordinates)
    active = []
    for region in rules["selected_regions"]:
        vector = evaluate_xyz(region["functions_xyz"], time)
        if vector is None:
            continue
        landmark = np.asarray(region["landmark_A"], float)
        if landmark.shape != (3,) or not np.isfinite(landmark).all():
            raise ValueError("Finite XYZ landmark required")
        membership = np.exp(-np.sum((coordinates - landmark) ** 2, axis=1) / (2 * width ** 2))
        field += (membership - membership.mean())[:, None] * vector
        active.append(region["region"])
    amplitude = float(np.sqrt(np.mean(np.sum(field * field, axis=1))))
    return (field / amplitude if amplitude > absolute_floor else np.zeros_like(field)), {
        "active_region_ids": active, "unnormalized_covariance_field_RMS": amplitude,
        "zero_translation_max_abs": float(np.max(np.abs(field.mean(axis=0)))),
    }


def synthesize_reference(reference, rules, *, source_reference_sha256=None):
    """Replace only directions/confidence; preserve original observational truth."""
    if list(reference.get("window", [])) != list(rules["window"]):
        raise ValueError("Reference and learned statistics windows must match exactly")
    if "branch_mutation" not in reference:
        raise ValueError("Original same-ancestor branch reference required")
    result = copy.deepcopy(reference)
    diagnostic_rows = []
    for original, frame in zip(reference["frames"], result["frames"]):
        time = float(original["time"])
        if not rules["window"][0] <= time <= rules["window"][1]:
            raise ValueError("Teacher frame outside learned scope")
        coords = original["teacher_endpoint_A"]
        old_direction = np.asarray(original["teacher_contrast_direction_unit"], float)
        confidence = np.asarray(original["teacher_contrast_confidence"], float)
        atom_weights = np.asarray(original["teacher_contrast_atom_weight"], float)
        provenance = original["teacher_contrast_provenance"]
        if old_direction.shape != np.asarray(coords).shape or confidence.shape != (len(coords),) or atom_weights.shape != np.asarray(coords).shape[:-1] or len(provenance) != len(coords):
            raise ValueError("Original branch teacher arrays are inconsistent")
        if not np.isfinite(old_direction).all() or not np.isfinite(confidence).all() or (confidence < 0).any():
            raise ValueError("Original branch directions/confidence must be finite")
        directions, confidences, nonzero, eligible_count, active, cosines = [], [], 0, 0, set(), []
        for teacher, old, c, evidence in zip(coords, old_direction, confidence, provenance):
            direction, diagnostic = joint_direction(teacher, time, rules, rules["absolute_covariance_floor"])
            active.update(diagnostic["active_region_ids"])
            has_direction = bool(np.any(direction))
            # Do not manufacture missing original branch eligibility. Frozen
            # controller checks these same original local mutation facts.
            eligible = c > 0 and bool(evidence.get("lower_immediate_parent_slots")) and float(evidence.get("raw_direction_RMS_A", 0)) > 0
            if not eligible:
                direction = np.zeros_like(direction)
            directions.append(direction.tolist())
            confidences.append(float(c) if np.any(direction) else 0.)
            eligible_count += int(eligible)
            nonzero += int(bool(np.any(direction)))
            if eligible and np.any(direction) and np.linalg.norm(old) > 0:
                cosines.append(float(np.sum(direction * old) / (np.linalg.norm(direction) * np.linalg.norm(old))))
        frame["teacher_original_branch_direction_unit"] = copy.deepcopy(original["teacher_contrast_direction_unit"])
        frame["teacher_original_branch_confidence"] = copy.deepcopy(original["teacher_contrast_confidence"])
        frame["teacher_contrast_direction_unit"] = directions
        frame["teacher_contrast_confidence"] = confidences
        diagnostic_rows.append({"time": time, "teachers": len(coords), "original_eligible_teachers": eligible_count,
                                "new_active_teachers": nonzero, "active_region_ids": sorted(active),
                                "mean_cosine_to_original_direction": float(np.mean(cosines)) if cosines else None})
    result["regional_reference"] = {
        "schema_version": "whole-window-regional-joint-reference-1.0",
        "direction_source": "Frozen adjusted covariance XYZ fits; teacher-endpoint density membership; centered complete joint mode",
        "source_reference_sha256": source_reference_sha256, "window": list(rules["window"]),
        "original_branch_direction_field": "teacher_original_branch_direction_unit",
        "original_branch_confidence_field": "teacher_original_branch_confidence",
        "original_provenance_unmodified": True, "original_atom_weights_unmodified": True,
        "amplitude_source": "Original branch raw_direction_RMS_A and original lower-mutation eligibility",
        "is_affinity_gradient": False, "frame_coverage": diagnostic_rows,
    }
    return result, {
        "teachers": sum(row["teachers"] for row in diagnostic_rows),
        "original_eligible_teachers": sum(row["original_eligible_teachers"] for row in diagnostic_rows),
        "new_active_teachers": sum(row["new_active_teachers"] for row in diagnostic_rows),
        "frames": diagnostic_rows,
    }


def compact_json_bytes(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")


def prepare_inputs(mining, reference, source_receipt, branch_receipt):
    """Validate both live source tools and bind every upstream file they attest."""
    mining, reference = Path(mining).resolve(), Path(reference).resolve()
    multi, bound = validate_live_receipt(source_receipt, "multi_depth_mutation")
    branch, branch_bound = validate_live_receipt(branch_receipt, "branch_mutation")
    for name, sha in branch_bound.items():
        if name in bound and bound[name] != sha:
            raise ValueError("Upstream receipts disagree on an input/source hash")
        bound[name] = sha
    for name in REQUIRED_FILES:
        path = mining / name
        if str(path) not in multi["output_files"]:
            raise ValueError("Mining data is not an actual multi-depth tool output")
    if str(reference) not in branch["output_files"]:
        raise ValueError("Branch reference is not an actual branch tool output")
    manifest = read_json(mining / "manifest.json")
    for name, spec in manifest["files"].items():
        if digest(mining / name) != spec["sha256"]:
            raise ValueError("Mining manifest output hash mismatch")
    original = json.loads(gzip.decompress(reference.read_bytes()))
    if original["window"] != manifest["window"]:
        raise ValueError("Multi-depth and original branch scopes differ")
    if original["branch_mutation"]["parent_reference_sha256"] != manifest["reference_sha256"]:
        raise ValueError("Branch and multi-depth tools have different original teacher libraries")
    return bound


def synthesize(mining, reference, output, output_reference, *, q_threshold=.05,
               minimum_batches=10, minimum_loo_support=.70, absolute_floor=1e-12):
    """Execute the pure data synthesis. Caller captures its actual live receipt."""
    mining, reference = Path(mining).resolve(), Path(reference).resolve()
    output, output_reference = Path(output).resolve(), Path(output_reference).resolve()
    if output.exists() or output_reference.exists():
        raise FileExistsError("Fresh summary directory and new reference path required")
    manifest = read_json(mining / "manifest.json")
    original = json.loads(gzip.decompress(reference.read_bytes()))
    rules = learn_rules(pd.read_parquet(mining / "whole_window_effects.parquet"),
                        pd.read_parquet(mining / "batch_window_statistics.parquet"),
                        read_json(mining / "fitted_trends.json"), read_json(mining / "feature_catalog.json"),
                        manifest["window"], q_threshold=q_threshold, minimum_batches=minimum_batches,
                        minimum_loo_support=minimum_loo_support, absolute_floor=absolute_floor)
    rules["source_manifest_sha256"] = digest(mining / "manifest.json")
    rules["source_reference_sha256"] = digest(reference)
    rules["source_evidence_sha256"] = digest(mining / "evidence.json")
    transformed, coverage = synthesize_reference(original, rules, source_reference_sha256=digest(reference))
    rules["teacher_coverage"] = coverage
    rule_bytes = compact_json_bytes(rules)
    if len(rule_bytes) >= 1024 * 1024:
        raise ValueError("Compact rule/coverage summary exceeds 1 MiB")
    output.mkdir(parents=True)
    rule_path = output / "regional_rules.json"
    rule_path.write_bytes(rule_bytes)
    transformed["regional_reference"]["rules_sha256"] = digest(rule_path)
    output_reference.parent.mkdir(parents=True, exist_ok=True)
    output_reference.write_bytes(gzip.compress(compact_json_bytes(transformed), mtime=0))
    report = {
        "schema_version": "regional-reference-manifest-1.0", "window": original["window"],
        "selected_regions": len(rules["selected_regions"]), "source_reference_sha256": digest(reference),
        "source_manifest_sha256": digest(mining / "manifest.json"), "coverage": {k: v for k, v in coverage.items() if k != "frames"},
        "reference_path": str(output_reference), "reference_sha256": digest(output_reference),
        "rules_sha256": digest(rule_path), "rules_bytes": len(rule_bytes),
        "new_model_or_affinity_head": False, "generation_performed": False,
        "immutable_original_fields": "All original fields except teacher_contrast_direction_unit and zero-field confidence; originals retained explicitly",
    }
    (output / "manifest.json").write_bytes(compact_json_bytes(report))
    return report
