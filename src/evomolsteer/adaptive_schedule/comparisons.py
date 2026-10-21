"""Paired validity transitions and synchronized native-batch uncertainty."""
import numpy as np

from .io import eligible


def validity_transitions(before, after):
    if len(before) != len(after):
        raise ValueError("Paired comparisons require the same original slots")
    result = {"attempts": len(before), "both_valid": 0, "invalid_to_valid": 0,
              "valid_to_invalid": 0, "both_invalid": 0}
    for a, b in zip(before, after):
        old, new = eligible(a), eligible(b)
        key = "both_valid" if old and new else "invalid_to_valid" if new else "valid_to_invalid" if old else "both_invalid"
        result[key] += 1
    return result


def shared_batch_bootstrap(rows, teacher_ids, field, seed=42):
    # All teachers receive the same resampled batch IDs. Shared starts are never
    # resampled independently across teachers. Missing paired scores stay missing.
    result = {"field": field, "teachers": len(teacher_ids), "paired_seed_batches": 5,
              "teacher_equal_mean_delta": None, "exploratory_ci": None}
    grouped = {(t, b): [r[field] for r in rows if r["teacher_id"] == t and r["batch"] == b
                        and r.get(field) is not None] for t in teacher_ids for b in range(5)}
    if not teacher_ids or any(not grouped[t, b] for t in teacher_ids for b in range(5)):
        result["reason"] = "Every teacher/batch requires a finite paired observation; no zero imputation"
        return result
    result["teacher_equal_mean_delta"] = float(np.mean([
        np.mean([v for b in range(5) for v in grouped[t, b]]) for t in teacher_ids]))
    draws = np.random.default_rng(seed).integers(0, 5, size=(5000, 5))
    means = []
    for t in teacher_ids:
        sums = np.array([sum(grouped[t, b]) for b in range(5)])
        counts = np.array([len(grouped[t, b]) for b in range(5)])
        means.append(sums[draws].sum(1) / counts[draws].sum(1))
    result["exploratory_ci"] = np.quantile(np.mean(means, axis=0), [.025, .975]).tolist()
    return result
