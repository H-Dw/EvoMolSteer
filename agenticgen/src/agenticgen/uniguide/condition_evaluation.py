"""Condition-only proper rigid fitting for evaluation, never for guidance gradients."""
import math
import numpy as np

POLICY = "condition-kabsch-mean-v1"
CONFIG = "condition_evaluation"
METRIC = "condition_aligned_rmsd_angstrom"
MEAN = "mean_condition_aligned_rmsd_angstrom"
POSE = "protein_frame_pose_rmsd_angstrom"


def enabled(value):
    return value.get(CONFIG, {}).get("policy") == POLICY


def fit(selected, target):
    """Fit fixed correspondences; accept degenerate sets without choosing new atoms."""
    x, y = np.asarray(selected, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or not len(x):
        raise ValueError("Matching nonempty condition point arrays required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("Nonfinite condition coordinates cannot be omitted")
    a, b = x - x.mean(0), y - y.mean(0)
    try:
        with np.errstate(over='raise', invalid='raise'):
            u, _, vt = np.linalg.svd(a.T @ b)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError('Condition fit did not converge; population is not rankable') from error
    rotation = u @ np.diag([1., 1., 1. if np.linalg.det(u @ vt) >= 0 else -1.]) @ vt
    translation = y.mean(0) - x.mean(0) @ rotation
    error = a @ rotation - b
    rmsd = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    if not math.isfinite(rmsd):
        raise FloatingPointError("Nonfinite fitted condition RMSD")
    return dict(rmsd_angstrom=rmsd, rotation=rotation.tolist(), translation_angstrom=translation.tolist())


def evaluate(endpoint, condition, assignment):
    from .absolute_coordinate import aligned, reference
    xyz = aligned(endpoint.detach().cpu().double(), condition).numpy()
    target = reference(condition, endpoint.detach().cpu().double()).numpy()
    slots = assignment.detach().cpu().numpy()
    if (xyz.ndim != 3 or slots.shape != (len(xyz),len(target))
            or slots.dtype.kind not in 'iu' or (slots < 0).any() or (slots >= xyz.shape[1]).any()
            or any(len(set(ids)) != len(ids) for ids in slots)):
        raise ValueError('One fixed injective condition assignment per candidate required')
    readings = [fit(points[ids], target) for points, ids in zip(xyz, slots)]
    return [r["rmsd_angstrom"] for r in readings], readings


def summarize(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise FloatingPointError("Missing/nonfinite aligned RMSD cannot be removed or replaced")
    with np.errstate(over='raise', invalid='raise'):
        mean = float(np.mean(np.asarray(values, dtype=np.float64)))
    return dict(policy_id=POLICY, expected=50, measured=len(values),
                arithmetic_mean=mean,
                retained_count=len(values), excluded_high_count=0, filtering_applied=False,
                ranking_eligible=len(values) == 50, status="complete" if len(values) == 50 else "incomplete_population",
                unit="angstrom", scope="fixed terminal condition correspondences only", gradient_modified=False)


def score(row):
    summary = row.get("coordinate_loss_statistics", {})
    value = row.get(MEAN)
    if (not enabled(row) or row.get("attempts") != 50 or not row.get("complete")
            or summary.get("policy_id") != POLICY or not summary.get("ranking_eligible")
            or summary.get("measured") != 50 or summary.get("retained_count") != 50
            or summary.get('expected') != 50 or summary.get('filtering_applied') is not False
            or summary.get("excluded_high_count") != 0
            or type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or summary.get("arithmetic_mean") != value
            or row.get("ranking_condition_coordinate_root_mean_square_deviation_angstrom") != value):
        raise ValueError("New evaluation requires a complete unfiltered condition-aligned mean; no legacy fallback")
    return value
