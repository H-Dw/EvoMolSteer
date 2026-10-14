"""Deterministic splits of observed source batches, independent of outcomes."""
from pathlib import Path
import numpy as np
from ..trajectory_source import trajectory_paths


def split_batches(available, fractions=None, seed=42):
    available = sorted({int(v) for v in available})
    if not available or any(v < 0 for v in available):
        raise ValueError('Nonempty observed batch identifiers required')
    fractions = fractions or {'discovery': .7, 'validation': .15, 'heldout': .15}
    names = list(fractions)
    weights = np.asarray(list(fractions.values()), float)
    if len(names) != len(set(names)) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Positive named split fractions required')
    sizes = weights / weights.sum() * len(available)
    counts = np.floor(sizes).astype(int)
    for i in np.argsort(-(sizes-counts), kind='stable')[:len(available)-counts.sum()]:
        counts[i] += 1
    shuffled = np.random.default_rng(seed).permutation(available)
    cuts = np.r_[0, counts.cumsum()]
    return {name: sorted(map(int, shuffled[cuts[i]:cuts[i+1]])) for i, name in enumerate(names)}


def source_batches(dataset, campaign, arm='single'):
    paths = trajectory_paths(Path(dataset)/'results'/campaign)
    result = {int(p.parent.name.split('_')[-1]): p for p in paths if p.parent.parent.name == arm}
    if not result:
        raise ValueError('No observed trajectories for requested campaign/arm')
    return result


def resolve_batches(dataset, campaign, batches=None, *, arm='single', fractions=None, seed=42, subset='discovery'):
    available = source_batches(dataset, campaign, arm)
    chosen = (split_batches(available, fractions, seed)[subset] if batches is None else list(map(int, batches)))
    if not chosen or len(chosen) != len(set(chosen)) or set(chosen)-set(available):
        raise ValueError('Requested batches absent, repeated or empty')
    return sorted(chosen)
