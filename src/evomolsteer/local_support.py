"""Size-independent support policies shared by mining and reward compilation.

Limits are configuration, not a requested number of atoms. Frozen historical
programs without a policy retain their original numerical meaning.
"""
import math
from numbers import Integral


def bounds(n_atoms, policy):
    """Return inclusive support bounds; never silently widen a local policy."""
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, Integral) or n_atoms < 3:
        raise ValueError('A local support requires at least three teacher atoms')
    low = policy.get('min_points', 2)
    high = policy.get('max_points')
    fraction = policy.get('max_fraction')
    if isinstance(low, bool) or not isinstance(low, Integral) or low < 2:
        raise ValueError('min_points must be an integer >= 2')
    if high is not None and (isinstance(high, bool) or not isinstance(high, Integral) or high < low):
        raise ValueError('max_points must be an integer >= min_points')
    if fraction is not None and (isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError('max_fraction must be strictly between zero and one')
    caps = [n_atoms - 1]
    if high is not None:
        caps.append(int(high))
    if fraction is not None:
        caps.append(math.floor(n_atoms * fraction + 1e-12))
    if high is None and fraction is None:
        raise ValueError('Declare max_points or max_fraction; no implicit point-count target')
    cap = min(caps)
    if cap < low:
        raise ValueError('No feasible local support for this molecule and policy')
    return int(low), int(cap)


def validate_slots(slots, n_atoms, policy):
    low, cap = bounds(n_atoms, policy)
    if (any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 or i >= n_atoms for i in slots)
            or len(set(slots)) != len(slots) or not low <= len(slots) <= cap):
        raise ValueError('Support violates its declared local size/coverage policy')
    return {'n_points': len(slots), 'n_total': int(n_atoms), 'fraction': len(slots) / n_atoms,
            'allowed_min_points': low, 'allowed_max_points': cap}


def nested_patches(patches, parent, policy=None, teacher_sizes=None):
    """True nested subsets, not a fixed allowlist of cardinalities."""
    support = set(parent['atom_slots'])
    selected = []
    for patch in patches:
        slots = patch['atom_slots']
        if patch['template_id'] != parent['template_id'] or not set(slots).issubset(support):
            continue
        if policy is not None:
            low = policy.get('min_points', 2)
            high = policy.get('max_points', len(support))
            if policy.get('max_fraction') is not None:
                n_atoms = (teacher_sizes or {}).get(parent['template_id'], parent.get('n_total'))
                if n_atoms is None:
                    raise ValueError('Fractional nested support selection needs the teacher atom count')
                low, high = bounds(n_atoms, policy)
            if not low <= len(slots) <= high:
                continue
        selected.append(patch['patch_id'])
    return selected


def candidate_sizes(n_atoms, required_points, policy):
    """Enumerate feasible sizes; the compiler may subsequently deduplicate them."""
    low, high = bounds(n_atoms, policy)
    return list(range(max(low, required_points), high + 1))
