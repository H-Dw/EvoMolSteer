"""Permutation-invariant regional coordinates and matched-slot transport proxies.

The frame is the aligned, fixed receptor frame in angstrom. Endpoint atom labels
are a conditional, detached mask, not a claim that early noise has chemistry.
"""
import numpy as np
from scipy.special import logsumexp


def regional_observables(current, endpoint, proposal, atoms, mask, catalog, t, dt,
                         spatial_width_A=4.0):
    if not (0<=t<1 and dt>0 and spatial_width_A>0):raise ValueError('Invalid transport domain or spatial width')
    names, columns, metadata = [], [], {}
    mask = np.asarray(mask, bool)
    delta = endpoint-current
    native = (proposal-current)/dt
    for region, record in sorted(catalog['regions'].items()):
        points = np.asarray(record['points_A'], float)
        origin = points.mean(0)
        distance = np.linalg.norm(current[:, :, None]-points[None, None], axis=-1)
        for channel in ('all', 'NOS'):
            eligible = mask.copy()
            if channel == 'NOS':
                eligible &= np.isin(atoms, [catalog['atom_vocabulary'][a] for a in ('N','O','S')])
            valid = eligible.any(1)
            logw = -.5*(distance.min(-1)/spatial_width_A)**2
            logw = np.where(eligible, logw, -np.inf)
            # Make empty masks safe; validity is still explicitly recorded.
            safe = np.where(valid[:, None], logw, np.where(mask, 0., -np.inf))
            weights = np.exp(safe-logsumexp(safe, axis=1)[:, None])
            center = (weights[..., None]*current).sum(1)
            spread = np.sqrt((weights*((current-center[:, None])**2).sum(-1)).sum(1))
            residual = (weights[..., None]*delta).sum(1)/(1-t)
            velocity = (weights[..., None]*native).sum(1)
            speed = np.sqrt((weights*(native**2).sum(-1)).sum(1))
            values = {**{f'centroid_{a}':center[:, k]-origin[k] for k,a in enumerate('xyz')},
                      'spread':spread,
                      **{f'endpoint_transport_{a}':residual[:, k] for k,a in enumerate('xyz')},
                      **{f'native_velocity_{a}':velocity[:, k] for k,a in enumerate('xyz')},
                      'native_speed':speed}
            for kind, value in values.items():
                name = f'{region}::{channel}::{kind}'
                names.append(name);columns.append(np.where(valid, value, np.nan))
                metadata[name] = {'region':region,'channel':channel,'kind':kind,
                    'unit':'A' if kind.startswith('centroid') or kind=='spread' else 'A/time',
                    'coordinate_control':kind.startswith('centroid') or kind=='spread',
                    'mask':'predicted endpoint hard labels held fixed',
                    'interpretation':'endpoint residual proxy' if kind.startswith('endpoint') else
                        'measured native step including stochastic/corrector terms' if kind.startswith('native') else
                        'current-state spatial Gaussian patch; permutation invariant'}
    return np.column_stack(columns), names, metadata


def regional_moments(current, atoms, mask, points, eligible_labels, width=4.):
    """Four coordinate observables shared by reference building and torch reward."""
    eligible = np.asarray(mask, bool).copy()
    if eligible_labels is not None: eligible &= np.isin(atoms, eligible_labels)
    valid = eligible.any(1)
    d = np.linalg.norm(current[:, :, None]-np.asarray(points)[None, None], axis=-1).min(-1)
    logits = np.where(eligible, -.5*(d/width)**2, -np.inf)
    logits = np.where(valid[:, None], logits, np.where(mask, 0., -np.inf))
    w = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    mu = (w[..., None]*current).sum(1)
    spread = np.sqrt((w*((current-mu[:, None])**2).sum(-1)).sum(1))
    z = np.column_stack([mu-np.asarray(points).mean(0), spread])
    return np.where(valid[:, None], z, np.nan)
