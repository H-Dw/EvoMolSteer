"""Identity-free conditional spatial motifs; numpy/torch share the definition.

Shell and pair kernels describe geometry, never physical energies or bonds.
Endpoint labels/anchors are detached conditional annotations on noisy slots.
"""
import numpy as np
from scipy.special import softmax

RADII = (2., 4., 6.)
PAIR_RADII = (1.5, 3., 5.)
KINDS = ('centroid_x', 'centroid_y', 'centroid_z',
         'shell_2', 'shell_4', 'shell_6', 'pair_1_5', 'pair_3', 'pair_5')


def numpy_motifs(x, atoms, mask, catalog, anchor, width=4., channels=('all', 'NOS', 'NOS_C')):
    names, columns, metadata = [], [], {}
    for region, record in sorted(catalog['regions'].items()):
        points = np.asarray(record['points_A'], float)
        d_anchor = np.linalg.norm(anchor[:, :, None]-points[None, None], axis=-1).min(-1)
        d_current = np.linalg.norm(x[:, :, None]-points[None, None], axis=-1).min(-1)
        pair = np.linalg.norm(x[:, :, None]-x[:, None], axis=-1)
        for channel in channels:
            eligible = np.asarray(mask, bool).copy()
            if channel == 'NOS':
                eligible &= np.isin(atoms, [catalog['atom_vocabulary'][e] for e in ('N', 'O', 'S')])
            valid = eligible.sum(1) >= 2
            cross = None
            if channel == 'NOS_C':
                polar = mask & np.isin(atoms,[catalog['atom_vocabulary'][e] for e in ('N','O','S')])
                carbon = mask & (atoms==catalog['atom_vocabulary']['C'])
                eligible = polar | carbon;valid = polar.any(1)&carbon.any(1)
                cross = (polar[:,:,None]&carbon[:,None])|(carbon[:,:,None]&polar[:,None])
            safe = np.where(eligible.any(1)[:, None], eligible, mask)
            weights = softmax(np.where(safe, -.5*(d_anchor/width)**2, -np.inf), axis=1)
            mu = (weights[..., None]*x).sum(1)-points.mean(0)
            pair_weight = weights[:, :, None]*weights[:, None]
            pair_weight *= ~np.eye(x.shape[1], dtype=bool)[None]
            if cross is not None:pair_weight *= cross
            pair_weight /= np.maximum(pair_weight.sum((1, 2))[:, None, None], 1e-30)
            values = [mu[:, k] for k in range(3)]
            values += [(weights*np.exp(-.5*((d_current-r)/1.)**2)).sum(1) for r in RADII]
            values += [(pair_weight*np.exp(-.5*((pair-r)/.75)**2)).sum((1, 2)) for r in PAIR_RADII]
            for kind, value in zip(KINDS, values):
                name = f'{region}::{channel}::motif_{kind}'
                names.append(name); columns.append(np.where(valid, value, np.nan))
                metadata[name] = {'region': region, 'channel': channel, 'kind': 'motif_'+kind,
                    'unit': 'A' if 'centroid' in kind else 'dimensionless', 'coordinate_control': True,
                    'interpretation': 'Conditional permutation-invariant geometry; not a hydrogen bond, physical contact, chemical graph or energy',
                    'representation': 'actual state; fixed endpoint spatial membership and labels'}
    return np.column_stack(columns), names, metadata


def torch_motifs(x, atoms, mask, catalog, anchor, width=4., channels=('all', 'NOS', 'NOS_C'), core_radius=5.):
    import torch
    anchor = anchor.detach()
    if atoms.ndim == 3: atoms = atoms.detach().argmax(-1)
    columns, validity = [], []
    core = torch.zeros_like(mask, dtype=torch.bool)
    for record in [v for _, v in sorted(catalog['regions'].items())]:
        points = x.new_tensor(record['points_A'])
        d_anchor = (anchor[:, :, None]-points[None, None]).square().sum(-1).clamp_min(1e-24).sqrt().amin(-1)
        d_current = (x[:, :, None]-points[None, None]).square().sum(-1).clamp_min(1e-24).sqrt().amin(-1)
        pair = (x[:, :, None]-x[:, None]).square().sum(-1).clamp_min(1e-24).sqrt()
        for channel in channels:
            eligible = mask.bool().clone()
            if channel == 'NOS': eligible &= torch.isin(atoms, atoms.new_tensor([catalog['atom_vocabulary'][e] for e in ('N','O','S')]))
            valid=eligible.sum(1)>=2;cross=None
            if channel == 'NOS_C':
                polar=mask.bool()&torch.isin(atoms,atoms.new_tensor([catalog['atom_vocabulary'][e] for e in ('N','O','S')]))
                carbon=mask.bool()&(atoms==catalog['atom_vocabulary']['C'])
                eligible=polar|carbon;valid=polar.any(1)&carbon.any(1)
                cross=(polar[:,:,None]&carbon[:,None])|(carbon[:,:,None]&polar[:,None])
            validity.append(valid)
            safe = torch.where(eligible.any(1)[:, None], eligible, mask.bool())
            w = (-.5*(d_anchor/width).square()).masked_fill(~safe, -torch.inf).softmax(1)
            mu = (w[..., None]*x).sum(1)-points.mean(0)
            pw = w[:, :, None]*w[:, None]
            pw = pw*(~torch.eye(x.shape[1], dtype=torch.bool, device=x.device))[None]
            if cross is not None:pw=pw*cross
            pw = pw/pw.sum((1, 2))[:, None, None].clamp_min(1e-30)
            values = [mu[:, k:k+1] for k in range(3)]
            values += [(w*torch.exp(-.5*((d_current-r)/1.).square())).sum(1, keepdim=True) for r in RADII]
            values += [(pw*torch.exp(-.5*((pair-r)/.75).square())).sum((1,2))[:,None] for r in PAIR_RADII]
            columns.extend(values); core |= eligible & (d_anchor <= core_radius)
    return torch.cat(columns, 1), torch.stack(validity).all(0), core
