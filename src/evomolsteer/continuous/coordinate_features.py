"""Permutation-invariant regional coordinates and matched-slot transport proxies.

The frame is the aligned, fixed receptor frame in angstrom. Endpoint atom labels
are a conditional, detached mask, not a claim that early noise has chemistry.
"""
import numpy as np
from scipy.special import logsumexp


def regional_observables(current, endpoint, proposal, atoms, mask, catalog, t, dt,
                         spatial_width_A=4.0,spatial_anchor='current',include_proposal=False,feature_family='geometry'):
    if not (0<=t<1 and dt>0 and spatial_width_A>0):raise ValueError('Invalid transport domain or spatial width')
    if feature_family not in ('geometry','transport','joint','shape','composition'):raise ValueError('Unknown feature family')
    names, columns, metadata = [], [], {}
    mask = np.asarray(mask, bool)
    if feature_family=='composition':
        # Global type counts are measured once, not repeated for each region.
        # These are endpoint predictions on noisy slots, not early chemistry.
        for element in ('N','O','S'):
            name=f'whole_ligand::all::predicted_{element}_count'
            names.append(name);columns.append(((atoms==catalog['atom_vocabulary'][element])&mask).sum(1).astype(float))
            metadata[name]={'region':'whole_ligand','channel':'all','kind':f'predicted_{element}_count',
                'unit':'count','coordinate_control':False,'spatial_anchor':'not_applicable','representation':'predicted endpoint hard labels',
                'interpretation':'Global masked endpoint-labelled slot count; not physical early chemistry or a discrete gradient target'}
        return np.column_stack(columns),names,metadata
    delta = endpoint-current
    native = (proposal-current)/dt
    for region, record in sorted(catalog['regions'].items()):
        points = np.asarray(record['points_A'], float)
        origin = points.mean(0)
        if spatial_anchor not in ('current','endpoint'):raise ValueError('Unknown spatial anchor')
        anchor=current if spatial_anchor=='current' else endpoint
        distance = np.linalg.norm(anchor[:, :, None]-points[None, None], axis=-1)
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
            if include_proposal:
                pc=(weights[...,None]*proposal).sum(1)
                ps=np.sqrt((weights*((proposal-pc[:,None])**2).sum(-1)).sum(1))
                values.update({**{f'proposal_centroid_{a}':pc[:,k]-origin[k] for k,a in enumerate('xyz')},'proposal_spread':ps})
            if feature_family=='shape':
                # The fixed receptor frame retains directional shape information
                # discarded by scalar spread. One eligible slot has a legal zero
                # covariance and no shape direction; diagnostics retain that fact.
                values={}
                values['shape_eligible_slots']=eligible.sum(1).astype(float)
                values['shape_informative']=(eligible.sum(1)>=2).astype(float)
                values['shape_effective_slots']=1/(weights**2).sum(1)
                for prefix,state in [('shape_',current)]+([('proposal_shape_',proposal)] if include_proposal else []):
                    mu=(weights[...,None]*state).sum(1);centered=state-mu[:,None]
                    tensor=np.einsum('bn,bni,bnj->bij',weights,centered,centered)
                    values[prefix+'trace']=np.trace(tensor,axis1=1,axis2=2)
                    for pair,i,j in [('xx',0,0),('yy',1,1),('zz',2,2),('xy',0,1),('xz',0,2),('yz',1,2)]:
                        values[prefix+pair]=tensor[:,i,j]*(np.sqrt(2) if i!=j else 1.)
            elif feature_family!='geometry':
                remaining=np.sqrt((weights*(delta**2).sum(-1)).sum(1))
                den=remaining*speed
                alignment=np.divide((weights*(delta*native).sum(-1)).sum(1),den,
                    out=np.full(len(current),np.nan),where=den>1e-12).clip(-1,1)
                core_anchor=distance.min(-1)<=5.
                core_current=np.linalg.norm(current[:,:,None]-points[None,None],axis=-1).min(-1)<=5.
                transport={'remaining_rms':remaining,'transport_coherence':alignment,
                    'effective_slots':1/(weights**2).sum(1),
                    'anchor_core_weight':(weights*core_anchor).sum(1),
                    'current_core_weight':(weights*core_current).sum(1)}
                values=transport if feature_family=='transport' else {**values,**transport}
            for kind, value in values.items():
                name = f'{region}::{channel}::{kind}'
                names.append(name);columns.append(value if kind in ('shape_eligible_slots','shape_informative') else np.where(valid, value, np.nan))
                metadata[name] = {'region':region,'channel':channel,'kind':kind,
                    'unit':'dimensionless' if kind in ('shape_eligible_slots','shape_informative','shape_effective_slots') else 'A^2' if 'shape_' in kind else 'A' if 'centroid' in kind or 'spread' in kind or kind=='remaining_rms' else
                        'dimensionless' if kind in ('transport_coherence','effective_slots','anchor_core_weight','current_core_weight') else 'A/time',
                    'coordinate_control':'centroid' in kind or 'spread' in kind or ('shape_' in kind and kind not in ('shape_eligible_slots','shape_informative','shape_effective_slots')),
                    'spatial_anchor':spatial_anchor,
                    'representation':'proposal' if kind.startswith('proposal') else 'current' if 'centroid' in kind or 'spread' in kind or 'shape_' in kind else 'matched-slot transport',
                    'mask':'predicted endpoint hard labels held fixed',
                    'interpretation':'Number of eligible endpoint-labelled slots; not physical early chemistry' if kind=='shape_eligible_slots' else
                        'At least two eligible slots; one-slot covariance is a legal zero with no shape direction' if kind=='shape_informative' else
                        'Conditional Gaussian weight concentration, not physical contacts or independent atoms' if kind=='shape_effective_slots' else
                        'Fixed-receptor-frame weighted central second moment; sqrt(2) off-diagonal scaling; trace repeats spread squared; not unique molecular geometry or physical energy' if 'shape_' in kind else
                        'weighted matched-slot cosine of remaining displacement and measured native velocity; SDE/corrector included' if kind=='transport_coherence' else
                        'Gaussian weight concentration; not count of physical contacts' if kind=='effective_slots' else
                        'Gaussian mass within fixed 5 A diagnostic radius; not a reward threshold' if kind.endswith('core_weight') else
                        'weighted remaining matched-slot displacement; not energy' if kind=='remaining_rms' else
                        'endpoint residual proxy' if kind.startswith('endpoint') else
                        'measured native step including stochastic/corrector terms' if kind.startswith('native') else
                        f'{spatial_anchor}-anchored atom slots, {"proposal" if kind.startswith("proposal") else "current"} coordinates; permutation invariant'}
    return np.column_stack(columns), names, metadata


def regional_moments(current, atoms, mask, points, eligible_labels, width=4.,anchor=None):
    """Four coordinate observables shared by reference building and torch reward."""
    eligible = np.asarray(mask, bool).copy()
    if eligible_labels is not None: eligible &= np.isin(atoms, eligible_labels)
    valid = eligible.any(1)
    if anchor is None:anchor=current
    d = np.linalg.norm(anchor[:, :, None]-np.asarray(points)[None, None], axis=-1).min(-1)
    logits = np.where(eligible, -.5*(d/width)**2 if width is not None else 0., -np.inf)
    logits = np.where(valid[:, None], logits, np.where(mask, 0., -np.inf))
    w = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    mu = (w[..., None]*current).sum(1)
    spread = np.sqrt((w*((current-mu[:, None])**2).sum(-1)).sum(1))
    z = np.column_stack([mu-np.asarray(points).mean(0), spread])
    return np.where(valid[:, None], z, np.nan)
