"""Compact formation/retention measurements on the declared local region.

Actual t=0.5 geometry is compared with the teacher's actual t=0.5 state.
This diagnostic is distinct from the endpoint-prediction reward observable.
Half-state correspondence is kept fixed for the unguided terminal suffix.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from evomolsteer.ancestral_modes.bank import lookup


def measure(half,terminal,mask,teacher,program):
    spec=program['mode_specs'][0];mode=teacher['modes'][spec['template_id']]
    slots=spec['atom_slots'];target=lookup(mode,program['teacher_time'])[slots]
    sites=np.asarray([v['coords_A'] for v in spec['pocket_sites']])
    rows=[]
    for x,y,keep in zip(half,terminal,mask):
        ids=np.flatnonzero(keep);cost=np.square(x[ids,None]-target[None]).sum(-1)
        rr,cc=linear_sum_assignment(cost);assignment=ids[rr[np.argsort(cc)]]
        def geometry(z):
            point=float(np.sqrt(np.square(z-target).sum(-1).mean()))
            a=np.linalg.norm(z[:,None]-z[None],axis=-1)
            b=np.linalg.norm(target[:,None]-target[None],axis=-1)
            tri=np.triu_indices(len(z),1)
            pair=float(np.sqrt(np.square(a[tri]-b[tri]).mean()))
            pocket=None
            if len(sites):
                a=np.linalg.norm(z[:,None]-sites[None],axis=-1)
                b=np.linalg.norm(target[:,None]-sites[None],axis=-1)
                pocket=float(np.sqrt(np.square(a-b).mean()))
            return {'point_error_A':point,'pair_error_A':pair,'pocket_error_A':pocket}
        h=geometry(x[assignment]);t=geometry(y[assignment])
        rows.append({'support_size':len(slots),'generated_slots_at_half':assignment.tolist(),
            **{'half_'+k:v for k,v in h.items()},**{'terminal_tracked_'+k:v for k,v in t.items()},
            'suffix_tracked_displacement_A':float(np.sqrt(np.square(y[assignment]-x[assignment]).sum(-1).mean()))})
    return rows
