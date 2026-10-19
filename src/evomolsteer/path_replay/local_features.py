"""Compact formation/retention measurements on the declared local region.

Actual t=0.5 geometry is compared with the teacher's actual t=0.5 state.
This diagnostic is distinct from the endpoint-prediction reward observable.
Half-state correspondence is kept fixed for the unguided terminal suffix.
Terminal geometry uses the real terminal teacher, never its half-time state.
Reward-boundary assignments can be supplied separately from observational matches.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from evomolsteer.ancestral_modes.bank import lookup


def measure(half,terminal,mask,teacher,program,assignments=None):
    spec=program['mode_specs'][0];mode=teacher['modes'][spec['template_id']]
    slots=spec['atom_slots'];target=lookup(mode,program['teacher_time'])[slots]
    final_target=np.asarray(mode.get('terminal_coords_A',mode['coords_A']))[slots]
    sites=np.asarray([v['coords_A'] for v in spec['pocket_sites']])
    rows=[]
    for row,(x,y,keep) in enumerate(zip(half,terminal,mask)):
        ids=np.flatnonzero(keep)
        if assignments is None:
            cost=np.square(x[ids,None]-target[None]).sum(-1)
            rr,cc=linear_sum_assignment(cost);assignment=ids[rr[np.argsort(cc)]]
        else:
            assignment=np.asarray(assignments[row],dtype=int)
            if len(assignment)!=len(slots) or len(set(assignment))!=len(slots) or not set(assignment)<=set(ids):
                raise ValueError('Boundary reward assignment differs from the declared support')
        def geometry(z,reference):
            point=float(np.sqrt(np.square(z-reference).sum(-1).mean()))
            a=np.linalg.norm(z[:,None]-z[None],axis=-1)
            b=np.linalg.norm(reference[:,None]-reference[None],axis=-1)
            tri=np.triu_indices(len(z),1)
            pair=float(np.sqrt(np.square(a[tri]-b[tri]).mean()))
            pocket=None
            if len(sites):
                a=np.linalg.norm(z[:,None]-sites[None],axis=-1)
                b=np.linalg.norm(reference[:,None]-sites[None],axis=-1)
                pocket=float(np.sqrt(np.square(a-b).mean()))
            return {'point_error_A':point,'pair_error_A':pair,'pocket_error_A':pocket}
        h=geometry(x[assignment],target);t=geometry(y[assignment],final_target)
        rr,cc=linear_sum_assignment(np.square(y[ids,None]-final_target[None]).sum(-1))
        rematched=geometry(y[ids[rr[np.argsort(cc)]]],final_target)
        rows.append({'support_size':len(slots),'generated_slots_at_half':assignment.tolist(),
            'assignment_source':'live_reward_last_control_forward' if assignments is not None else 'observational_actual_half_geometry',
            **{'half_'+k:v for k,v in h.items()},**{'terminal_tracked_'+k:v for k,v in t.items()},
            **{'terminal_rematched_'+k:v for k,v in rematched.items()},
            'suffix_tracking_residual_A':float(np.sqrt(np.square((y[assignment]-x[assignment])-(final_target-target)).sum(-1).mean())),
            'suffix_tracked_displacement_A':float(np.sqrt(np.square(y[assignment]-x[assignment]).sum(-1).mean()))})
    return rows
