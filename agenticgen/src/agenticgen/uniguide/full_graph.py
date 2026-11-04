"""Complete chemistry correspondence anchored by the independent geometric core."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .absolute_coordinate import aligned


def assignment(xyz, valid, condition, core_slots):
    ids=[a['id'] for a in condition['atoms']]
    core=[ids.index(i) for i in condition['controlled_atom_ids']]
    other=[i for i in range(len(ids)) if i not in core]
    points=aligned(xyz.detach().cpu().double(),condition).numpy()
    target=np.asarray(condition['reference_coordinates_angstrom'],dtype=np.float64)
    result=[]
    for x,mask,slots in zip(points,valid.detach().cpu().numpy(),core_slots.detach().cpu().numpy()):
        available=np.flatnonzero(mask)
        if len(available)!=len(ids) or len(set(slots))!=len(core) or not set(slots)<=set(available):
            raise ValueError('Full graph requires exactly one valid slot per teacher heavy atom')
        mapping=np.empty(len(ids),dtype=np.int64);mapping[core]=slots
        remaining=np.asarray([i for i in available if i not in slots],dtype=np.int64)
        if other:
            cost=np.sum((target[other,None]-x[remaining][None])**2,axis=-1)
            if not np.isfinite(cost).all():raise FloatingPointError('Nonfinite full graph correspondence')
            a,b=linear_sum_assignment(cost);mapping[np.asarray(other)[a]]=remaining[b]
        result.append(mapping.tolist())
    return torch.tensor(result,device=xyz.device,dtype=torch.long)
