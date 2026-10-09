import copy
import numpy as np
import pytest
import torch
from evomolsteer.generation.structure_field_reward import StructureFieldReward

def example():
    reference = {'schema_version': 'structure-field-reference-1.0', 'window': [0.1, .4],
        'steer_sources': [], 'protein_points_A': [[0, 0, 0], [5, 2, 0]],
        'bound_ligand_A': [[3, 0, 0], [3, 1, 0], [3, 2, 0]]}
    program = {'window': [.1, .4], 'ligand_anchor_weight': 1, 'contact_weight': .25, 'repulsion_weight': 2}
    return program, reference

def test_conditional_gradient_and_padding():
    p, r = example(); reward = StructureFieldReward(p, r)
    x = torch.tensor([[[2.8, .1, .2], [3.3, 1.1, .2], [999., 999., 999.]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    anchor = x.detach()
    value, _ = reward(x, None, mask, .2, anchor)
    gradient, = torch.autograd.grad(value.sum(), x)
    assert torch.isfinite(gradient).all() and gradient[0, 2].norm() == 0
    direction = gradient / gradient.norm(); eps = 1e-5
    numerical = (reward(x+eps*direction, None, mask, .2, anchor)[0] - reward(x-eps*direction, None, mask, .2, anchor)[0])/(2*eps)
    assert float(numerical.detach()) == pytest.approx(float((gradient*direction).sum()), rel=1e-5)

def test_frame_covariance_and_permutation():
    p, r = example(); x = torch.tensor([[[2.8, .1, .2], [3.3, 1.1, .2]]], dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    old = StructureFieldReward(p, r)(x, None, mask, .2, x)[0]
    transform = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=x.dtype)
    shift = torch.tensor([7., 3., -2.], dtype=x.dtype)
    modified = copy.deepcopy(r)
    for key in ['protein_points_A', 'bound_ligand_A']:
        modified[key] = (torch.tensor(r[key], dtype=x.dtype) @ transform + shift).tolist()
    new_x = (x @ transform + shift).flip(1)
    new = StructureFieldReward(p, modified)(new_x, None, mask, .2, new_x)[0]
    assert torch.allclose(old, new, atol=1e-12)

def test_no_hidden_teacher_and_dynamic_window():
    p, r = example(); reward = StructureFieldReward(p, r)
    assert reward.active(.1, .11) and not reward.active(.09, .1) and not reward.active(.4, .41)
    r['steer_sources'] = ['secret']
    with pytest.raises(ValueError, match='contamination'): StructureFieldReward(p, r)
