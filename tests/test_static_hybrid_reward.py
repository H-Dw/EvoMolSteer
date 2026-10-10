import copy
import numpy as np
import pytest
import torch
from evomolsteer.generation.static_hybrid_reward import StaticHybridReward
from evomolsteer.generation.structure_field_reward import StructureFieldReward
from evomolsteer.generation.flowcompat_controller import measured_pullback


def setup(rho=.5):
    p = {'window': [.2, .6], 'original_pose_prior_mass': rho, 'contact_weight': 0,
         'repulsion_weight': 0, 'mixture_temperature': .5}
    r = {'schema_version': 'static-hybrid-coordinate-library-1.0', 'window': [.2, .6],
         'bound_ligand_A': [[3., 0, 0], [3., 1, 0]],
         'teacher_bank_A': [[[3.3, 0, 0], [3.3, 1, 0]], [[3., .2, 0], [3., 1.2, 0]]],
         'protein_points_A': [[0., 0, 0], [6., 2, 0]]}
    x = torch.tensor([[[3.1, .1, .1], [3.2, 1.1, .1]]], dtype=torch.float64)
    return p, r, x


def test_original_only_matches_existing_scalar_and_gradient():
    p, r, x = setup(1); x.requires_grad_(True); mask = torch.ones((1, 2), dtype=torch.bool)
    original = {**r, 'schema_version': 'structure-field-reference-1.0', 'steer_sources': []}
    old_p = {**p, 'ligand_anchor_weight': 1}
    left = StaticHybridReward(p, r)(x, None, mask, .3, x.detach())[0]
    right = StructureFieldReward(old_p, original)(x, None, mask, .3, x.detach())[0]
    gl, = torch.autograd.grad(left.sum(), x, retain_graph=True)
    gr, = torch.autograd.grad(right.sum(), x)
    assert torch.allclose(left, right, atol=1e-12)
    assert torch.allclose(gl, gr, atol=1e-12)


def test_conditional_finite_difference_and_masked_padding():
    p, r, x = setup(); x = torch.cat([x, x.new_full((1, 1, 3), 999)], 1).requires_grad_(True)
    mask = torch.tensor([[True, True, False]]); reward = StaticHybridReward(p, r); anchor = x.detach()
    v, d = reward(x, None, mask, .3, anchor)
    g, = torch.autograd.grad(v.sum(), x); direction = g / g.norm(); e = 1e-5
    numeric = (reward(x+e*direction, None, mask, .3, anchor)[0] -
               reward(x-e*direction, None, mask, .3, anchor)[0]) / (2*e)
    assert float(numeric.detach()) == pytest.approx(float((g*direction).sum()), rel=1e-6)
    assert not g[0, 2].any() and torch.isfinite(g).all()
    assert 0 <= float(d['hybrid_endpoint_cancellation_ratio'][0]) <= 1+1e-12


def test_rigid_frame_and_atom_order_covariance():
    p, r, x = setup(); mask = torch.ones((1, 2), dtype=torch.bool)
    old = StaticHybridReward(p, r)(x, None, mask, .3, x)[0]
    rotation = x.new_tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = x.new_tensor([7., 3., -2.]); new = copy.deepcopy(r)
    for key in ['bound_ligand_A', 'teacher_bank_A', 'protein_points_A']:
        new[key] = (torch.tensor(r[key], dtype=x.dtype) @ rotation + shift).tolist()
    y = (x @ rotation + shift).flip(1)
    assert torch.allclose(old, StaticHybridReward(p, new)(y, None, mask, .3, y)[0], atol=1e-12)


def test_dynamic_window_and_late_contact_schedule():
    p, r, x = setup(); p.update(contact_weight=.1, contact_phase_power=2)
    reward = StaticHybridReward(p, r); mask = torch.ones((1, 2), dtype=torch.bool)
    assert reward.active(.2, .21) and not reward.active(.19, .2) and not reward.active(.6, .61)
    _, start = reward(x, None, mask, .2, x); _, mid = reward(x, None, mask, .4, x)
    value, outside = reward(x, None, mask, .7, x)
    assert start['hybrid_contact_weight'][0] == 0
    assert float(mid['hybrid_contact_weight'][0]) == pytest.approx(.025)
    assert value[0] == 0 and outside['dose_gate'][0] == 0


def test_pullback_uses_shared_forward_and_detaches_affinity():
    p, r, x = setup(); calls = []
    def predict(z):
        calls.append(True)
        return {'coords': z*.8 + .2*x, 'atomics': torch.zeros((1, 2, 3), dtype=z.dtype),
                'affinity': {'target': z.sum((1, 2))}}, None
    mask = torch.ones((1, 2), dtype=torch.bool)
    pred, _, g, _, _, _, _ = measured_pullback(predict, StaticHybridReward(p, r),
                                               x, mask, 1., x.new_zeros((1, 3)), .3)
    assert len(calls) == 1 and torch.isfinite(g).all() and g.norm() > 0
    assert not pred['affinity']['target'].requires_grad


def test_invalid_parameters_and_mismatch_rejected():
    p, r, _ = setup(); p['original_pose_prior_mass'] = 1.1
    with pytest.raises(ValueError, match='parameters'): StaticHybridReward(p, r)
    p['original_pose_prior_mass'] = .5; r['window'] = [.1, .5]
    with pytest.raises(ValueError, match='Window'): StaticHybridReward(p, r)


def test_contact_component_diagnostics_match_autograd_without_changing_reward():
    p, r, x = setup(); x.requires_grad_(True); mask = torch.ones((1, 2), dtype=torch.bool)
    geometry, _ = StaticHybridReward(p, r)(x, None, mask, .4, x.detach())
    gg, = torch.autograd.grad(geometry.sum(), x, retain_graph=True)
    p.update(contact_weight=.2, contact_phase_power=2)
    total, detail = StaticHybridReward(p, r)(x, None, mask, .4, x.detach())
    gt, = torch.autograd.grad(total.sum(), x)
    contact = gt - gg
    share = contact.norm() / (gg.norm() + contact.norm())
    cosine = (gg * contact).sum() / (gg.norm() * contact.norm())
    assert float(detail['hybrid_endpoint_contact_gradient_share'][0]) == pytest.approx(float(share), rel=1e-10)
    assert float(detail['hybrid_endpoint_geometry_contact_cosine'][0]) == pytest.approx(float(cosine), rel=1e-10)
