"""Scalar, gradient and ancestry contract tests; no FLOWR forward is mocked as data."""
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from evomolsteer.generation.branch_mixture_reward import BranchMixtureReward
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward


def fixture(dtype=torch.float64):
    root = Path(__file__).resolve().parents[1]
    program = json.loads((root / 'configs/experiments/skill_ablation_v1/incumbent.json').read_text())
    reference = json.loads(gzip.decompress((root / 'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz').read_bytes()))
    reference['times'] = reference['times'][:3]
    reference['frames'] = reference['frames'][:3]
    for frame in reference['frames']:
        frame['teacher_endpoint_A'] = frame['teacher_endpoint_A'][:5]
        frame['teacher_scores'] = frame['teacher_scores'][:5]
        frame['teacher_batches'] = frame['teacher_batches'][:5]
        shape = np.asarray(frame['teacher_endpoint_A']).shape
        direction = np.sin(np.arange(np.prod(shape)).reshape(shape) * .31 + .7)
        direction /= np.sqrt((direction ** 2).sum(-1).mean(-1))[:, None, None]
        frame['teacher_contrast_direction_unit'] = direction.tolist()
        frame['teacher_contrast_confidence'] = [.2] * shape[0]
        frame['teacher_contrast_atom_weight'] = np.tile(np.linspace(.5, 1.5, shape[1]), (shape[0], 1)).tolist()
        frame['teacher_contrast_provenance'] = [
            {'lag2_observed': True, 'common_grandparent_slot': 3,
             'immediate_parent_slot': 5, 'lower_immediate_parent_slots': [6, 7],
             'distinct_observed_mutations': 3, 'raw_direction_RMS_A': .04}
            for _ in range(shape[0])]
    new = copy.deepcopy(program)
    new['reward_view'] = 'endpoint_branch_mixture'
    new['branch_mixture'] = {'virtual_mass': .25}
    coords = np.asarray(reference['frames'][2]['teacher_endpoint_A'])[:2] + .1
    x = torch.tensor(coords, dtype=dtype).requires_grad_(True)
    mask = torch.ones(x.shape[:2], dtype=torch.bool)
    atoms = torch.zeros_like(mask, dtype=torch.long)
    return program, new, reference, x, atoms, mask, reference['times'][2]


def evaluate(reward, x, atoms, mask, time, anchor=None):
    value, detail = reward(x, atoms, mask, time, x.detach() if anchor is None else anchor)
    gradient, = torch.autograd.grad(value.sum(), x)
    return value, gradient, detail


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('spec', [{}, {'virtual_mass': 0.}, {'virtual_mass': .5, 'direction_sign': 0.}])
def test_null_and_zero_sign_use_exact_baseline_arithmetic(dtype, spec):
    p, n, r, x, atoms, mask, time = fixture(dtype)
    n['branch_mixture'] = spec
    a, ga, _ = evaluate(EndpointGeometryReward(p, r), x, atoms, mask, time)
    b, gb, detail = evaluate(BranchMixtureReward(n, r), x, atoms, mask, time)
    assert torch.equal(a, b) and torch.equal(ga, gb)
    assert detail['branch_virtual_mass_mean'].count_nonzero() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reason', ['confidence', 'ancestor', 'lower', 'raw'])
def test_zero_eligibility_exact_baseline_even_with_regional_weights(dtype, reason):
    p, n, r, x, atoms, mask, time = fixture(dtype)
    n['branch_mixture']['region_weight_mix'] = 1.
    for frame in r['frames']:
        if reason == 'confidence':
            frame['teacher_contrast_confidence'] = [0.] * len(frame['teacher_scores'])
        for item in frame['teacher_contrast_provenance']:
            if reason == 'ancestor': item['lag2_observed'] = False
            if reason == 'lower': item['lower_immediate_parent_slots'] = []
            if reason == 'raw': item['raw_direction_RMS_A'] = 0.
    a, ga, _ = evaluate(EndpointGeometryReward(p, r), x, atoms, mask, time)
    b, gb, detail = evaluate(BranchMixtureReward(n, r), x, atoms, mask, time)
    assert torch.equal(a, b) and torch.equal(ga, gb)
    assert detail['branch_region_weight_rms_difference'].count_nonzero() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_first_two_nodes_exact_baseline_despite_invalid_support_claim(dtype):
    p, n, r, x, atoms, mask, _ = fixture(dtype)
    for time in r['times'][:2]:
        a, ga, _ = evaluate(EndpointGeometryReward(p, r), x, atoms, mask, time)
        b, gb, detail = evaluate(BranchMixtureReward(n, r), x, atoms, mask, time)
        assert torch.equal(a, b) and torch.equal(ga, gb)
        assert detail['branch_eligible_teacher_count'].count_nonzero() == 0


def test_nonzero_joint_branch_gradient_matches_finite_difference():
    p, n, r, x, atoms, mask, time = fixture()
    reward = BranchMixtureReward(n, r)
    anchor = x.detach().clone()
    value, g, detail = evaluate(reward, x, atoms, mask, time, anchor)
    _, old_g, _ = evaluate(EndpointGeometryReward(p, r), x, atoms, mask, time, anchor)
    assert torch.isfinite(value).all() and (g - old_g).norm() > 1e-5
    direction = g / g.norm()
    epsilon = 1e-5
    hi, _ = reward(x.detach() + epsilon * direction, atoms, mask, time, anchor)
    lo, _ = reward(x.detach() - epsilon * direction, atoms, mask, time, anchor)
    assert torch.allclose((hi.sum() - lo.sum()) / (2 * epsilon), (g * direction).sum(), rtol=1e-6, atol=1e-9)
    assert torch.allclose(detail['branch_virtual_prior_mass'], x.new_full((2,), .25))


def test_positive_and_negative_have_same_support_and_physical_dose():
    _, n, r, x, atoms, mask, time = fixture()
    r_original = copy.deepcopy(r)
    plus = BranchMixtureReward(n, r)
    minus_program = copy.deepcopy(n)
    minus_program['branch_mixture']['direction_sign'] = -1.
    minus = BranchMixtureReward(minus_program, r)
    _, gp, dp = evaluate(plus, x, atoms, mask, time)
    _, gm, dm = evaluate(minus, x, atoms, mask, time)
    for key in ['branch_virtual_teacher_shift_rms_A', 'branch_virtual_mass_mean', 'branch_virtual_prior_mass', 'branch_eligible_teacher_count']:
        assert torch.equal(dp[key], dm[key])
    assert torch.allclose(dp['branch_virtual_teacher_shift_rms_A'], x.new_full((2,), .04))
    assert not torch.equal(gp, gm)
    assert r == r_original
    assert plus.reference['frames'][2]['teacher_endpoint_A'] == r['frames'][2]['teacher_endpoint_A']
    assert plus.reference['frames'][2]['teacher_scores'] == r['frames'][2]['teacher_scores']


def test_shift_uses_bounded_natural_amplitude_not_confidence_scaling():
    _, n, r, x, atoms, mask, time = fixture()
    frame = r['frames'][2]
    frame['teacher_contrast_confidence'] = [1e-6] * len(frame['teacher_scores'])
    for item in frame['teacher_contrast_provenance']: item['raw_direction_RMS_A'] = .4
    _, _, detail = evaluate(BranchMixtureReward(n, r), x, atoms, mask, time)
    assert torch.allclose(detail['branch_virtual_teacher_shift_rms_A'], x.new_full((2,), .2))
    assert torch.allclose(detail['branch_virtual_mass_mean'], x.new_full((2,), .25))


def test_regional_weights_are_confidence_conditional_and_zero_support_is_identity():
    _, n, r, x, atoms, mask, time = fixture()
    n['branch_mixture'] = {'virtual_mass': 0., 'region_weight_mix': 1.}
    frame = r['frames'][2]
    frame['teacher_contrast_confidence'] = [.2, 0., .2, 0., .2]
    reward = BranchMixtureReward(n, r)
    _, _, c, w, eligible, _ = reward._fields(reward.reference['frames'][2], 2)
    effective = 1. + n['branch_mixture']['region_weight_mix'] * (c * eligible)[:, None] * (w - 1.)
    assert np.array_equal(effective[[1, 3]], np.ones_like(effective[[1, 3]]))
    assert np.allclose(effective[[0, 2, 4]], 1 + .2 * (w[[0, 2, 4]] - 1))
    _, _, detail = evaluate(reward, x, atoms, mask, time)
    assert detail['branch_region_weight_rms_difference'].max() > 0
    assert detail['branch_virtual_mass_mean'].count_nonzero() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_no_selected_eligible_teacher_preserves_baseline_exactly(dtype):
    p, n, r, x, atoms, mask, time = fixture(dtype)
    n['branch_mixture']['region_weight_mix'] = 1.
    for key in ['teacher_endpoint_A', 'teacher_scores', 'teacher_batches', 'teacher_contrast_direction_unit',
                'teacher_contrast_confidence', 'teacher_contrast_atom_weight', 'teacher_contrast_provenance']:
        frame = r['frames'][2]
        frame[key].append(copy.deepcopy(frame[key][-1]))
    frame['teacher_endpoint_A'][-1] = (np.asarray(frame['teacher_endpoint_A'][-1]) + 100.).tolist()
    frame['teacher_contrast_confidence'] = [0.] * 5 + [.2]
    a, ga, _ = evaluate(EndpointGeometryReward(p, r), x, atoms, mask, time)
    b, gb, detail = evaluate(BranchMixtureReward(n, r), x, atoms, mask, time)
    assert torch.equal(a, b) and torch.equal(ga, gb)
    assert detail['branch_virtual_mass_mean'].count_nonzero() == 0


@pytest.mark.parametrize('spec', [{'virtual_mass': .50001}, {'direction_sign': -1.001}, {'region_weight_mix': 1.01}, {'virtual_mass': float('nan')}, {'confidence_power': .5}])
def test_unregistered_or_unbounded_configuration_rejected(spec):
    _, n, r, *_ = fixture()
    n['branch_mixture'] = spec
    with pytest.raises(ValueError): BranchMixtureReward(n, r)
