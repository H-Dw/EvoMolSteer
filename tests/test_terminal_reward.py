import copy
import numpy as np
import pytest
import torch
from evomolsteer.continuous.affinity_geometry import names
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward


def fixture():
    points = np.zeros((1, 3))
    target = np.array([[-1., 0., 0.], [1., 0., 0.]])
    ref = dict(schema_version='affinity-endpoint-library-1.0',
        reference_variant='terminal-descendant-endpoint-library-1.0', allowed_reward_views=['endpoint_pointcloud'],
        window=[.1, .4], times=[.1], features=names(points), feature_scale=[1.]*len(names(points)),
        landmarks_A=points.tolist(), origin_A=[0., 0., 0.],
        frames=[dict(teacher_endpoint_A=[target.tolist(), (target+.2).tolist(), (target-.3).tolist()],
            teacher_scores=[8., 8., 8.], teacher_batches=[0, 0, 1], teacher_base_log_weight=[-np.log(2), -np.log(2), 0.])])
    p = dict(window=[.1, .4], reward_view='endpoint_pointcloud', derivative_path='flowr_endpoint_vjp',
        teacher_neighbors=3, teacher_score_beta=0., teacher_endpoint_temperature_A2=4., mixture_temperature=.5)
    return ref, p, target


def test_terminal_coordinate_gradient_with_equal_batch_base_prior():
    ref, p, target = fixture()
    x = torch.tensor(target*.6)[None].requires_grad_(True)
    anchor = x.detach()
    value, _ = EndpointGeometryReward(p, ref)(x, torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool), .1, anchor)
    teachers = x.new_tensor(ref['frames'][0]['teacher_endpoint_A'])
    anchor_cost = (anchor[:, None]-teachers[None]).square().sum(-1).mean(2)
    logits = -anchor_cost/4+x.new_tensor(ref['frames'][0]['teacher_base_log_weight'])
    q = (x[:, None]-teachers[None]).square().sum(-1).mean(2)
    expected = .5*torch.logsumexp(logits.log_softmax(1)-(torch.sqrt(1+q)-1)/.5, 1)
    torch.testing.assert_close(value, expected)
    grad, = torch.autograd.grad(value.sum(), x)
    for eps in (1e-5,):
        direction = torch.tensor([[[.3, -.2, .1], [-.2, .1, -.3]]], dtype=x.dtype)
        reward = EndpointGeometryReward(p, ref)
        def f(z): return reward(z, torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool), .1, anchor)[0].sum()
        finite = (f(x.detach()+eps*direction)-f(x.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close((grad*direction).sum(), finite, rtol=1e-7, atol=1e-8)


def test_terminal_reference_rejects_instantaneous_strata_or_wrong_base_mass():
    ref, p, _ = fixture()
    with pytest.raises(ValueError): EndpointGeometryReward(dict(p, reward_view='endpoint_direction'), ref)
    bad = copy.deepcopy(ref); bad['frames'][0]['teacher_base_log_weight'] = [0., 0., 0.]
    with pytest.raises(ValueError): EndpointGeometryReward(p, bad)
    bad = copy.deepcopy(ref); bad['frames'][0].pop('teacher_base_log_weight')
    with pytest.raises(ValueError): EndpointGeometryReward(p, bad)
