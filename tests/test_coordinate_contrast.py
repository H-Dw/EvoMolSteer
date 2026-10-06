import copy
import numpy as np
import torch
import pytest
from evomolsteer.generation.coordinate_contrast import (
    gaussian_log_components,support_gate,CoordinateSelectionContrastReward)
from evomolsteer.generation.coordinate_reference import mode_record


def fixture(identical=False):
    mode={'center_A':[0.,0.,0.,1.], 'covariance_A2':np.eye(4).tolist(),
        'background_center_A':[.5,0.,0.,1.], 'background_covariance_A2':(np.eye(4)*1.2).tolist(),
        'paired_gaussian_KL_nats':.15,'background_support_q90':3.,'background_support_q98':5.}
    if identical:
        mode.update(background_center_A=mode['center_A'],background_covariance_A2=mode['covariance_A2'],paired_gaussian_KL_nats=0.)
    ref={'schema_version':'current-coordinate-mixture-1.0','window':[.2,.6],'times':[.2,.6],
        'regions':{'patch':{'points_A':[[0,0,0],[1,0,0]]}},'channel':'all','spatial_width_A':4.,
        'spatial_anchor':'endpoint','frames':[{'modes':[copy.deepcopy(mode)]} for _ in range(2)]}
    p={'window':[.2,.6],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.,
       'reward_view':'selection_contrast','contrast_bound_nats':1.}
    return CoordinateSelectionContrastReward(p,ref)


def test_normalized_gaussian_matches_distribution_with_unequal_volumes():
    z=torch.tensor([[1.,2.],[.3,.4]],dtype=torch.float64)
    center=[[0.,0.],[1.,1.]];cov=[[[1.,.2],[.2,2.]],[[.4,0.],[0.,3.]]]
    actual,_=gaussian_log_components(z,center,cov)
    expected=torch.stack([torch.distributions.MultivariateNormal(z.new_tensor(m),z.new_tensor(c)).log_prob(z)-np.log(2)
                          for m,c in zip(center,cov)],1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)


def test_contrast_fd_conditional_anchor_and_support_rejection():
    r=fixture();gen=torch.Generator().manual_seed(42)
    x=torch.randn(3,8,3,generator=gen,dtype=torch.float64,requires_grad=True)
    anchor=torch.randn(3,8,3,generator=gen,dtype=torch.float64,requires_grad=True)
    mask=torch.ones((3,8),dtype=torch.bool);a=torch.full((3,8),3)
    value,d=r(x,a,mask,.2,anchor);g,ga=torch.autograd.grad(value.sum(),(x,anchor),allow_unused=True)
    assert ga is None and value.abs().max()<1. and d['dose_gate'].min()>0
    direction=g/g.norm();eps=1e-5
    numerical=(r(x+eps*direction,a,mask,.2,anchor)[0].sum()-r(x-eps*direction,a,mask,.2,anchor)[0].sum())/(2*eps)
    torch.testing.assert_close(numerical,(g*direction).sum(),rtol=1e-7,atol=1e-9)
    assert r.active(.2,.21) and not r.active(.6,.61)
    _,far=r(x+100,a,mask,.2,anchor)
    assert far['dose_gate'].eq(0).all()


def test_identical_density_does_not_create_a_normalized_direction():
    r=fixture(True);x=torch.tensor([[[-1.,0,0],[1.,0,0]]],dtype=torch.float64,requires_grad=True)
    mask=torch.ones((1,2),dtype=torch.bool);a=torch.full((1,2),3)
    value,d=r(x,a,mask,.2,x.detach());g,=torch.autograd.grad(value.sum(),x)
    assert value.eq(0).all() and g.eq(0).all() and d['dose_gate'].eq(0).all()


def test_empirical_support_not_chi_squared_or_synthetic_coverage():
    v=np.arange(40,dtype=float).reshape(10,4)/20.;w=np.ones(10)/10
    m=mode_record(v,w,2,1.,.15)
    assert m['paired_gaussian_KL_nats']==pytest.approx(0,abs=1e-12)
    residual=v-v.mean(0);cov=np.asarray(m['background_covariance_A2'])
    q=np.einsum('bi,ib->b',residual,np.linalg.solve(cov,residual.T))/4
    assert m['background_support_q90']==pytest.approx(np.quantile(q,.90))
    gate=support_gate(torch.tensor([[1.,2.]],dtype=torch.float64),[1.,2.],[1.,2.],torch.ones(1,2)/2)
    assert gate.eq(0).all()  # degeneracy is native fallback, not made-up support


def test_double_density_preserves_float_native_update_and_zero_equivalence():
    from evomolsteer.generation.local_reward import bounded_local_step
    from evomolsteer.generation.multistage_reward import preserve_native_geometry
    r=fixture();x=torch.tensor([[[-1.,0,0],[1.,0,0]]],requires_grad=True)
    mask=torch.ones((1,2),dtype=torch.bool);a=torch.full((1,2),3)
    value,d=r(x,a,mask,.2,x.detach());g,=torch.autograd.grad(value.sum(),x)
    assert d['dose_gate'].dtype==torch.float64 and g.dtype==torch.float32
    constraints={'backtrack_attempts':7,'severe_receptor_clash_A':.8,'max_pair_distance_change_A':.06}
    pocket=torch.tensor([[[10.,10,10.]]]);pm=torch.ones((1,1),dtype=torch.bool)
    for eta in (0.,.05):
        step,_=bounded_local_step(g,torch.ones_like(x)*.01,mask,d['dose_gate'].to(g),eta,1.,.025,torch.ones(1))
        actual,_=preserve_native_geometry(x.detach(),step,mask,pocket,pm,1.,constraints)
        assert actual.dtype==x.dtype
        if eta==0:assert torch.equal(x.detach()+actual,x.detach())
