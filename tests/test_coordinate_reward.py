import numpy as np
import torch
import pytest
from evomolsteer.continuous.coordinate_features import regional_moments
from evomolsteer.generation.coordinate_reward import CoordinateMixtureReward


def reference():
    return {'schema_version':'current-coordinate-mixture-1.0','window':[.1,.4],'times':[.1,.2,.3,.4],
        'regions':{'patch':{'points_A':[[0,0,0],[1,0,0]]}},'channel':'all','spatial_width_A':4.,
        'frames':[{'modes':[{'center_A':[0,0,0,1.5],'covariance_A2':np.eye(4).tolist()}]}]*4}


def test_numpy_torch_parity_and_directional_derivative():
    ref=reference();program={'window':[.1,.4],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.}
    reward=CoordinateMixtureReward(program,ref)
    x=torch.randn(12,8,3,dtype=torch.float64,requires_grad=True);a=torch.full((12,8),3);m=torch.ones((12,8),dtype=torch.bool)
    z,valid,_=reward.observables(x,a,m)
    native=regional_moments(x.detach().numpy(),a.numpy(),m.numpy(),ref['regions']['patch']['points_A'],None,4.)
    np.testing.assert_allclose(native,z.detach().numpy(),atol=1e-12)
    value,_=reward(x,a,m,.2);g,=torch.autograd.grad(value.sum(),x);d=g/g.norm();eps=1e-5
    numerical=(reward(x+eps*d,a,m,.2)[0].sum()-reward(x-eps*d,a,m,.2)[0].sum())/(2*eps)
    assert torch.allclose(numerical,(g*d).sum(),rtol=1e-7)
    assert reward.active(.1,.2) and not reward.active(.0,.1) and not reward.active(.4,.5)
    with pytest.raises(ValueError):reward(x,a,m,.21)


def test_nos_missing_zero_gradient_and_nonmutating_mask():
    ref=reference();ref['channel']='NOS';ref['atom_vocabulary']={'N':4,'O':5,'S':9}
    reward=CoordinateMixtureReward({'window':[.1,.4],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.},ref)
    x=torch.randn(2,4,3,dtype=torch.float64,requires_grad=True);a=torch.full((2,4),3);m=torch.ones((2,4),dtype=torch.bool)
    value,detail=reward(x,a,m,.2);assert not detail['available'].any();assert value.eq(0).all()
    g,=torch.autograd.grad(value.sum(),x);assert g.eq(0).all()
    mask=m.numpy().copy();z=regional_moments(x.detach().numpy(),a.numpy(),mask,[[0,0,0]],[4,5,9])
    assert np.isnan(z).all() and mask.all()


def test_upper_spread_has_no_pressure_below_bound():
    ref=reference()
    for f in ref['frames']:f.update(upper_spread_A=[2.],spread_scale_A=[.5])
    program={'window':[.1,.4],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.,'reward_view':'spread_upper'}
    reward=CoordinateMixtureReward(program,ref)
    x=torch.tensor([[[-.1,0,0],[.1,0,0]]],dtype=torch.float64,requires_grad=True)
    mask=torch.ones((1,2),dtype=torch.bool);atoms=torch.full((1,2),3)
    v,_=reward(x,atoms,mask,.2);g,=torch.autograd.grad(v.sum(),x)
    assert v.eq(0).all() and g.eq(0).all()
    far=(x.detach()*30).requires_grad_(True);v,_=reward(far,atoms,mask,.2);g,=torch.autograd.grad(v.sum(),far)
    assert v.lt(0).all() and (g*far).sum()<0
