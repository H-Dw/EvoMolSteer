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


def test_uniform_global_control_is_radius_gyration():
    ref=reference();ref['spatial_weighting']='uniform_global_control'
    r=CoordinateMixtureReward({'window':[.1,.4],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.},ref)
    x=torch.randn(3,5,3,dtype=torch.float64);m=torch.ones((3,5),dtype=torch.bool);a=torch.full((3,5),3)
    z,_,_=r.observables(x,a,m)
    expected=(x-x.mean(1)[:,None]).square().sum(-1).mean(1).sqrt()
    torch.testing.assert_close(z[:,3],expected)


def test_pose_projection_retains_ascent_without_rigid_motion():
    from evomolsteer.generation.coordinate_reward import remove_rigid_pose_gradient
    x=torch.randn(4,8,3,dtype=torch.float64);g=torch.randn_like(x);m=torch.ones((4,8),dtype=torch.bool)
    p=remove_rigid_pose_gradient(g,x,m)
    torch.testing.assert_close(p.sum(1),torch.zeros(4,3,dtype=x.dtype),atol=1e-12,rtol=0.)
    torch.testing.assert_close(torch.cross(x-x.mean(1)[:,None],p,dim=-1).sum(1),torch.zeros(4,3,dtype=x.dtype),atol=1e-12,rtol=0.)
    assert ((g*p).sum((1,2))>=0).all()
    torch.testing.assert_close((g*p).sum((1,2)),p.square().sum((1,2)),atol=1e-12,rtol=1e-12)


def test_endpoint_anchor_conditional_gradient_and_slot_permutation():
    ref=reference();ref['spatial_anchor']='endpoint';ref['control_representation']='proposal'
    r=CoordinateMixtureReward({'window':[.1,.4],'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.},ref)
    generator=torch.Generator().manual_seed(42)
    x=torch.randn(3,7,3,generator=generator,dtype=torch.float64,requires_grad=True)
    anchor=torch.randn(3,7,3,generator=generator,dtype=torch.float64,requires_grad=True)*8
    m=torch.ones((3,7),dtype=torch.bool);a=torch.full((3,7),3)
    value,detail=r(x,a,m,.2,anchor)
    expected=regional_moments(x.detach().numpy(),a.numpy(),m.numpy(),ref['regions']['patch']['points_A'],None,4.,anchor.detach().numpy())
    np.testing.assert_allclose(expected,detail['observables_A'].detach().numpy(),atol=1e-12)
    g,ga=torch.autograd.grad(value.sum(),(x,anchor),allow_unused=True);assert ga is None
    d=g/g.norm();eps=1e-5
    numerical=(r(x+eps*d,a,m,.2,anchor)[0].sum()-r(x-eps*d,a,m,.2,anchor)[0].sum())/(2*eps)
    torch.testing.assert_close(numerical,(g*d).sum(),rtol=1e-7,atol=1e-10)
    order=[6,0,4,1,5,3,2]
    torch.testing.assert_close(value,r(x[:,order],a[:,order],m[:,order],.2,anchor[:,order])[0])
    with pytest.raises(ValueError):r(x,a,m,.2)


def test_selected_background_moment_audit():
    from evomolsteer.generation.coordinate_reference import mode_record
    rng=np.random.default_rng(42);v=rng.normal(size=(50,8));w=np.full(50,.02)
    record=mode_record(v,w,0,1.,.15)
    assert abs(record['selected_probability_ESS']-50)<1e-10
    assert abs(record['selected_KL_to_uniform'])<1e-12
    assert record['dimension_normalized_mean_contrast']<1e-12
    np.testing.assert_allclose(record['center_A'],record['background_center_A'],atol=1e-12)
    assert np.linalg.eigvalsh(record['background_covariance_A2']).min()>0
    w=np.arange(1,51,dtype=float);w/=w.sum();record=mode_record(v,w,0,1.,.15)
    assert record['dimension_normalized_mean_contrast']>0
    assert record['selected_probability_ESS']<50 and record['selected_KL_to_uniform']>0


def test_flow_dose_avoids_startup_contraction_and_is_frame_equivariant():
    from evomolsteer.generation.coordinate_reward import predictive_flow_increment
    from evomolsteer.generation.local_reward import bounded_local_step
    x=torch.tensor([[[2.,0,0],[-2.,0,0]]]);end=x*.5;g=torch.ones_like(x)
    flow=predictive_flow_increment(x,end,0.,.01)
    # At g_t=100 and dt=.01, the native SDE score drift nearly cancels x.
    native=flow-x;mask=torch.ones((1,2),dtype=torch.bool);gate=torch.ones(1)
    _,a=bounded_local_step(g,native,mask,gate,.05,1.,100.,torch.ones(1)*100)
    _,b=bounded_local_step(g,flow,mask,gate,.05,1.,100.,torch.ones(1)*100)
    assert a['requested_rms_A'].item()>100*b['requested_rms_A'].item()
    rot=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    torch.testing.assert_close(predictive_flow_increment(x@rot+5,end@rot+5,0.,.01),flow@rot)
    with pytest.raises(ValueError):predictive_flow_increment(x,end,.2,.01,True)


def test_first_update_cap_is_per_molecule_masked_and_disabled_exactly():
    from evomolsteer.generation.coordinate_reward import calibration_increment
    native=torch.tensor([[[10.,0,0],[999.,999.,999.]],[[1.,0,0],[999.,999.,999.]]])
    flow=torch.tensor([[[1.,0,0],[0.,0,0]],[[2.,0,0],[0.,0,0]]])
    mask=torch.tensor([[True,False],[True,False]])
    rng=torch.get_rng_state().clone()
    assert calibration_increment(native,flow,mask) is native
    assert calibration_increment(native,flow,mask,initial_update_dose='cap_to_flow',first_controlled=False) is native
    capped=calibration_increment(native,flow,mask,initial_update_dose='cap_to_flow',first_controlled=True)
    torch.testing.assert_close(capped[:,0,0],torch.tensor([1.,1.]))
    assert torch.equal(rng,torch.get_rng_state()) and native[0,0,0]==10
    with pytest.raises(ValueError):calibration_increment(native,flow,mask,'predictive_flow','cap_to_flow',True)
    with pytest.raises(ValueError):calibration_increment(native,None,mask,initial_update_dose='cap_to_flow',first_controlled=True)
    zero=calibration_increment(native,torch.zeros_like(flow),mask,initial_update_dose='cap_to_flow',first_controlled=True)
    assert zero.eq(0).all()


def test_pair_guard_can_limit_internal_deformation_below_atom_cap():
    from evomolsteer.generation.multistage_reward import preserve_native_geometry
    x=torch.tensor([[[0.,0,0],[1.,0,0]]],dtype=torch.float64)
    delta=torch.tensor([[[-.02,0,0],[.02,0,0]]],dtype=torch.float64)
    mask=torch.ones((1,2),dtype=torch.bool);pocket=torch.tensor([[[10.,10,10]]],dtype=torch.float64)
    pm=torch.ones((1,1),dtype=torch.bool)
    constraints={'backtrack_attempts':7,'severe_receptor_clash_A':.8,'max_pair_distance_change_A':.06}
    old,a=preserve_native_geometry(x,delta,mask,pocket,pm,1.,constraints)
    assert a['backtrack_factor'].item()==1. and delta.norm(dim=-1).max()<.025
    new,b=preserve_native_geometry(x,delta,mask,pocket,pm,1.,{**constraints,'max_pair_distance_change_A':.01})
    assert b['geometry_accepted'].all() and b['backtrack_factor'].item()<1
    difference=(torch.cdist(x+new,x+new)-torch.cdist(x,x)).abs().max()
    assert difference<=.01 and new.norm()<old.norm()
