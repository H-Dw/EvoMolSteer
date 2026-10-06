import numpy as np
import pytest
import torch
from evomolsteer.continuous.coordinate_features import regional_observables
from evomolsteer.generation.coordinate_shape import CoordinateShapeReward, dimensionless_moments, time_weights


def fixture():
    ref = {'schema_version':'regional-shape-mixture-1.0', 'window':[.2,.7], 'times':[.2,.7],
        'channel':'NOS','atom_vocabulary':{'N':1,'O':2,'S':3}, 'spatial_width_A':4.,
        'feature_unit':'A^2','spatial_anchor':'endpoint','control_representation':'proposal',
        'regions':{'patch':{'points_A':[[0.,0,0],[1.,0,0]]}}, 'feature_scale_A2':[1.,2.,3.,4.,5.,6.],
        'frames':[{'time':t,'modes':[{'center_scaled':[.4]*6,'covariance_dimensionless':np.eye(6).tolist()}]} for t in (.2,.7)]}
    program = {'window':[.2,.7], 'mixture_temperature':.25,'robust_delta':1.,'core_radius_A':5.}
    return CoordinateShapeReward(program,ref)


def test_tensor_matches_analysis_numpy_and_permutation_and_translation():
    reward = fixture();rng = np.random.default_rng(42)
    x = rng.normal(size=(2,5,3));anchor = rng.normal(size=(2,5,3));atoms = np.ones((2,5),int);mask = np.ones((2,5),bool)
    catalog = {'regions':reward.reference['regions'],'atom_vocabulary':reward.reference['atom_vocabulary']}
    v,names,_ = regional_observables(x,anchor,x,atoms,mask,catalog,.2,.01,4.,'endpoint',True,'shape')
    columns = [names.index('patch::NOS::proposal_shape_'+k) for k in ('xx','yy','zz','xy','xz','yz')]
    raw,valid,_ = reward.observables(torch.tensor(x),torch.tensor(atoms),torch.tensor(mask),torch.tensor(anchor))
    np.testing.assert_allclose(raw.numpy(),v[:,columns],rtol=1e-13,atol=1e-13)
    assert valid.all()
    shifted = reward.observables(torch.tensor(x+10),torch.tensor(atoms),torch.tensor(mask),torch.tensor(anchor))[0]
    torch.testing.assert_close(raw,shifted,rtol=1e-12,atol=1e-12)
    order = [3,1,4,0,2]
    permuted = reward.observables(torch.tensor(x[:,order]),torch.tensor(atoms[:,order]),torch.tensor(mask[:,order]),torch.tensor(anchor[:,order]))[0]
    torch.testing.assert_close(raw,permuted,rtol=1e-12,atol=1e-12)


def test_conditional_tensor_fd_no_translation_force_and_native_dtype():
    reward = fixture();generator=torch.Generator().manual_seed(42)
    x=torch.randn(2,7,3,dtype=torch.float64,generator=generator,requires_grad=True)
    anchor=torch.randn(2,7,3,dtype=torch.float64,generator=generator,requires_grad=True)
    atoms=torch.ones((2,7),dtype=torch.long);mask=torch.ones((2,7),dtype=torch.bool)
    value,_=reward(x,atoms,mask,.2,anchor)
    g,ga=torch.autograd.grad(value.sum(),(x,anchor),allow_unused=True)
    assert ga is None
    torch.testing.assert_close(g.sum(1),torch.zeros(2,3,dtype=torch.float64),atol=1e-12,rtol=0)
    direction=g/g.norm();eps=1e-5
    num=(reward(x+eps*direction,atoms,mask,.2,anchor)[0].sum()-reward(x-eps*direction,atoms,mask,.2,anchor)[0].sum())/(2*eps)
    torch.testing.assert_close(num,(g*direction).sum(),rtol=1e-7,atol=1e-9)
    xf=x.detach().float().requires_grad_(True)
    gf,=torch.autograd.grad(reward(xf,atoms,mask,.2,anchor)[0].sum(),xf)
    assert gf.dtype==torch.float32 and torch.isfinite(gf).all()
    assert reward.active(.2,.21) and not reward.active(.7,.71)


def test_uninformative_one_slot_and_missing_nos_produce_no_dose_or_gradient():
    reward=fixture();x=torch.randn(2,3,3,dtype=torch.float64,requires_grad=True)
    atoms=torch.tensor([[1,0,0],[0,0,0]]);mask=torch.ones((2,3),dtype=torch.bool)
    value,detail=reward(x,atoms,mask,.2,x.detach())
    gradient,=torch.autograd.grad(value.sum(),x)
    assert not detail['available'].any() and value.eq(0).all() and gradient.eq(0).all() and detail['dose_gate'].eq(0).all()
    with pytest.raises(ValueError,match='active atom'):
        reward(x,atoms,torch.zeros_like(mask),.2,x.detach())


def test_fixed_whole_window_scaling_has_consistent_units_and_positive_covariance():
    weights=time_weights([.2,.3,.7]);np.testing.assert_allclose(weights,[.1,.5,.4])
    v=np.arange(60,dtype=float).reshape(10,6);prob=np.ones(10)/10;scale=np.arange(1.,7.)
    m=dimensionless_moments(v,prob,scale)
    np.testing.assert_allclose(np.asarray(m['center_scaled'])*scale,v.mean(0))
    assert np.linalg.eigvalsh(m['covariance_dimensionless']).min()>=.01-1e-12
    with pytest.raises(ValueError):time_weights([.2,.2,.7])


def test_reference_rejects_bad_late_covariance_and_wrong_units_before_execution():
    import copy
    r=fixture()
    for key,value in [('feature_unit','A'),('feature_scale_A2',[1.]*4)]:
        ref=copy.deepcopy(r.reference);ref[key]=value
        with pytest.raises(ValueError):CoordinateShapeReward(r.program,ref)
    ref=copy.deepcopy(r.reference);ref['frames'][-1]['modes'][0]['covariance_dimensionless'][0][0]=-1.
    with pytest.raises(ValueError,match='Positive definite'):CoordinateShapeReward(r.program,ref)
