import copy
import numpy as np
import pytest
import torch
from test_coordinate_shape_reward import fixture
from evomolsteer.generation.coordinate_conditioning import (
    CoordinateCountConditionedShapeReward, conditional_record, tensor_min_eigenvalue)


def conditioned():
    r = fixture()
    centers = [[1.,1.,1.,.1,.1,.1], [1.1,1.1,1.1,.1,.1,.1]]
    modes = [{'center_scaled':c,'covariance_dimensionless':np.eye(6).tolist(),
              'source_batch':i,'n_available':3,'selected_mass':[.2,.1][i]} for i,c in enumerate(centers)]
    r.reference['batches']=[0,1]
    r.reference['conditional_support']={'min_group_size':3,'min_batches':2,'min_prior_ESS':1.5,
        'max_LOO_center_change_scaled_RMS':1.}
    r.reference['conditional_frames'] = [{'time':t,'combinations':[{'counts':[7,0,0],
        'modes':copy.deepcopy(modes),'mode_prior':[2/3,1/3]}]} for t in r.times]
    return CoordinateCountConditionedShapeReward(r.program,r.reference)


def test_exact_counts_supported_fd_and_unsupported_native_fallback():
    r = conditioned();rng=torch.Generator().manual_seed(42)
    x=torch.randn(2,7,3,dtype=torch.float64,generator=rng,requires_grad=True)
    anchor=x.detach().clone().requires_grad_(True)
    atoms=torch.ones((2,7),dtype=torch.long);atoms[1,0]=2
    mask=torch.ones((2,7),dtype=torch.bool)
    value,d=r(x,atoms,mask,.2,anchor)
    g,ga=torch.autograd.grad(value.sum(),(x,anchor),allow_unused=True)
    assert ga is None and d['available'].tolist()==[True,False] and d['dose_gate'][1]==0 and g[1].eq(0).all()
    assert torch.isfinite(g).all() and d['conditional_component_ESS'][0]>1
    direction=g/g.norm();epsilon=1e-5
    numerical=(r(x+epsilon*direction,atoms,mask,.2,anchor)[0].sum()-r(x-epsilon*direction,atoms,mask,.2,anchor)[0].sum())/(2*epsilon)
    torch.testing.assert_close(numerical,(g*direction).sum(),rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(g.sum(1),torch.zeros((2,3),dtype=torch.float64),atol=1e-12,rtol=0)


def test_conditional_mass_and_root_ess_do_not_treat_clones_as_independent():
    v=np.zeros((3,6));v[:,0]=[1.,4.,9.]
    record=conditional_record(v,[.1,.2,.1],[0,0,1],3,np.ones(6))
    assert record['selected_mass']==pytest.approx(.4) and record['root_weight_ESS']==pytest.approx(1.6)
    assert record['unique_roots']==2 and record['n_available']==3
    assert record['center_A2'][0]==pytest.approx(4.5) and record['mean_tensor_min_eigenvalue_A2']==0
    assert np.linalg.eigvalsh(record['covariance_dimensionless']).min()>=.01-1e-12


def test_conditional_reference_rejects_impossible_means_and_duplicate_counts():
    r=conditioned();ref=copy.deepcopy(r.reference)
    ref['conditional_frames'][-1]['combinations'][0]['modes'][0]['center_scaled'][0]=-1.
    with pytest.raises(ValueError,match='PSD'):CoordinateCountConditionedShapeReward(r.program,ref)
    ref=copy.deepcopy(r.reference);ref['conditional_frames'][0]['combinations']*=2
    with pytest.raises(ValueError,match='Distinct'):CoordinateCountConditionedShapeReward(r.program,ref)


def test_conditional_reference_rejects_duplicate_batches_or_unstable_prior():
    r=conditioned();ref=copy.deepcopy(r.reference)
    ref['conditional_frames'][0]['combinations'][0]['modes'][1]['source_batch']=0
    with pytest.raises(ValueError,match='batch support'):CoordinateCountConditionedShapeReward(r.program,ref)
    ref=copy.deepcopy(r.reference);combo=ref['conditional_frames'][0]['combinations'][0]
    combo['mode_prior']=[.95,.05];combo['modes'][0]['selected_mass']=.95;combo['modes'][1]['selected_mass']=.05
    with pytest.raises(ValueError,match='ESS'):CoordinateCountConditionedShapeReward(r.program,ref)
