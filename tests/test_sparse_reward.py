from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from evomolsteer.io import read_json
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.generation.sparse_reward import SparseEndpointReward
from evomolsteer.continuous.sparse_design import validate_sparse_fields


def fixture():
    root=Path(__file__).resolve().parents[1]
    ref=load_reference(root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz')
    prior=read_json(root/'docs/experiments/ck2_affinity_geometry30_20261007/endpoint_mining/sparse_coordinate_prior_v2.json')
    p=dict(window=ref['window'],reward_view='endpoint_direction',derivative_path='flowr_endpoint_vjp',
           geometry_feature_weights=prior['feature_weights'],geometry_block_weights=[0,1,2,1],
           geometry_direction_mode='linear_node_effect',geometry_direction_functions=prior['node_effect_functions'],geometry_direction_times=ref['times'])
    return ref,prior,p


def test_supported_reward_spatial_derivative_matches_finite_difference():
    ref,prior,p=fixture();reward=SparseEndpointReward(p,ref)
    x=torch.tensor(ref['frames'][20]['teacher_endpoint_A'][:1],dtype=torch.float64).requires_grad_(True)
    mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    value,_=reward(x,atoms,mask,.2,x.detach());g,=torch.autograd.grad(value.sum(),x);direction=g/g.norm()
    epsilon=1e-5
    a,_=reward(x.detach()+epsilon*direction,atoms,mask,.2,x.detach())
    b,_=reward(x.detach()-epsilon*direction,atoms,mask,.2,x.detach())
    torch.testing.assert_close((a-b).sum()/(2*epsilon),(g*direction).sum(),rtol=1e-5,atol=1e-6)
    for lo,hi,total in [(0,3,0),(3,9,1),(9,14,2),(14,74,1)]:assert reward.field_weights[lo:hi].sum()==pytest.approx(total)
    assert not reward.active(.5,.51)


def test_sparse_validation_rejects_floor_fields_and_invented_functions():
    ref,prior,p=fixture();d={'geometry_feature_weights':prior['feature_weights'],'geometry_block_weights':[0,1,2,1],'direction_mode':'linear_node_effect',
                           'geometry_direction_functions':prior['node_effect_functions']}
    validate_sparse_fields(d,prior)
    bad=copy.deepcopy(d);excluded=next(i for i,v in enumerate(prior['evidence']) if 'scale_floor' in v['excluded_reasons'])
    bad['geometry_feature_weights'][excluded]=1
    with pytest.raises(ValueError):validate_sparse_fields(bad,prior)
    bad=copy.deepcopy(d);bad['geometry_direction_functions']['pair_kernel_2.5']['values_z'][0]+=1
    with pytest.raises(ValueError):validate_sparse_fields(bad,prior)


def test_recorded_pair_curve_changes_sign_without_using_its_time_derivative():
    ref,prior,p=fixture();f=prior['direction_functions']['pair_kernel_2.5']
    first,last=np.polynomial.legendre.legval([-1,1],f['legendre_coefficients'])
    assert first>0 and last<0
    audit=prior['curve_fidelity_audit']['pair_kernel_2.5']
    assert audit['last_empirical_z']>0 and audit['last_fit_z']<0 and not audit['legendre_runtime_allowed']
    d={'geometry_feature_weights':prior['feature_weights'],'geometry_block_weights':[0,1,2,1],'direction_mode':'legendre_effect',
       'geometry_direction_functions':prior['direction_functions']}
    with pytest.raises(ValueError,match='contradicts'):validate_sparse_fields(d,prior)
    bad=copy.deepcopy(p);bad['geometry_block_weights']=[1,0,0,0]
    with pytest.raises(ValueError):SparseEndpointReward(bad,ref)


def test_public_factory_and_controller_cannot_silently_use_dense_formula():
    from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
    from evomolsteer.generation.endpoint_controller import EndpointCoordinateExtension
    import inspect
    ref,prior,p=fixture()
    assert isinstance(make_coordinate_reward(p,ref),SparseEndpointReward)
    assert 'make_coordinate_reward' in inspect.getsource(EndpointCoordinateExtension.configure)
    bad=copy.deepcopy(p);bad['geometry_direction_mode']='legendre_effect';bad['geometry_direction_functions']=prior['direction_functions']
    with pytest.raises(ValueError,match='certificates'):make_coordinate_reward(bad,ref)
