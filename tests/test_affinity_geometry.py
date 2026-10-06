import numpy as np
import torch
import pytest
from evomolsteer.continuous.affinity_geometry import numpy_geometry,torch_geometry

def test_coordinate_features_ignore_atom_order_and_match_torch():
    rng=np.random.default_rng(42);x=rng.normal(size=(2,8,3));p=rng.normal(size=(3,3));o=np.array([1,2,3])
    a=numpy_geometry(x,p,o);b=torch_geometry(torch.tensor(x),p,o).numpy()
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a,numpy_geometry(x[:,::-1],p,o),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a,numpy_geometry(x+10,p+10,o+10),rtol=1e-12,atol=1e-12)

def test_coordinate_fields_have_real_spatial_derivatives():
    rng=np.random.default_rng(42);x=torch.tensor(rng.normal(size=(2,8,3)),requires_grad=True);p=rng.normal(size=(3,3));o=np.zeros(3)
    v=torch_geometry(x,p,o).square().sum();g,=torch.autograd.grad(v,x);d=g/g.norm();eps=1e-5
    numerical=(torch_geometry(x+eps*d,p,o).square().sum()-torch_geometry(x-eps*d,p,o).square().sum())/(2*eps)
    torch.testing.assert_close(numerical,(g*d).sum(),rtol=1e-5,atol=1e-5)

def pointcloud_fixture():
    from evomolsteer.continuous.affinity_geometry import names
    a=np.zeros((2,3));a[:,0]=-1;b=-a;points=np.zeros((1,3))
    ref={'schema_version':'affinity-coordinate-library-1.0','window':[0,.5],'times':[0.],
        'features':names(points),'feature_scale':[1.]*len(names(points)),'landmarks_A':points.tolist(),'origin_A':[0,0,0],
        'frames':[{'teacher_endpoint_A':[a.tolist(),b.tolist()],'teacher_proposal_A':[a.tolist(),b.tolist()],'teacher_scores':[0.,0.]}]}
    p={'window':[0,.5],'reward_view':'affinity_pointcloud','teacher_neighbors':2,'teacher_score_beta':0.,'mixture_temperature':.5}
    return p,ref,torch.tensor(a)[None]

def test_pointcloud_mixture_does_not_prefer_compromise_to_modes():
    from evomolsteer.generation.affinity_geometry_reward import AffinityGeometryReward
    p,ref,a=pointcloud_fixture();r=AffinityGeometryReward(p,ref);mask=torch.ones((1,2),dtype=torch.bool);atoms=torch.zeros((1,2),dtype=torch.int64)
    at,_=r(a,atoms,mask,0.,torch.zeros_like(a));mid,_=r(torch.zeros_like(a),atoms,mask,0.,torch.zeros_like(a))
    assert at.item()>mid.item()
    x=(a*.4).requires_grad_(True);v,_=r(x,atoms,mask,0.,torch.zeros_like(a));g,=torch.autograd.grad(v.sum(),x)
    assert torch.isfinite(g).all() and g.norm()>0
    swapped,_=r(x.flip(1),atoms+3,mask,0.,torch.zeros_like(a).flip(1));torch.testing.assert_close(v,swapped)

@pytest.mark.parametrize('change',[{'geometry_block_weights':[-1,1,1,1]},{'geometry_block_weights':[0,0,0,0]},
    {'teacher_endpoint_temperature_A2':0},{'time_ramp_power':float('nan')},{'mixture_temperature':float('nan')}])
def test_invalid_geometry_controls_fail_before_inference(change):
    from evomolsteer.generation.affinity_geometry_reward import AffinityGeometryReward
    p,ref,_=pointcloud_fixture();p.update(change)
    with pytest.raises(ValueError):AffinityGeometryReward(p,ref)
