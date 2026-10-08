import torch
import pytest
from evomolsteer.generation.dynamic_region_reward import regional_response

def test_regional_scalar_finite_difference_and_label_swap():
    z=torch.tensor([[.4,-.2]],dtype=torch.float64,requires_grad=True)
    p=[[1.,0.],[1.5,.4]];n=[[-1.,0.],[-1.5,-.4]];var=[1.,.5]
    r,_=regional_response(z,p,n,var);g,=torch.autograd.grad(r.sum(),z)
    swapped,_=regional_response(z,n,p,var);h,=torch.autograd.grad(swapped.sum(),z)
    torch.testing.assert_close(g,-h)
    d=g/g.norm();eps=1e-5
    a,_=regional_response(z+eps*d,p,n,var);b,_=regional_response(z-eps*d,p,n,var)
    torch.testing.assert_close(((a-b)/(2*eps)).sum(),(g*d).sum(),rtol=1e-6,atol=1e-6)

def test_identical_cohorts_cannot_create_signal():
    z=torch.tensor([[.3,.7]],requires_grad=True);p=[[0.,0.],[1.,1.]]
    r,_=regional_response(z,p,p,[1,1]);g,=torch.autograd.grad(r.sum(),z)
    assert r.item()==0 and torch.count_nonzero(g)==0

def test_common_bandwidth_contrast_is_bounded():
    z=torch.tensor([[100.,-100.]])
    r,_=regional_response(z,[[1,1]],[[-1,-1]],[1,1],bound=2)
    assert torch.isfinite(r).all() and abs(r.item())<=2

def test_zero_addition_recovers_incumbent_scalar_and_gradient():
    import copy
    from evomolsteer.generation.dynamic_region_reward import DynamicRegionReward
    from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
    from evomolsteer.continuous.affinity_geometry import names
    fields=names([[0,0,0]])
    r={'schema_version':'affinity-endpoint-library-1.0','window':[.2,.7],'times':[.2],
       'features':fields,'feature_scale':[1.]*len(fields),'landmarks_A':[[0,0,0]],'origin_A':[0,0,0],
       'reference_variant':'dynamic-cohort-endpoint-library-1.0','dynamic_cohort':{'selected_feature_indices':[14]},
       'frames':[{'teacher_endpoint_A':[[[1,0,0],[1,1,0]]],'teacher_scores':[2.],
           'dynamic_region':{'positive_centers':[[0.]*len(fields)],'negative_centers':[[1.]*len(fields)],
               'feature_scale':[1.]*len(fields),'positive_variance':[1.]*len(fields),'negative_variance':[1.]*len(fields)}}]}
    p={'window':[.2,.7],'reward_view':'endpoint_dynamic_region','regional_weight':0.,'derivative_path':'flowr_endpoint_vjp'}
    a=DynamicRegionReward(p,r);old=copy.deepcopy(p);old['reward_view']='endpoint_pointcloud';b=EndpointGeometryReward(old,r)
    x=torch.tensor([[[.3,0,0],[.2,1,0]]],requires_grad=True);mask=torch.ones((1,2),dtype=torch.bool);atoms=torch.zeros((1,2),dtype=torch.long)
    v,_=a(x,atoms,mask,.2,x.detach());w,_=b(x,atoms,mask,.2,x.detach())
    torch.testing.assert_close(v,w,rtol=0,atol=0)
    g,=torch.autograd.grad(v.sum(),x);h,=torch.autograd.grad(w.sum(),x);torch.testing.assert_close(g,h,rtol=0,atol=0)
    assert a.active(.2,.21) and not a.active(.7,.71)
