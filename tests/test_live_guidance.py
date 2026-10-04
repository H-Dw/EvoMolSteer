import copy
import numpy as np
import pytest
import torch
from evomolsteer.evidence import focus_arm
from evomolsteer.generation.scalar_guidance import RegionalReward,live_pullback,bounded_displacement,time_gate
from evomolsteer.generation.gradient_controller import gradient_source


def example():
    catalog={'features':{'rg':{'kind':'radius_gyration'}},'regions':{}}
    term={'id':'compact','feature':'rg','operator':'decrease_until_target','target':.1,'scale':1.,'weight':1.,
          'stage_start':0.,'stage_end':.2,'gate_width':.02}
    reward=RegionalReward({'schema_version':'live-regional-1.0','representation':'predicted_endpoint_world_A','terms':[term]},catalog)
    x=torch.tensor([[[1.,0.,0.],[-1.,0.,0.]],[[2.,0.,0.],[-2.,0.,0.]]],dtype=torch.float64)
    return reward,x,torch.ones(2,2,dtype=torch.bool)


def test_focus_arm_never_falls_back_to_joint():
    assert focus_arm({'selection_arms':['single'],'evidence_arm':'single'})=='single'
    with pytest.raises(ValueError):focus_arm({'selection_arms':['single']})


def test_live_chain_rule_and_no_batch_average():
    reward,x,mask=example()
    def forward(v):return {'coords':2*v},{'coords':v}
    pred,cond,g,values,_=live_pullback(forward,reward,x,mask,3.,torch.zeros(2,3),0.,['compact'])
    assert not pred['coords'].requires_grad and not cond['coords'].requires_grad
    eps=1e-6
    d=x.clone()*0;d[0,0,0]=1
    numeric=(reward(6*(x+eps*d),mask,0.,['compact'])[0][0]-reward(6*(x-eps*d),mask,0.,['compact'])[0][0])/(2*eps)
    assert numeric.item()==pytest.approx(g[0,0,0].item(),rel=1e-6)
    _,_,alone,_,_=live_pullback(forward,reward,x[:1],mask[:1],3.,torch.zeros(1,3),0.,['compact'])
    torch.testing.assert_close(alone[0],g[0])
    assert (g*x).sum()<0


def test_support_saturation_and_cap():
    reward,x,mask=example()
    xx=x.clone().requires_grad_()
    inactive=reward(xx,mask,.2,['compact'])[0]
    assert torch.count_nonzero(torch.autograd.grad(inactive.sum(),xx)[0])==0
    for time in [-.1,.2,.3,.5,1.]:assert time_gate(time,0,.2,.02)==0
    reward.terms[0]['target']=100.
    assert torch.count_nonzero(torch.autograd.grad(reward(xx,mask,0.,['compact'])[0].sum(),xx)[0])==0
    g=torch.ones_like(x)*100
    delta,factor=bounded_displacement(g,.01,1.,3.,.025)
    assert float(delta.norm(dim=-1).max())*3==pytest.approx(.025)
    zero,_=bounded_displacement(g,.01,0.,3.,.025)
    assert torch.count_nonzero(zero)==0


def test_normalized_softmin_matches_numpy_and_rigid_motion():
    reward,x,mask=example()
    reward.catalog['features']['distance']={'kind':'distance_softmin','region':'p','temperature_A':.25}
    points=[[4.,0.,0.],[4.,1.,0.]];reward.catalog['regions']['p']={'points_A':points}
    z=reward.observable('distance',x,mask)
    from scipy.special import logsumexp
    d=np.linalg.norm(x.numpy()[:,:,None,:]-np.array(points)[None,None,:,:],axis=-1)
    expected=-.25*(logsumexp(-d/.25,axis=(1,2))-np.log(4))
    np.testing.assert_allclose(z.numpy(),expected)
    shifted=copy.deepcopy(reward);shifted.catalog['regions']['p']['points_A']=(np.array(points)+3).tolist()
    torch.testing.assert_close(z,shifted.observable('distance',x+3,mask))


def test_bad_live_reward_rejected():
    reward,x,mask=example()
    def forward(v):return {'coords':v*float('nan')},{}
    with pytest.raises(ValueError,match='Invalid reward'):live_pullback(forward,reward,x,mask,1.,torch.zeros(2,3),0.,['compact'])


def test_source_instrumentation_fails_closed():
    with pytest.raises(RuntimeError):gradient_source('def other():\n    pass\n')
