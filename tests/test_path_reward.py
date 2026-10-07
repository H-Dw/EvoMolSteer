import copy
import numpy as np
import torch
import pytest
from evomolsteer.generation.path_reward import PathValueReward
def fixture():
    f={'time':.2,'teacher_endpoint_A':[[[-1.,0.,0.],[0.,1.,0.]],[[1.,0.,0.],[2.,1.,0.]]],
       'teacher_scores':[7.,9.],'teacher_path_ids':[['a'],['b']],'teacher_base_log_weight':[-np.log(2)]*2}
    r={'window':[.2,.6],'times':[.2,.3],'frames':[f,{**copy.deepcopy(f),'time':.3}],
       'reference_variant':'decoded-terminal-path-library-1.0'}
    p={'window':r['window'],'derivative_path':'flowr_endpoint_vjp','teacher_neighbors':2}
    x=torch.tensor([[[0.,0.,0.],[1.,1.,0.]]],dtype=torch.double,requires_grad=True)
    return p,r,x
def gradient(p,r,x,t=.2):
    reward=PathValueReward(p,r);mask=torch.ones(x.shape[:2],dtype=torch.bool)
    v,d=reward(x,mask.long(),mask,t,x.detach());g=torch.autograd.grad(v.sum(),x)[0]
    return reward,v,g,d
def test_true_scalar_derivative_points_toward_higher_future():
    p,r,x=fixture();reward,v,g,d=gradient(p,r,x)
    assert g[:,:,0].sum()>0
    eps=1e-5;mask=torch.ones(x.shape[:2],dtype=torch.bool);direction=g/g.norm()
    hi,_=reward(x+eps*direction,mask,mask,.2,x.detach());lo,_=reward(x-eps*direction,mask,mask,.2,x.detach())
    assert float((hi-lo)/(2*eps))==pytest.approx(float(g.norm()),rel=1e-5)
def test_constant_labels_and_single_neighbor_are_plateaus():
    p,r,x=fixture();r['frames'][0]['teacher_scores']=[8,8]
    assert gradient(p,r,x)[2].abs().max()==0
    p['teacher_neighbors']=1;r['frames'][0]['teacher_scores']=[7,9]
    assert gradient(p,r,x)[2].abs().max()==0
def test_label_shuffle_reverses_coordinate_direction():
    p,r,x=fixture();first=gradient(p,r,x)[2];r['frames'][0]['teacher_scores'].reverse()
    assert float((first*gradient(p,r,x)[2]).sum())<0
def test_history_uses_real_ids_and_rejects_broken_paths():
    p,r,x=fixture();p['path_history_mix']=.5;reward=PathValueReward(p,r)
    reward.set_history(x.detach(),.2);prior=reward.prior(r['frames'][1],x.detach().numpy(),.3)
    assert np.allclose(prior.sum(1),1)
    r['frames'][1]['teacher_path_ids']=[['missing'],['missing2']]
    with pytest.raises(ValueError):reward.prior(r['frames'][1],x.detach().numpy(),.3)
def test_dynamic_support_and_invalid_parameters():
    p,r,x=fixture();reward=PathValueReward(p,r)
    assert reward.active(.2,.21) and not reward.active(.59,.61) and not reward.active(.1,.2)
    p['path_history_mix']=1
    with pytest.raises(ValueError):PathValueReward(p,r)
