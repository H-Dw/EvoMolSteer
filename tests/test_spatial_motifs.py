import copy
import numpy as np
import pytest
import torch
from evomolsteer.continuous.spatial_motifs import numpy_motifs,torch_motifs
from evomolsteer.generation.motif_reward import MotifReward


def fixture():
    rng=np.random.default_rng(42);x=rng.normal(size=(3,6,3));a=x+.3
    labels=np.array([[0,1,2,0,1,2]]*3);mask=np.ones((3,6),bool)
    cat={'regions':{'r':{'points_A':[[0.,0.,0.],[0.,1.,0.]]}},'atom_vocabulary':{'C':0,'N':1,'O':2,'S':3}}
    return x,a,labels,mask,cat


def test_numpy_torch_permutation_equivalence_and_live_derivative():
    x,a,labels,mask,cat=fixture();v,names,_=numpy_motifs(x,labels,mask,cat,a)
    xx=torch.tensor(x,requires_grad=True);args=(torch.tensor(labels),torch.tensor(mask),cat,torch.tensor(a))
    actual,valid,_=torch_motifs(xx,*args)
    np.testing.assert_allclose(actual.detach(),v,atol=1e-12)
    assert valid.all() and len(names)==27
    perm=[3,1,4,0,5,2];other,*_=numpy_motifs(x[:,perm],labels[:,perm],mask[:,perm],cat,a[:,perm])
    np.testing.assert_allclose(other,v,atol=1e-12)
    torch.autograd.gradcheck(lambda z:torch_motifs(z,*args)[0],(xx,),atol=1e-5)


def reference():
    x,a,labels,mask,cat=fixture();v,names,_=numpy_motifs(x,labels,mask,cat,a,channels=('all',))
    mode={'center_scaled':np.mean(v,0).tolist(),'background_center_scaled':(np.mean(v,0)+.1).tolist(),
          'covariance_dimensionless':np.eye(9).tolist(),'background_covariance_dimensionless':(np.eye(9)*1.2).tolist()}
    ref={**cat,'schema_version':'spatial-motif-mixture-1.0','window':[.1,.4],'times':[.1,.2,.4],
         'feature_scale':[1.]*9,'features':names,'channels':['all'],'spatial_width_A':4.,
         'frames':[{'time':t,'modes':[mode]} for t in [.1,.2,.4]]}
    p={'window':[.1,.4],'mixture_temperature':.5,'robust_delta':2.,'core_radius_A':5.,'reward_view':'motif_mixture'}
    return p,ref,x,a,labels,mask


@pytest.mark.parametrize('view',['motif_mixture','motif_contrast'])
@pytest.mark.parametrize('component',['joint','pair','shell','centroid'])
def test_reward_gradcheck_and_dynamic_window(view,component):
    p,ref,x,a,labels,mask=reference();p.update(reward_view=view,motif_components=component,time_ramp_power=1.)
    reward=MotifReward(p,ref);xx=torch.tensor(x,requires_grad=True)
    f=lambda z:reward(z,torch.tensor(labels),torch.tensor(mask),.2,torch.tensor(a))[0]
    torch.autograd.gradcheck(f,(xx,),atol=1e-5)
    _,detail=reward(xx,torch.tensor(labels),torch.tensor(mask),.2,torch.tensor(a))
    assert detail['dose_gate'].gt(0).all()
    assert reward.active(.1,.11) and not reward.active(.4,.41)
    with pytest.raises(ValueError):reward(xx,torch.tensor(labels),torch.tensor(mask),.25,torch.tensor(a))


def test_empty_nos_is_undefined_not_graph_veto():
    x,a,labels,mask,cat=fixture();labels[:]=0
    v,valid,_=torch_motifs(torch.tensor(x),torch.tensor(labels),torch.tensor(mask),cat,torch.tensor(a),channels=('NOS',))
    assert not valid.any() and torch.isfinite(v).all()


def test_zero_empirical_contrast_has_zero_dose_after_normalization():
    p,ref,x,a,labels,mask=reference();p['reward_view']='motif_contrast'
    for frame in ref['frames']:
        for mode in frame['modes']:
            mode['background_center_scaled']=copy.deepcopy(mode['center_scaled'])
            mode['background_covariance_dimensionless']=copy.deepcopy(mode['covariance_dimensionless'])
    value,detail=MotifReward(p,ref)(torch.tensor(x),torch.tensor(labels),torch.tensor(mask),.2,torch.tensor(a))
    assert value.abs().max()==0 and detail['dose_gate'].abs().max()==0
