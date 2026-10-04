import copy
import numpy as np
import pytest
import torch
from evomolsteer.generation.multistage_reward import MultistageReward, native_relative_step, preserve_native_geometry
from evomolsteer.generation.prototypes import measure_patch, fit_local_stages, regularize_covariance, retained_copies


def fixture():
    fs=['p::hetero_distance_softmin','q::hetero_distance_softmin']
    catalog={'atom_vocabulary':{'N':4,'O':5,'S':9},
        'features':{f:{'kind':'hetero_distance_softmin','elements':['N','O','S'],'region':f[0],'temperature_A':.25} for f in fs},
        'regions':{'p':{'points_A':[[0.,0.,0.],[0.,1.,0.]]},'q':{'points_A':[[2.,0.,0.],[2.,1.,0.]]}}}
    program={'schema_version':'live-multistage-1.0','representation':'predicted_endpoint_world_A','features':fs,
        'knot_times':[0.,.25,.5],'knot_centers_A':[[[4.,3.],[3.,4.]],[[3.,2.],[2.,3.]],[[2.,2.],[2.5,2.5]]],
        'covariance_times':[0.,.5],'covariance_A2':[[[1.,.5],[.5,1.]],[[.8,.3],[.3,.8]]],
        'mixture_weights':[.5,.5],'pseudo_huber_delta':1.,'mixture_temperature':1.}
    x=torch.tensor([[[4.,1.,.5],[3.,2.,0.],[5.,1.,0.]],[[5.,1.,0.],[4.,2.,0.],[3.,1.,0.]]],dtype=torch.float64)
    atoms=torch.tensor([[4,3,5],[5,3,9]]);mask=torch.ones(2,3,dtype=torch.bool)
    return MultistageReward(program,catalog),x,atoms,mask


def test_typed_observable_numerical_identity():
    reward,x,atoms,mask=fixture()
    actual,valid=reward.observables(x,atoms,mask)
    expected=measure_patch(x.numpy(),atoms.numpy(),mask.numpy(),reward.catalog,reward.features)
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12)
    assert valid.all()


def test_live_chain_rule_fixed_atoms_batch_and_full_horizon():
    reward,x,atoms,mask=fixture(); eps=1e-6
    for time in [0.,.08,.25,.5,.51,.75,.99]:
        xx=x.clone().requires_grad_(True)
        def value(v):return reward(1.7*v,atoms,mask,time)[0]
        r=value(xx);g,=torch.autograd.grad(r.sum(),xx)
        assert torch.isfinite(g).all() and g.norm()>0
        d=torch.zeros_like(x); d[0,0,0]=eps
        numeric=(value(x+d)[0]-value(x-d)[0])/(2*eps)
        assert numeric==pytest.approx(g[0,0,0].item(),rel=1e-6,abs=1e-8)
        alone=x[:1].clone().requires_grad_(True)
        alone_g,=torch.autograd.grad(reward(1.7*alone,atoms[:1],mask[:1],time)[0].sum(),alone)
        torch.testing.assert_close(g[:1],alone_g)
    np.testing.assert_allclose(reward.references(.99)[0],reward.references(.5)[0])
    np.testing.assert_allclose(reward.references(0,static=True)[0],reward.references(.5)[0])
    with pytest.raises(ValueError):reward.references(1.1)


def test_missing_types_zero_safe_gradient_not_favorable_observation():
    reward,x,atoms,mask=fixture();atoms[0]=3
    x.requires_grad_(True)
    r,z,valid,_=reward(x,atoms,mask,.7)
    g,=torch.autograd.grad(r.sum(),x)
    assert not valid[0] and valid[1] and torch.isfinite(g).all()
    assert not g[0].count_nonzero() and r[0]==0


def test_permutation_and_joint_rigid_transform():
    reward,x,a,m=fixture();r=reward(x,a,m,.1)[0]
    torch.testing.assert_close(r,reward(x.flip(1),a.flip(1),m.flip(1),.1)[0])
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]); shift=np.array([3.,-1.,2.])
    other=copy.deepcopy(reward.catalog)
    for region in other['regions'].values():region['points_A']=(np.array(region['points_A'])@rot+shift).tolist()
    transformed=MultistageReward(reward.program,other)
    torch.testing.assert_close(r,transformed(x@torch.tensor(rot)+torch.tensor(shift),a,m,.1)[0])


def test_local_knots_covariance_and_ancestry():
    t=np.linspace(0,.5,51);centers=np.stack([np.exp(-20*t),np.sin(20*t)],-1)[:,None,:]
    cov=np.repeat(np.eye(2)[None],51,axis=0)
    knots,audit=fit_local_stages(t,centers,cov,.02)
    assert knots[0]==0 and knots[-1]==50 and audit['maximum_whitened_error']<=.02
    regularized=regularize_covariance(np.zeros_like(cov))
    assert np.linalg.eigvalsh(regularized).min()>=.01-1e-12
    offspring=np.array([[2,0,1],[0,2,1],[2,0,1]])
    parents=np.array([[0,1,2],[0,0,2],[1,1,2]])
    assert retained_copies(offspring,parents).tolist()==[[2,0,1],[0,2,1],[2,0,1]]


def test_native_relative_strength_and_caps():
    _,x,_,mask=fixture();g=torch.ones_like(x);native=g*.1
    step,audit=native_relative_step(g,native,mask,.15,1.,10.,torch.ones(2)*10)
    torch.testing.assert_close(step,native*.15)
    limited,_=native_relative_step(g*1e3,native,mask,.3,1.,.025,torch.tensor([.01,10.]))
    assert limited.norm(dim=-1).max()<=.025+1e-12
    assert (limited[0].square().sum()/3).sqrt()<=.01+1e-12
    zero,_=native_relative_step(g,native,mask,0.,1.,.025,torch.ones(2))
    assert not zero.count_nonzero()
    no_g,_=native_relative_step(g*0,native,mask,.3,1.,.025,torch.ones(2))
    assert not no_g.count_nonzero()


def test_geometry_backtracking_preserves_native_and_fixed_mask():
    native=torch.tensor([[[1.201,0.,0.],[3.,0.,0.]]],dtype=torch.float64)
    original=native.clone();delta=torch.tensor([[[-.025,0.,0.],[0.,0.,0.]]],dtype=torch.float64)
    m=torch.ones(1,2,dtype=torch.bool);p=torch.zeros(1,1,3,dtype=torch.float64);pm=torch.ones(1,1,dtype=torch.bool)
    c={'severe_receptor_clash_A':1.2,'max_pair_distance_change_A':.05,'backtrack_attempts':5}
    actual,guard=preserve_native_geometry(native,delta,m,p,pm,1.,c)
    assert not actual.count_nonzero() and not guard['geometry_accepted'][0]
    torch.testing.assert_close(native,original)
    native[0,0,0]=1.21
    actual,guard=preserve_native_geometry(native,delta,m,p,pm,1.,c)
    assert guard['backtrack_factor'][0]==.25


def test_invalid_reference_rejected():
    reward,*_=fixture();bad=copy.deepcopy(reward.program);bad['covariance_A2'][0]=[[1.,2.],[2.,1.]]
    with pytest.raises(ValueError,match='positive definite'):MultistageReward(bad,reward.catalog)


def test_empirical_distribution_distance_preserves_multisets():
    from evomolsteer.generation.multistage_evaluation import energy_distance_squared, nearest_pose_chamfer
    x=np.array([[1.,2.],[2.,3.],[1.,2.]])
    assert energy_distance_squared(x,x[::-1])==pytest.approx(0,abs=1e-12)
    assert energy_distance_squared(x,x+3)>0
    assert np.isnan(energy_distance_squared(x[:0],x))
    shape=np.array([[[0.,0.,0.],[2.,0.,0.],[0.,0.,0.]]])
    mask=np.array([[True,True,False]])
    np.testing.assert_allclose(nearest_pose_chamfer(shape,mask,shape[:,::-1],mask[:,::-1]),0)
    np.testing.assert_allclose(nearest_pose_chamfer(shape,mask,shape+np.array([0.,1.,0.]),mask),1.)


def test_full_horizon_summary_does_not_hide_missing_atoms_or_late_activity():
    import pandas as pd
    from evomolsteer.generation.multistage_evaluation import summarize_control
    records=[]
    for step in range(100):
        for slot in range(2):
            valid=slot==0
            records.append(dict(arm='multistage_full',batch=0,slot=slot,step=step,time=step*.01,
                observable_available=valid,active=True,evaluated=True,requested_rms_A=.01,
                injection_rms_A=.005 if valid else 0.,gradient_norm=1. if valid else 0.,
                geometry_accepted=True,backtrack_factor=1.,cap_factor=.5,
                injection_max_atom_A=.008 if valid else 0.,cumulative_injection_rms_A=(step+1)*.005 if valid else 0.,
                applied_native_rms_ratio=.05 if valid else 0.,gradient_native_cosine=.2))
    result=summarize_control(pd.DataFrame(records)).iloc[0]
    assert result.available_active_particle_steps==100
    assert result.evaluated_steps_per_batch_min==100
    assert result.nonzero_injection_steps_per_batch_min==100
    assert result.median_applied_over_requested==pytest.approx(.5)
    assert result.late_nonzero_particle_fraction==pytest.approx(.5)
