"""Run on the specified remote host; no local test execution."""
import copy
import math
import numpy as np
import pytest
import torch
from agenticgen.uniguide.condition import condition_map,coordinate_loss,guidance_delta
from agenticgen.uniguide.protocol import validate_action,choose,coarse_scales,expand,regressed
from agenticgen.uniguide.io import Budget,read,frozen
from agenticgen.uniguide.review import teacher_triggers,recover_base,vector_change


def test_fixed_assignment_vjp_and_scale_conversion():
    x=torch.tensor([[[.2,1.,0.],[2.,.4,0.],[3.,1.,.5]]],dtype=torch.double,requires_grad=True)
    # Cross-slot Jacobian deliberately gives non-core coordinates a gradient.
    endpoint=2*(x+.2*x.mean(1,keepdim=True))+torch.tensor([4.,5.,6.])
    target,selected,assignment=condition_map(endpoint,[[4.,5.,6.]],torch.ones(1,3,dtype=torch.bool))
    assert not target.requires_grad and not selected.requires_grad and len(set(assignment[0].tolist()))==1
    gradient=torch.autograd.grad(coordinate_loss(endpoint,target,selected).sum(),x)[0]
    assert gradient[0,2].abs().sum()>0
    direction=torch.tensor([[[.1,.2,.3],[.4,.1,-.2],[-.1,.3,.1]]],dtype=torch.double)
    def f(z): return coordinate_loss(2*(z+.2*z.mean(1,keepdim=True))+torch.tensor([4.,5.,6.]),target,selected).sum()
    eps=1e-6
    numerical=(f(x.detach()+eps*direction)-f(x.detach()-eps*direction))/(2*eps)
    assert torch.allclose(numerical,(gradient*direction).sum(),rtol=1e-7,atol=1e-8)


def test_flow_score_algebra_linear_scale_and_no_clipping():
    x=torch.randn(2,5,3,dtype=torch.double);v=torch.randn_like(x);g=torch.randn_like(x)
    t=torch.tensor([.1,.7],dtype=torch.double);gamma=.4;S=torch.tensor([50.,300.],dtype=torch.double);dt=.01
    q=(t[:,None,None]*v-x)/(1-t+1e-6)[:,None,None]
    q2=q-S[:,None,None]*g
    v2=(q2*(1-t+1e-6)[:,None,None]+x)/t[:,None,None]
    delta,_=guidance_delta(g,t,dt,S,gamma,torch.ones(2,5))
    assert torch.allclose(delta,dt*((v2+gamma*q2)-(v+gamma*q)),rtol=1e-12,atol=1e-12)
    doubled,_=guidance_delta(g,t,dt,2*S,gamma,torch.ones(2,5))
    assert torch.equal(doubled,2*delta) and delta.abs().max()>.025
    zero,_=guidance_delta(g,torch.zeros(2),dt,[0,0],gamma,torch.ones(2,5))
    assert torch.equal(zero,torch.zeros_like(g))
    with pytest.raises(FloatingPointError): guidance_delta(g*float('nan'),t,dt,S,gamma,torch.ones(2,5))


def test_condition_map_injective_and_unmatched_targets_equal_prediction():
    x=torch.tensor([[[0.,0,0],[1.,0,0],[9.,0,0],[.1,0,0]]])
    valid=torch.tensor([[1,1,1,0]],dtype=torch.bool)
    target,mask,ids=condition_map(x,[[.1,0,0],[.2,0,0]],valid)
    assert set(ids[0].tolist())=={0,1}
    assert torch.equal(target[~mask],x[~mask])
    with pytest.raises(ValueError): condition_map(x,[],valid)


def test_log_actions_no_rounding_duplicate_or_hidden_endpoint():
    anchors={'a':{'schedule':[1.]*4}};domain={'min_index':-4,'max_index':4}
    p=dict(anchor_id='a',mode='coarse',branch='redistribute',schedule=[2.,.5,1.,1.])
    assert validate_action(p,anchors,domain,[])==p['schedule']
    for changes in [{'schedule':[3.,.5,1,1]},{'schedule':[2.,1,1,1]},{'mode':'fine'},{'branch':'increase'}]:
        with pytest.raises(ValueError): validate_action(dict(p,**changes),anchors,domain,[])
    with pytest.raises(ValueError): validate_action(p,anchors,domain,[p['schedule']])
    assert len(coarse_scales())==26
    e=expand([1,2,3,4]);assert e[9]==0 and e[10]==1 and e[25]==1 and e[26]==2 and e[90]==4 and e[91]==0


def test_budget_failed_attempts_remain_charged(tmp_path):
    b=Budget(tmp_path);b.reserve('a','calibration',4990,'x');b.reserve('a','calibration',4990,'x')
    with pytest.raises(ValueError): b.reserve('b','calibration',20,'y')
    with pytest.raises(ValueError): b.reserve('a','calibration',10,'y')
    b.reserve('b','calibration',10,'y');assert sum(x['count'] for x in read(tmp_path/'attempts.json').values())==5000


def test_frozen_json_roundtrip_does_not_reject_plip_tuples(tmp_path):
    p=tmp_path/'template.json'
    frozen(p,{'fingerprints':[('hbond',1,'A')]})
    frozen(p,{'fingerprints':[('hbond',1,'A')]})
    with pytest.raises(ValueError): frozen(p,{'fingerprints':[('hbond',2,'A')]})


def row(i,c,g): return {'id':i,'C':c,'gap':g,'best':8.,'complete':True}


def test_anchored_selection_and_two_adjacent_regressions():
    assert choose([row('a',1,.5),row('b',1.019,.1),row('c',1.03,0)])['id']=='b'
    versions=[{'id':f'UG{i+1:02d}','overall':row(f'UG{i+1:02d}',c,g)} for i,(c,g) in enumerate([(1,.5),(1.04,.4),(1.08,.3)])]
    assert recover_base(versions)==('UG01',True)
    versions.append({'id':'UG04','overall':row('UG04',1.12,.2),'recovery_derived':True})
    assert recover_base(versions)==('UG04',False)
    assert regressed(row('a',1,.1),row('b',1.01,.2))
    assert not regressed(row('a',1,.2),row('b',1.01,.1))


def test_stagnation_either_teacher_and_missing_vectors(tmp_path):
    old=row('a',1,.2);new=row('b',1.01,.21)
    evidence=teacher_triggers(old,new,[[1]*4]*15,[[1]*4]*12+[[2]*4]*3,{'known':False})
    assert len(evidence['reasons'])==2
    assert not vector_change(tmp_path/'missing',tmp_path/'missing')['known']
    import h5py
    for name,sign in [('a',1),('b',-1)]:
        with h5py.File(tmp_path/name,'w') as f:f['guidance_delta_A']=np.ones((100,2,3,3))*sign
    changed=vector_change(tmp_path/'a',tmp_path/'b')
    assert changed['known'] and changed['relative_change']==[2.]*4
    assert not changed['all_windows_at_most_5pct']


def test_native_twenty_row_crn_is_shared_and_external_rng_unchanged():
    from agenticgen.uniguide.random_streams import NativeStreams,capture,equal
    states=[]
    for seed in [42,100045,200048,300051,400054]:
        torch.manual_seed(seed);states.append(capture())
    outside=capture()
    streams=NativeStreams(states,3,10)
    samples=streams.normal('randn',(150,17,3),dtype=torch.float32)
    assert torch.equal(samples[:50],samples[50:100]) and torch.equal(samples[:50],samples[100:])
    assert equal(outside,capture())
    probabilities=torch.full((150,17,4),.25)
    cats=streams.categorical(probabilities)
    assert torch.equal(cats[:50],cats[50:100]) and torch.equal(cats[:50],cats[100:])
    assert equal(outside,capture())
