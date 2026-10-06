from pathlib import Path
import copy
import numpy as np
import torch
import pytest
from evomolsteer.io import read_json,digest
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.generation.persistent_reward import PersistentEndpointReward
from evomolsteer.continuous.endpoint_kinematics import parent_mean


def fixture():
    root=Path(__file__).resolve().parents[1];e=root/'docs/experiments/ck2_affinity_geometry30_20261007'
    p=read_json(e/'sparse_skill_test_v2/compiled_design.json');p.update(reward_view='endpoint_supported_attractor',
        history_strength=.5,history_time_power=2.,prototype_robust_delta=2.,geometry_salience_mode='observed_abs_effect',
        history_prior_relative_path=(e/'kinematics_mining_v3/kinematics_prior.json').relative_to(root).as_posix(),
        history_prior_sha256=digest(e/'kinematics_mining_v3/kinematics_prior.json'))
    ref=load_reference(root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz')
    return p,ref


def test_bounded_attraction_and_history_spatial_derivative():
    p,ref=fixture();reward=PersistentEndpointReward(p,ref)
    x=torch.tensor(ref['frames'][25]['teacher_endpoint_A'][:1],dtype=torch.float64)+.08
    previous=torch.tensor(ref['frames'][24]['teacher_endpoint_A'][:1],dtype=torch.float64)
    reward.set_history(previous,.24);x.requires_grad_(True);mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    value,detail=reward(x,atoms,mask,.25,x.detach());g,=torch.autograd.grad(value.sum(),x);d=g/g.norm();eps=1e-5
    a,_=reward(x.detach()+eps*d,atoms,mask,.25);b,_=reward(x.detach()-eps*d,atoms,mask,.25)
    torch.testing.assert_close((a-b).sum()/(2*eps),(g*d).sum(),rtol=1e-5,atol=1e-6)
    assert value.max()<=0 and detail['lineage_increment_penalty'].min()>=0 and detail['history_active'].all()
    assert not reward.active(.5,.51)
    reward.set_history(previous,.23)
    with pytest.raises(ValueError,match='preceding'):reward(x,atoms,mask,.25)


def test_increment_parent_groups_count_each_parent_once():
    values=np.array([[1.],[1.],[1.],[5.]])
    mean,n=parent_mean(values,np.array([0,0,0,1]),np.ones(4,bool))
    assert n==2 and mean[0]==3 # clone-weighted mean would be2


def test_persistent_inputs_reject_wrong_source_and_nonfinite_parameters():
    p,ref=fixture();bad=copy.deepcopy(p);bad['history_strength']=float('nan')
    with pytest.raises(ValueError):PersistentEndpointReward(bad,ref)
    bad=copy.deepcopy(p);bad['history_prior_sha256']='wrong'
    with pytest.raises(ValueError,match='prior'):PersistentEndpointReward(bad,ref)


def test_factory_registers_correct_loss_and_detaches_history():
    from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
    p,ref=fixture();reward=make_coordinate_reward(p,ref)
    assert isinstance(reward,PersistentEndpointReward)
    previous=torch.tensor(ref['frames'][24]['teacher_endpoint_A'][:1],dtype=torch.float64,requires_grad=True)
    x=torch.tensor(ref['frames'][25]['teacher_endpoint_A'][:1],dtype=torch.float64,requires_grad=True)
    reward.set_history(previous,.24);mask=torch.ones(x.shape[:2],dtype=torch.bool)
    value,_=reward(x,torch.zeros_like(mask,dtype=torch.long),mask,.25)
    g,old=torch.autograd.grad(value.sum(),(x,previous),allow_unused=True)
    assert old is None and g.isfinite().all() and g.abs().sum()>0


def test_nonzero_window_start_ignores_unobserved_pre_window_history():
    p,ref=fixture();root=Path(__file__).resolve().parents[1]
    history=read_json(root/p['history_prior_relative_path'])
    ref=copy.deepcopy(ref);history=copy.deepcopy(history)
    ref['times']=[round(t+.1,6) for t in ref['times']];ref['window']=[.1,.6]
    history.update(times=ref['times'],window=ref['window']);p.update(window=ref['window'],geometry_direction_times=ref['times'])
    reward=PersistentEndpointReward(p,ref,history)
    x=torch.tensor(ref['frames'][0]['teacher_endpoint_A'][:1]);mask=torch.ones(x.shape[:2],dtype=torch.bool)
    reward.set_history(x,.09);_,d=reward(x,torch.zeros_like(mask,dtype=torch.long),mask,.1)
    assert not d['history_active'].any() and not d['lineage_increment_penalty'].any()


def test_one_time_history_vjp_certificate_records_two_checks(tmp_path):
    # Nonlinear fixture checks certificate/routing semantics; not a FLOWR GPU test.
    from types import SimpleNamespace
    from evomolsteer.generation.endpoint_controller import EndpointCoordinateExtension,endpoint_pullback
    p,ref=fixture();e=EndpointCoordinateExtension();e.reward=PersistentEndpointReward(p,ref)
    e.out=tmp_path;e.preflight_records=[];e.preflight_forward_calls=0;e.preflight_done=False;e.history_preflight_done=False
    e.model=SimpleNamespace(coord_scale=1.,_lineage=SimpleNamespace(t=0.))
    e.before=torch.tensor(ref['frames'][0]['teacher_endpoint_A'][:1],dtype=torch.float32)
    curr={'mask':torch.ones(e.before.shape[:2],dtype=torch.bool)};com=e.before.new_zeros((1,3))
    def forward(x):
        return {'coords':x+.01*torch.sin(x),'atomics':x.new_ones((*x.shape[:2],1)),
                'affinity':{'pic50':x.square().sum((1,2))}},{'coords':x}
    _,_,g,_,_,atoms,anchor=endpoint_pullback(forward,e.reward,e.before,curr['mask'],1.,com,0.)
    e.preflight(forward,curr,com,g,atoms,anchor)
    e.model._lineage.t=.25;e.reward.set_history(e.before,.24)
    _,_,g,_,_,atoms,anchor=endpoint_pullback(forward,e.reward,e.before,curr['mask'],1.,com,.25)
    e.preflight(forward,curr,com,g,atoms,anchor,history_active=True)
    certificate=read_json(tmp_path/'coordinate_gradient_preflight.json')
    assert certificate['passed'] and certificate['history_audit_completed']
    assert certificate['one_time_extra_forward_calls']==8 and len(certificate['validations'])==2
    assert certificate['production_extra_forward_calls_per_step']==0
