import copy
import numpy as np
import pytest
import torch
from evomolsteer.nc_guidance.autoresearch import validate_program,execution_identity,NATIVE_DEFAULTS
from evomolsteer.nc_guidance.window_features import fit_window
from evomolsteer.nc_guidance.reward import NCReward
from test_nc_guidance import example


def resolved():
    p,t=example();p.update(teacher_time=.5,steps=100,native_options=copy.deepcopy(NATIVE_DEFAULTS),geometry_mode='aggregate')
    return p,t


def test_dynamic_clock_is_propagated_and_out_of_window_rejected():
    p,t=resolved();t['time']=.6;p.update(teacher_time=.6,steps=60,window=[.2,.6])
    assert len(validate_program(p,t,[0,.6])['expected_active_steps'])==24
    p['teacher_time']=.5
    with pytest.raises(ValueError,match='clock|time'):validate_program(p,t,[0,.6])


def test_unknown_native_option_is_not_silently_ignored():
    p,t=resolved();p['native_options']['fictional_parameter']=1
    with pytest.raises(ValueError,match='native option'):validate_program(p,t,[0,.5])


def test_cache_changes_with_native_dynamics_source_seed_and_initial_batch():
    p,t=resolved();cfg={'pilot_initial_batch':0,'pilot_samples':24,'seed':42}
    base=execution_identity(p,t,cfg,{'code':'a','initial':'x'})
    for new in [{**cfg,'seed':43},{**cfg,'pilot_initial_batch':2}]:assert execution_identity(p,t,new,{'code':'a','initial':'x'})!=base
    assert execution_identity(p,t,cfg,{'code':'b','initial':'x'})!=base
    p['native_options']['cat_noise_level']=0
    assert execution_identity(p,t,cfg,{'code':'a','initial':'x'})!=base


def test_pointwise_curvature_changes_direction_while_aggregate_cancels():
    p,t=resolved();mask=torch.ones((1,6),dtype=torch.bool)
    x=torch.tensor([t['coords_A']],dtype=torch.double)+torch.tensor([[[.1,0,0],[.4,.1,0],[.05,0,0],[1.7,.5,0],[0,0,0],[0,0,0]]],dtype=torch.double)
    logits=torch.zeros((1,6,5),dtype=torch.double);directions={}
    for mode in ('aggregate','pointwise'):
        for delta in (.2,2.):
            z=x.clone().requires_grad_(True);p.update(geometry_mode=mode);p['dose']['pointcloud_delta_A']=delta
            reward=NCReward(p,t);term,detail=reward.terms(z,z,logits,mask)
            grad=torch.autograd.grad(term['half'].sum(),z)[0];directions[mode,delta]=grad/grad.norm()
    assert torch.allclose(directions['aggregate',.2],directions['aggregate',2.],atol=1e-10)
    assert (directions['pointwise',.2]-directions['pointwise',2.]).abs().max()>.01


def test_whole_window_fit_reports_signed_rate_without_time_point_pvalues():
    t=np.linspace(0,.6,61);y=2+3*t-2*t*t
    fits=fit_window(t,y);assert fits[1]['rmse']<1e-12
    assert np.allclose(fits[1]['coefficients_ascending'],[2,3,-2])
    assert fits[1]['p_value'] is None and fits[1]['domain']==[0.,.6]
