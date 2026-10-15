import copy
import numpy as np
import pytest
import torch
from evomolsteer.nc_guidance.autoresearch import validate_program,execution_identity,NATIVE_DEFAULTS
from evomolsteer.nc_guidance.window_features import fit_window
from evomolsteer.nc_guidance.reward import NCReward
from test_nc_guidance import example
from evomolsteer.final_guidance.native import instrument_coord_telemetry


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


def test_noise_observer_preserves_output_and_rng_consumption():
    source='''def coord_step(self, curr_coords, pred_coords, mask, times, step_size):
    coord_velocity=(pred_coords-curr_coords)/(1-times[:,None,None])
    gt=1/(times[0]+.01)
    score=(times[:,None,None]*coord_velocity-curr_coords)/(1-times[:,None,None]+1e-6)
    eps=torch.randn(curr_coords.shape)
    std_eps=torch.sqrt(2*gt*self.coord_noise_level)
    coords=curr_coords+(coord_velocity+gt*score)*step_size+eps*std_eps*step_size
    return coords*mask[...,None]
'''
    from types import SimpleNamespace
    original={'torch':torch};observed={'torch':torch}
    exec(source,original);exec(instrument_coord_telemetry(source),observed)
    model=SimpleNamespace(use_sde_simulation=True,coord_strategy='continuous',coord_noise_level=.2)
    x=torch.ones((2,6,3));y=x*.8;mask=torch.ones((2,6));times=torch.tensor([.3,.3])
    torch.manual_seed(42);a=original['coord_step'](model,x,y,mask,times,.01);ra=torch.get_rng_state()
    torch.manual_seed(42);b=observed['coord_step'](model,x,y,mask,times,.01);rb=torch.get_rng_state()
    assert torch.equal(a,b) and torch.equal(ra,rb)
    assert set(model._guidance_components)=={'predictive','score_drift','random'}


def test_type_confidence_weighting_relaxes_saturated_coordinate_pressure():
    from types import SimpleNamespace
    from evomolsteer.nc_guidance.reward import NCControl
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False)
        def __call__(self,curr,pocket,times,**kw):
            x=curr['coords'];bias=torch.zeros((1,6,5),dtype=x.dtype);bias[0,0,4]=8;bias[0,3,3]=8
            column=x.new_tensor([0.,0.,0.,-1.,1.])
            return {'coords':x+.05,'atomics':bias+x[...,0,None]*column*.01}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    p,t=resolved();curr={'coords':torch.tensor([t['coords_A']],dtype=torch.double),'mask':torch.ones((1,6),dtype=torch.bool)}
    values=[]
    for mode in ('none','linear_deficit'):
        q=copy.deepcopy(p);q['type_confidence_weighting']=mode
        c=NCControl(Model(),q,t,torch.zeros((1,3),dtype=torch.double))
        c.forward(curr,{},[torch.tensor([.1])]*3,None,None,None,10,.01)
        values.append(c.extra['type_mixture_input_rms'])
    assert 0<values[1]<values[0]/100
