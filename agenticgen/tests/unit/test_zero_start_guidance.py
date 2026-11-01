"""Six-window admission, zero scales and native raw999 endpoints on CPU."""
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from agenticgen.uniguide import guidance_windows as bounds, linear_scale as scale
from agenticgen.uniguide import linear_calibration as calibration, absolute_coordinate as coordinates
from agenticgen.uniguide import guidance_endpoints as endpoints
from agenticgen.uniguide.geometry_sampling import GeometryControl, scale_binding
from agenticgen.uniguide.geometry_batch import CandidateControl
from agenticgen.uniguide.geometry_dynamics import policy as dynamics
from agenticgen.uniguide.absolute_runtime import validate_config
from agenticgen.uniguide.tool_runtime import configure, assets
from agenticgen.uniguide.io import read, write, Budget
from test_absolute_coordinate_protocol import condition
from test_dynamic_matching_jump import dynamic_condition


def config(root):
    folder=Path(__file__).resolve().parents[2]/'configs'
    c=read(folder/'uniguide_absolute_coordinate.json')
    c.update(read(folder/'uniguide_six_windows_zero_start.json'))
    c.update(root=str(root),generation_context_root=str(root.parent/'frozen'))
    return validate_config(c)


def test_six_window_overlay_and_exact_closed_boundaries(tmp_path):
    c=config(tmp_path/'new')
    assert c['agent_tools']['skill_release']=='2.10.0' and len(assets(c))==4
    assert c['guidance_control']['maximum_attempts_per_teacher']==500
    assert c['guidance_control']['initial_scales'][0]==0
    assert c['guidance_control']['search_minimum']==0
    strengths=[10,20,30,40,50,60]
    actual=bounds.expand(strengths,c)
    for value,(lo,hi) in zip(strengths,bounds.SIX):
        assert actual[lo:hi+1]==[value]*(hi-lo+1)
    assert len(actual)==100 and actual[0]==actual[10]==10 and actual[11]==20 and actual[99]==60
    assert scale_binding([0]*6,window_bounds=bounds.windows(c))[scale.EFFECTIVE]==[0]*6
    for bad in (dict(c,guidance_endpoints=None),dict(c,prefix_inpainting={'enabled':True}),
                dict(c,agent_tools=dict(c['agent_tools'],skill_release='2.9.0'))):
        with pytest.raises(ValueError):validate_config(bad)


def test_endpoints_only_mask_extra_guidance_and_keep_old_mode(tmp_path):
    c=config(tmp_path/'new')
    unit=object.__new__(GeometryControl);unit.absolute_config=c;unit.schedule=bounds.expand([10]*6,c)
    for step in (0,99):assert unit.step_scales(step,10)==[0.]*10
    for step in (1,10,11,98):assert unit.step_scales(step,10)==[10]*10
    mixed=object.__new__(CandidateControl);mixed.absolute_config=c
    mixed.units=[{},{}];mixed.failed_units={};mixed.expanded=[unit.schedule,bounds.expand([20]*6,c)]
    assert mixed.step_scales(0,20)==[0.]*20 and mixed.step_scales(99,20)==[0.]*20
    assert mixed.step_scales(10,20)==[10]*10+[20]*10
    old=dict(c);old.pop('guidance_endpoints');old['windows']=bounds.DEFAULT
    assert endpoints.apply([3]*10,99,old)==[3]*10
    assert dynamics(c)['guidance_endpoints']['gradient_update_steps']==[1,98]


def test_zero_assessment_and_zero_reference_need_no_logarithm(tmp_path):
    from test_linear_guidance import row
    from agenticgen.uniguide.calibration_search import assess as legacy_assess
    c=config(tmp_path/'new');rows=[]
    for value,error in [(0,.1),(10,.2),(20,.3)]:
        r=row(value,error=error);r[scale.EFFECTIVE]=[value]*6
        r['window_diagnostics']=(r['window_diagnostics']*2)[:6];rows.append(r)
    usable,curve,_=calibration.assess(rows)
    assert len(usable)==3 and curve[1]['growth_comparison']=='zero_reference_has_no_multiplicative_ratio'
    with pytest.raises(ValueError):legacy_assess(rows)
    d=calibration.derive_domain(rows,'teacher','teachercondition','sampling',c['guidance_control'],c['guidance_exploration'],bounds.SIX)
    assert d[scale.REFERENCE]==0 and d['initial_source']==rows[0]['id']
    assert scale.effective([0]*6,d)==[0]*6 and scale.effective([1]*6,d)==[10]*6
    assert scale.encode([0,10,20,0,30,0],d)==[0,1,2,0,3,0]
    failed=[dict(rows[0],complete=False),*rows[1:]]
    remaining,curve,boundary=calibration.assess(failed)
    assert remaining==rows[1:] and boundary is None
    assert curve[0]['status']=='nonfinite_measurement'


def test_calibration_starts_zero_and_can_spend_500_once(tmp_path,monkeypatch):
    import test_linear_guidance as helpers
    original=helpers.row
    def six_row(*args,**kwargs):
        result=original(*args,**kwargs)
        result['window_diagnostics']=(result['window_diagnostics']*2)[:6]
        return result
    monkeypatch.setattr(helpers,'row',six_row)
    executor,calls,_=helpers.install_fake(monkeypatch,tmp_path,
        errors=lambda a,n:.1+(a[0]-(20 if n<50 else 180))**2/100000)
    c=config(tmp_path)
    subjects=[dict(teacher=dict(target='teacher'),condition=dict(condition_sha256='teachercondition'))]
    result=calibration.run(c,subjects,lambda *a:None,executor)
    assert calls[0][1]==[0]*6 and calls[0][2]==10
    assert 350 < calibration.used(tmp_path,'teacher') <=500
    assert read(tmp_path/'ATTEMPT_LIMITS.json')['per_teacher_stage_caps']['calibration']==500
    before=list(calls)
    assert calibration.run(c,subjects,lambda *a:pytest.fail('reloaded'),executor)==result and calls==before
    assert len({(tuple(r[1]),r[2]) for r in calls})==len(calls)
    c['guidance_control']['maximum_attempts_per_teacher']=501
    with pytest.raises(ValueError):configure(c)


def test_teacher_and_global_caps_use_declared_500(tmp_path):
    c=config(tmp_path);calibration.ensure_limits(c,2)
    assert read(tmp_path/'ATTEMPT_LIMITS.json')['stage_caps']['calibration']==1000
    write(tmp_path/'BUDGET_SCOPE.json',dict(stage='calibration',teacher_id='teacher'))
    Budget(tmp_path).reserve('zero','calibration',500,'zero')
    with pytest.raises(ValueError,match='Teacher'):
        Budget(tmp_path).reserve('overflow','calibration',10,'overflow')
    old=configure(dict(root=str(tmp_path/'old'),protocol=coordinates.POLICY,rounds=5,
                       agent_tools=dict(skill_release='2.9.0')))
    calibration.ensure_limits(old,1)
    assert read(tmp_path/'old/ATTEMPT_LIMITS.json')['per_teacher_stage_caps']['calibration']==350


def test_actual_forward_skips_vjp_at_endpoints_and_zero_scale(dynamic_condition,tmp_path):
    c=config(tmp_path/'new');c['chemical_inpainting']={'enabled':False};calls=[]
    class Model:
        coord_scale=1.
        integrator=SimpleNamespace(noise_scheduler=lambda t:t.new_tensor(0.),coord_noise_level=.2)
        def __call__(self,current,*a,**kw):
            calls.append((torch.is_grad_enabled(),current['coords'].requires_grad))
            return dict(coords=current['coords']*1.01+.01)
        def _get_predictions(self,out):return out,None
    control=object.__new__(GeometryControl)
    control.model=Model();control.geometry_condition=dynamic_condition;control.absolute_config=c
    control.scale_binding=scale_binding([1.]*6,window_bounds=bounds.windows(c))
    control.schedule=control.expand_schedule([1.]*6);control.sampling_policy=dynamics(c)
    control.prefix_policy={'enabled':False};control.com=torch.zeros(10,3);control.rows=[]
    control.resume=None;control.folder=tmp_path;control.last_assignment=None
    control.random_streams=SimpleNamespace(states=[])
    control.absolute_state=coordinates.AssignmentState(control)
    stored={};control.store=lambda key,step,value:stored.__setitem__((key,step),value.detach().clone() if torch.is_tensor(value) else value)
    curr=dict(coords=torch.tensor([[[0.,0.,0.],[1.4,0.,0.],[9.,9.,9.]]]).repeat(10,1,1),mask=torch.ones(10,3,dtype=torch.bool))
    before=torch.get_rng_state().clone()
    for step in range(100):
        control.forward(curr,{},[torch.full((10,),step*.00999)]*3,None,None,None,step,.00999)
        assert calls[-1]==((False,False) if step in (0,99) else (True,True))
        if step in (0,99):
            assert not torch.count_nonzero(control.delta) and not torch.count_nonzero(stored['input_gradient',step])
            assert control.rows[-1]['applied_guidance_scales']==[0.]*10
        else:
            assert torch.count_nonzero(control.delta) and torch.count_nonzero(stored['input_gradient',step])
    control.schedule=[0.]*100
    control.forward(curr,{},[torch.full((10,),.1)]*3,None,None,None,10,.00999)
    assert calls[-1]==(False,False) and not torch.count_nonzero(control.delta)
    assert control.absolute_state.rows[0]['first_step']==0
    assert len(calls)==101
    assert torch.equal(before,torch.get_rng_state())


def test_assignment_trace_checks_the_new_prefix(dynamic_condition,tmp_path):
    c=config(tmp_path/'new');write(tmp_path/'OUTPUT_POLICY.json',dynamics(c))
    write(tmp_path/'steps.json',[dict(step=i) for i in range(12)])
    points=np.array([[[0.,0.,0.],[1.4,0.,0.],[9.,9.,9.]]],dtype=np.float32)
    with h5py.File(tmp_path/'trace.h5','w') as h:
        h['endpoint_A']=np.repeat(points[None],12,axis=0)
        h['mask']=np.ones((12,1,3),dtype=bool)
        h['assignment']=np.tile(np.array([0,1]),(12,1,1))
    assert coordinates.verify_assignment_trace(tmp_path,dynamic_condition)['completed_active_steps']==list(range(12))
    with h5py.File(tmp_path/'trace.h5','r+') as h:h['assignment'][0]=[[1,0]]
    with pytest.raises(ValueError,match='Hungarian'):coordinates.verify_assignment_trace(tmp_path,dynamic_condition)


def test_terminal_diagnostics_use_initial_zero_and_last_gradient_98(dynamic_condition,tmp_path):
    from agenticgen.uniguide.geometry_sampling import terminal_records
    write(tmp_path/'OUTPUT_POLICY.json',dynamics(config(tmp_path/'new')))
    with h5py.File(tmp_path/'trace.h5','w') as trace:
        assignments=np.tile([1,2],(100,10,1))
        assignments[0]=[0,1];assignments[98]=[0,2]
        trace['assignment']=assignments
    x=torch.tensor([[[0.,0.,0.],[0.,2.,0.],[1.4,0.,0.]]]).repeat(10,1,1)
    output=dict(coords=x,mask=torch.ones(10,3,dtype=torch.bool))
    model=SimpleNamespace(_generate_mols=lambda output,sanitise=True:[None]*10)
    rows=terminal_records(model,output,dynamic_condition,tmp_path,10)
    assert all(r[coordinates.RMSD]==0 for r in rows)
    receipt=read(tmp_path/'COORDINATE_ASSIGNMENT.json')
    assert receipt['first_observed_assignment_step']==0 and receipt['last_gradient_assignment_step']==98
    assert receipt['initial_assignment']==[[0,1]]*10 and receipt['last_guidance_assignment']==[[0,2]]*10
    assert receipt['terminal_to_last_guidance_switch_fraction']==[0.]*10
