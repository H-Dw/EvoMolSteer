import copy
import numpy as np
import pytest
import torch

from evomolsteer.final_guidance.reward import bridge,HalfStateReward,active_updates
from evomolsteer.final_guidance.workflow import compile_programs,operative_text,validate_analyst,CONTROL_SEMANTICS


def example():
    teacher={'candidate_id':'F1','time':.5,'representation':'current_world_A','active_slots':[0,1,2],
        'coords_A':[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]}
    analyst={'candidate_id':'F1','decision':'propose','hypotheses':[{'id':'h1','atom_slots':[0,1]}]}
    dose={'pointcloud_delta_A':1.,'native_rms_ratio':.33,'max_atom_step_A':.15,'max_cumulative_rms_A':6.}
    design={'candidate_id':'F1','decision':'activate','hypothesis_ids':['h1'],'atom_slots':[0,1],
        'reward_family':'actual_half_state_pointcloud','teacher_time':.5,'dose':dose,'control_semantics':CONTROL_SEMANTICS}
    cfg={'teacher_time':.5,'terminal_time':1.,'guidance_starts':[0.,.1,.3],'guidance_end':.5,'steps':100,'dose':dose}
    return teacher,analyst,design,cfg


def test_exact_bindings_and_three_start_matrix():
    teacher,analyst,design,cfg=example();programs=compile_programs(cfg,analyst,design,teacher,'a','d','t')
    assert len(programs)==6
    assert [len(p['expected_active_steps']) for p in programs]==[50,40,20,50,40,20]
    assert programs[0]['atom_slots']==[0,1] and programs[3]['atom_slots']==[0,1,2]
    assert 'exact slots [0, 1]' in operative_text(programs[0])
    changed=copy.deepcopy(design);changed['atom_slots']=[1,2]
    with pytest.raises(ValueError,match='slots differ'):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    changed=copy.deepcopy(teacher);changed['representation']='predicted_endpoint_world_A'
    with pytest.raises(ValueError):compile_programs(cfg,analyst,design,changed,'a','d','t')


def test_half_clock_bridge_uses_identity_and_real_prediction_derivative():
    x=torch.tensor([[[1.,2.,3.]]],dtype=torch.double,requires_grad=True);endpoint=2*x+1
    z=bridge(x,endpoint,.2,.5);z.sum().backward()
    assert torch.allclose(x.grad,torch.full_like(x,.5/.8+2*.3/.8))
    assert torch.equal(bridge(x,endpoint,.5,.5),x)
    with pytest.raises(ValueError):bridge(x,endpoint,.6,.5)


def test_local_reward_direct_background_zero_and_assignment_not_slot_identity():
    teacher,analyst,design,cfg=example();p=compile_programs(cfg,analyst,design,teacher,'a','d','t')[0]
    reward=HalfStateReward(p,teacher)
    z=torch.tensor([[[2.1,0.,0.],[.1,0.,0.],[0.,3.,0.]]],dtype=torch.double,requires_grad=True);mask=torch.ones((1,3),dtype=torch.bool)
    value,detail=reward.value(z,mask);g=torch.autograd.grad(value.sum(),z)[0]
    assert detail['assignment'].tolist()==[[1,0]]
    assert torch.equal(g[0,2],torch.zeros(3,dtype=g.dtype))
    step=1e-5;direction=torch.tensor([[[1.,.2,0.],[.3,0.,0.],[0.,0.,0.]]],dtype=torch.double)
    plus=reward.value(z.detach()+step*direction,mask,detail['assignment'])[0]
    minus=reward.value(z.detach()-step*direction,mask,detail['assignment'])[0]
    assert torch.allclose((plus-minus)/(2*step),(g*direction).sum((1,2)),atol=1e-8)


def test_clocks_are_dynamic_and_invalid_starts_fail():
    teacher,analyst,design,cfg=example();teacher['time']=.6;design['teacher_time']=.6;cfg.update(teacher_time=.6,guidance_end=.6,guidance_starts=[.2])
    programs=compile_programs(cfg,analyst,design,teacher,'a','d','t')
    assert len(programs)==2 and len(programs[0]['expected_active_steps'])==40
    cfg['guidance_starts']=[.6]
    with pytest.raises(ValueError):compile_programs(cfg,analyst,design,teacher,'a','d','t')


def test_native_adapter_keeps_the_native_update_and_uses_control_once():
    from evomolsteer.final_guidance.native import instrument
    source='''def _generate(self):
    curr = 0
    pocket_data = times = cond = pocket_equis = pocket_invs = None
    step_sizes = [1, 1]
    for i, step_size in enumerate(step_sizes):
        out = self(curr, pocket_data, times, training=False)
        curr = curr + out
    return curr
'''
    events=[]
    class Control:
        def forward(self,curr,pocket,times,cond,equis,invs,i,dt):events.append(('forward',i));return 2
        def after(self,curr,i):events.append(('after',i));return curr+1
    class Model:_half_guidance=Control()
    scope={};exec(instrument(source),scope)
    assert scope['_generate'](Model())==6
    assert events==[('forward',0),('after',0),('forward',1),('after',1)]
    with pytest.raises(ValueError):instrument('def _generate(self):\n    return 1')


def test_compiler_rejects_parameter_and_candidate_drift():
    teacher,analyst,design,cfg=example()
    changed=copy.deepcopy(design);changed['dose']['native_rms_ratio']=.5
    with pytest.raises(ValueError):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    changed=copy.deepcopy(design);changed['candidate_id']='F2'
    with pytest.raises(ValueError):compile_programs(cfg,analyst,changed,teacher,'a','d','t')
    changed=copy.deepcopy(design);changed['control_semantics']['optimization']='reward_descent'
    with pytest.raises(ValueError):compile_programs(cfg,analyst,changed,teacher,'a','d','t')


def test_measurement_and_selected_slots_share_compact_support():
    from evomolsteer.final_guidance.evidence import compact_regions
    x=np.array([[[.5,.1,0.],[1.,.3,0.],[10.,10.,10.]]])
    a=compact_regions(x,[[0.,0.,0.]],2.)
    changed=x.copy();changed[0,2]=[30.,-30.,25.]
    assert np.array_equal(a,compact_regions(changed,[[0.,0.,0.]],2.))
    empty=compact_regions(x,[[50.,50.,50.]],2.)
    assert empty[0,0]==0 and np.isnan(empty[0,1:]).all()


def test_analyst_cannot_activate_undefined_or_misnamed_evidence():
    import jsonschema
    packet={'candidate':{'candidate_id':'F1'},'regions':[{'region_id':'R00','local_usable':True,'atom_slots':[0]}],
        'one_to_rest':[{'evidence_id':'F1/R00/4','region_id':'R00','observable':'cov_xx','candidate_value':0.,'candidate_minus_rest':-.2}]}
    answer={'candidate_id':'F1','decision':'propose','hypotheses':[{'id':'h1','region_id':'R00','atom_slots':[0],
        'evidence_id':'F1/R00/4','observable':'cov_xx','claim_kind':'axis_dispersion'}]}
    with pytest.raises(jsonschema.ValidationError,match='one-point covariance'):validate_analyst(answer,packet)
    packet['regions'][0]['atom_slots']=[0,1];answer['hypotheses'][0]['atom_slots']=[0,1]
    validate_analyst(answer,packet)
    packet['one_to_rest'][0]['candidate_value']=None
    with pytest.raises(jsonschema.ValidationError,match='Unavailable'):validate_analyst(answer,packet)


def test_control_uses_generator_vjp_and_does_not_differentiate_affinity():
    from types import SimpleNamespace
    from evomolsteer.final_guidance.native import Control
    teacher,analyst,design,cfg=example();p=compile_programs(cfg,analyst,design,teacher,'a','d','t')[0]
    class Model:
        coord_scale=1.
        integrator=SimpleNamespace(use_cosine_scheduler=False)
        calls=0
        def __call__(self,state,pocket,times,**kwargs):
            self.calls+=1;x=state['coords']
            return {'coords':2*x+x.mean(1,keepdim=True),'affinity':{'pic50':x.sum((1,2))}}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    x=torch.tensor([[[.1,.2,.3],[1.5,.3,.2],[0.,2.,.1]]],dtype=torch.double)
    curr={'coords':x,'mask':torch.ones((1,3),dtype=torch.bool)}
    model=Model();control=Control(model,p,teacher,torch.zeros((1,3),dtype=torch.double))
    with torch.no_grad():control.forward(curr,None,[torch.tensor([.2])]*3,None,None,None,20,.01)
    actual,_,_,direct,_,_=control.cached
    leaf=x.clone().requires_grad_(True);z=bridge(leaf,2*leaf+leaf.mean(1,keepdim=True),float(torch.tensor(.2)),.5)
    expected=torch.autograd.grad(control.reward.value(z,curr['mask'])[0].sum(),leaf)[0]
    assert torch.allclose(actual,expected,atol=1e-12) and model.calls==1
    assert actual.masked_fill(direct[...,None],0).abs().max()>0
    control.after(curr,20)
    assert control.rows[0]['direct_unmatched_gradient_max']==0 and control.rows[0]['nonzero_molecules']==1
