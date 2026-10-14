import torch
from types import SimpleNamespace
from evomolsteer.partial_search.two_clock import TwoClockReward,TwoClockControl
from evomolsteer.final_guidance.reward import bridge


def example():
    teacher={'candidate_id':'F1','time':.5,'representation':'current_world_A','active_slots':[0,1,2],
        'coords_A':[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],'terminal_representation':'terminal_world_A',
        'terminal_coords_A':[[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]}
    program={'candidate_id':'F1','teacher_time':.5,'family':'actual_partial_two_clock','scope':'local','atom_slots':[0,1],
        'atom_weights':[.5,.5],'endpoint_mix':.5,'window':[0.,.5],
        'dose':{'pointcloud_delta_A':1.,'native_rms_ratio':.33,'max_atom_step_A':.15,'max_cumulative_rms_A':6.}}
    return teacher,program


def test_two_view_loss_uses_one_correspondence_and_has_correct_derivative():
    teacher,p=example();reward=TwoClockReward(p,teacher)
    x=torch.tensor([[[.1,.2,.1],[1.4,.3,0.],[0.,3.,.2]]],dtype=torch.double,requires_grad=True)
    y=2*x+x.mean(1,keepdim=True);z=bridge(x,y,.2,.5);mask=torch.ones((1,3),dtype=torch.bool)
    value,detail=reward.value_two(z,y,mask);gradient=torch.autograd.grad(value.sum(),x)[0]
    direction=torch.ones_like(x);epsilon=1e-5
    def value_at(a):
        b=2*a+a.mean(1,keepdim=True)
        return reward.value_two(bridge(a,b,.2,.5),b,mask,detail['assignment'])[0]
    difference=(value_at(x.detach()+epsilon*direction)-value_at(x.detach()-epsilon*direction))/(2*epsilon)
    assert torch.allclose(difference,(gradient*direction).sum((1,2)),atol=1e-8)


def test_production_transfer_control_keeps_real_model_pullback_and_one_forward():
    teacher,p=example()
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False);calls=0
        def __call__(self,state,pocket,times,**kwargs):
            self.calls+=1;x=state['coords'];return {'coords':2*x+x.mean(1,keepdim=True),'affinity':{'pic50':x.sum((1,2))}}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    model=Model();control=TwoClockControl(model,p,teacher,torch.zeros((1,3),dtype=torch.double))
    x=torch.tensor([[[.1,.2,.1],[1.4,.3,0.],[0.,3.,.2]]],dtype=torch.double);mask=torch.ones((1,3),dtype=torch.bool)
    with torch.no_grad():control.forward({'coords':x,'mask':mask},None,[torch.tensor([.2])]*3,None,None,None,20,.01)
    leaf=x.clone().requires_grad_(True);endpoint=2*leaf+leaf.mean(1,keepdim=True)
    value=control.reward.value_two(bridge(leaf,endpoint,float(torch.tensor(.2)),.5),endpoint,mask)[0]
    expected=torch.autograd.grad(value.sum(),leaf)[0]
    assert torch.allclose(control.cached[0],expected,atol=1e-12) and model.calls==1
    control.after({'coords':x,'mask':mask},20)
    assert control.rows[0]['direct_unmatched_gradient_max']==0 and control.rows[0]['endpoint_direct_unmatched_gradient_max']==0
