import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from evomolsteer.nc_guidance.reward import NCReward,NCControl
from evomolsteer.nc_guidance.workflow import compile_program,numerical_signature


def example():
    xyz=[[0.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[0.,3.,0.],[4.,3.,0.],[7.,7.,0.]]
    teacher={'candidate_id':'F1','coords_A':xyz,'terminal_coords_A':xyz,'time':.5,'counterfactual_labels':False,
        'atom_vocabulary':{'C':3,'N':4},'active_slots':list(range(6))}
    anchor={'slot':0,'neighbor_slots':[1,2],'donor':{'coords_A':[0.,3.,0.],'outward':[0.,-1.,0.]}}
    p={'family':'local_nc_layout','scope':'local','candidate_id':'F1','atom_slots':[0,1,2,3],
        'typed_slots':[0,3],'typed_elements':['N','C'],'anchors':[anchor],'max_local_points':4,
        'window':[.1,.5],'weights':{'half':.3,'endpoint':.4,'type':.3,'orientation':0.},
        'dose':{'pointcloud_delta_A':1.,'native_rms_ratio':.33,'max_atom_step_A':.15,'max_cumulative_rms_A':6.},
        'type_mode':'full_ce','gradient_balance':'unit_rms','assignment_endpoint_fraction':1.,
        'assignment_type_cost':0.,'type_ramp':False,'anchor_distance_A':3.,'anchor_width_A':.4}
    return p,teacher


def test_nc_loss_has_exact_zero_direct_gradient_outside_local_coordinates_and_logits():
    p,t=example();r=NCReward(p,t);z=torch.tensor([t['coords_A']],dtype=torch.double,requires_grad=True)
    y=(z.detach()+.05).requires_grad_(True);logits=torch.zeros((1,6,5),dtype=torch.double,requires_grad=True);mask=torch.ones((1,6),dtype=torch.bool)
    terms,info=r.terms(z,y,logits,mask);value=sum(p['weights'].get(k,0)*v for k,v in terms.items())
    gz,gy,gl=torch.autograd.grad(value.sum(),(z,y,logits));unmatched=~info['direct_mask']
    assert gz[unmatched].abs().max()==0 and gy[unmatched].abs().max()==0 and gl[unmatched].abs().max()==0
    assert gl.abs().max()>0 and info['direct_mask'].sum()==4


def test_type_target_and_counterfactual_labels_change_reward_without_moving_teacher():
    p,t=example();z=torch.tensor([t['coords_A']],dtype=torch.double);mask=torch.ones((1,6),dtype=torch.bool)
    logits=torch.zeros((1,6,5),dtype=torch.double);logits[0,0,4]=4;logits[0,3,3]=4
    r=NCReward(p,t);v,a=r.terms(z,z,logits,mask);swapped=copy.deepcopy(p);swapped['typed_elements']=['C','N']
    other=NCReward(swapped,t).terms(z,z,logits,mask,a['assignment'])[0]
    assert v['type']>other['type']+3 and torch.equal(v['endpoint'],other['endpoint'])


def test_selected_logits_derivative_matches_fixed_assignment_difference():
    p,t=example();r=NCReward(p,t);z=torch.tensor([t['coords_A']],dtype=torch.double);mask=torch.ones((1,6),dtype=torch.bool)
    logits=torch.randn((1,6,5),dtype=torch.double,requires_grad=True);terms,info=r.terms(z,z,logits,mask)
    grad=torch.autograd.grad(terms['type'].sum(),logits)[0];direction=torch.arange(30,dtype=torch.double).reshape_as(logits)/30;h=1e-5
    fd=(r.terms(z,z,logits.detach()+h*direction,mask,info['assignment'])[0]['type']-r.terms(z,z,logits.detach()-h*direction,mask,info['assignment'])[0]['type'])/(2*h)
    assert torch.allclose(fd,(grad*direction).sum((1,2)),atol=1e-8)


def test_full_cloud_and_unbound_orientation_are_rejected():
    p,t=example();p['atom_slots']=list(range(6));p['max_local_points']=6
    with pytest.raises(ValueError,match='proper local'):NCReward(p,t)
    p,t=example();p['anchors'][0]['neighbor_slots']=[1,5]
    with pytest.raises(ValueError,match='outside local'):NCReward(p,t)


def test_coordinate_guidance_receives_real_atomic_decoder_jacobian_without_affinity_gradient():
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False)
        def __call__(self,curr,pocket,times,**kw):
            x=curr['coords'];column=torch.tensor([0.,0.,0.,-1.,1.],dtype=x.dtype)
            return {'coords':x+.05,'atomics':x[...,0,None]*column,'affinity':{'pic50':x.sum((1,2))}}
        def _get_predictions(self,out):return {'coords':out['coords'],'atomics':out['atomics'].softmax(-1)},None
    p,t=example();p['weights']={'half':0.,'endpoint':0.,'type':1.,'orientation':0.}
    control=NCControl(Model(),p,t,torch.zeros((1,3),dtype=torch.double))
    curr={'coords':torch.tensor([t['coords_A']],dtype=torch.double),'mask':torch.ones((1,6),dtype=torch.bool)}
    control.forward(curr,{},[torch.tensor([.1])]*3,None,None,None,10,.01)
    assert control.extra['type_gradient_rms']>0 and control.extra['atomic_output_in_coordinate_vjp']
    before=curr['coords'].clone();after=control.after(curr,10)
    assert (after['coords']-before).abs().max()>0 and control.rows[-1]['nonzero_molecules']==1
    assert not control.extra['affinity_vjp']


def test_designer_cannot_detach_patch_or_change_enabled_terms():
    p,t=example();contract={k:p[k] for k in ['window','weights','dose','type_mode','gradient_balance','assignment_endpoint_fraction','assignment_type_cost','type_ramp','anchor_distance_A','anchor_width_A']}
    cfg={'execution_contract':contract,'round_id':'R01','max_local_points':4,'teacher_time':.5,'steps':100}
    patch={k:p[k] for k in ('atom_slots','typed_slots','typed_elements','anchors')};patch.update(patch_id='L1',template_id='T1',evidence_id='L1/finite_layout')
    a={'patch_id':'L1','evidence_id':'L1/finite_layout'};d={'patch_id':'L1','decision':'activate','execution_contract':contract}
    assert compile_program(cfg,patch,t,a,d,'t','a','d')['atom_slots']==p['atom_slots']
    d['patch_id']='L2'
    with pytest.raises(ValueError,match='detached'):compile_program(cfg,patch,t,a,d,'t','a','d')


def test_local_role_loss_touches_only_mapped_charge_and_neighbor_bond_logits():
    p,t=example();p['weights']['role']=.15;t.update(charge_vocabulary={'0':1},local_adjacency={'0':[1,2]})
    r=NCReward(p,t);z=torch.tensor([t['coords_A']],dtype=torch.double);mask=torch.ones((1,6),dtype=torch.bool)
    logits=torch.zeros((1,6,5),dtype=torch.double);bonds=torch.randn((1,6,6,5),dtype=torch.double,requires_grad=True)
    charges=torch.randn((1,6,3),dtype=torch.double,requires_grad=True)
    terms,detail=r.terms(z,z,logits,mask,decoder={'bonds':bonds,'charges':charges})
    gb,gc=torch.autograd.grad(terms['role'].sum(),(bonds,charges));ids=detail['assignment'][0]
    assert gc[0,ids[0]].abs().max()>0 and gb[0,ids[0],ids[1]].abs().max()>0
    assert gc[0,ids[3]].abs().max()==0 and gb[0,ids[3]].abs().max()==0


def test_numeric_identity_excludes_cap_metadata_but_tracks_actual_target_labels():
    p,t=example();other=copy.deepcopy(p);other['max_local_points']=5
    assert numerical_signature(p,t)==numerical_signature(other,t)
    other['typed_elements']=['C','N']
    assert numerical_signature(p,t)!=numerical_signature(other,t)


def test_local_decoder_logit_ascent_changes_only_typed_predictions():
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False)
        def __call__(self,curr,pocket,times,**kw):
            x=curr['coords'];column=torch.tensor([0.,0.,0.,-1.,1.],dtype=x.dtype)
            return {'coords':x+.05,'atomics':x[...,0,None]*column,'affinity':{'pic50':x.sum((1,2))}}
        def _get_predictions(self,out):return {'coords':out['coords'],'atomics':out['atomics'].softmax(-1)},None
    p,t=example();p.update(atomic_logit_strength=8.,logit_max_shift=4.)
    control=NCControl(Model(),p,t,torch.zeros((1,3),dtype=torch.double))
    curr={'coords':torch.tensor([t['coords_A']],dtype=torch.double),'mask':torch.ones((1,6),dtype=torch.bool)}
    original=control.model(curr,{},[])['atomics'];changed=control.forward(curr,{},[torch.tensor([.1])]*3,None,None,None,10,.01)['atomics']
    touched=(changed-original).abs().sum(-1)>0
    assert touched.sum()==2 and control.extra['unmatched_logit_change_max']==0
    assert control.extra['typed_probability_after_logit_step']>control.extra['typed_correct_probability']
    assert (changed-original).abs().max()<=4. and torch.equal(curr['coords'],torch.tensor([t['coords_A']],dtype=torch.double))
