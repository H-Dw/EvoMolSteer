import copy
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from evomolsteer.local_motif.reward import MotifReward,MotifControl,OPERATORS
from evomolsteer.local_motif.semantic_contract import identity,references
from evomolsteer.local_motif.conditional_layout import contrasts
from evomolsteer.local_motif.suffix_probe import GroupedSuffix
from test_nc_guidance import example


def inputs():
    old,t=example();t['native_elements']=['C','C','C','C','C','C'];t['coords_A'][2]=[0.,1.,0.]
    p={'family':'local_motif_v2','operator':'point_band','representation':'bridge','atom_slots':[0,1,2],
       'teacher_time':.5,'window':[.1,.5],'coordinate_updates_only':True,'smc':False,'affinity_head_gradient':False,
       'base_band_A':.1,'band_multiplier':1.,'orientation_triplets':[[0,1,2]],'pocket_sites':[{'coords_A':[0,4.,2.]}],
       'orientation_band':.01,'controller':'unit_rms','gradient_floor_multiplier':1.,'assignment_element_cost':0.,
       'dose':old['dose'],'steps':100,'native_options':{}}
    return p,t


@pytest.mark.parametrize('operator',OPERATORS)
def test_derivative_and_sparse_support(operator):
    p,t=inputs();p['operator']=operator;r=MotifReward(p,t)
    z=torch.tensor([t['coords_A']],dtype=torch.double)+torch.arange(18,dtype=torch.double).reshape(1,6,3)*.03
    z.requires_grad_(True);mask=torch.ones((1,6),dtype=torch.bool);logits=torch.zeros((1,6,5),dtype=torch.double)
    value,info=r.evaluate(z,logits,mask);ids=info['assignment'];grad=torch.autograd.grad(value.sum(),z)[0]
    d=torch.linspace(-.5,.6,18,dtype=torch.double).reshape_as(z);h=1e-6
    f=lambda x:r.evaluate(x,logits,mask,ids)[0].sum()
    fd=(f(z.detach()+h*d)-f(z.detach()-h*d))/(2*h)
    assert torch.allclose(fd,(grad*d).sum(),atol=2e-8,rtol=1e-5)
    assert grad[~info['direct_mask']].abs().max()==0


def test_attained_band_has_zero_gradient_and_accepts_no_NC_labels():
    p,t=inputs();z=torch.tensor([t['coords_A']],dtype=torch.double,requires_grad=True)
    v,_=MotifReward(p,t).evaluate(z,torch.zeros((1,6,5),dtype=torch.double),torch.ones((1,6),dtype=torch.bool))
    assert v.item()==0 and torch.autograd.grad(v.sum(),z)[0].abs().max()==0


@pytest.mark.parametrize('operator',OPERATORS)
def test_common_rigid_transform_preserves_reward(operator):
    p,t=inputs();p['operator']=operator;z=torch.tensor([t['coords_A']],dtype=torch.double)+.3
    logits=torch.zeros((1,6,5),dtype=torch.double);mask=torch.ones((1,6),dtype=torch.bool)
    v=MotifReward(p,t).evaluate(z,logits,mask)[0]
    Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    offset=np.array([2.,3.,4.]);tt=copy.deepcopy(t);pp=copy.deepcopy(p)
    for k in ['coords_A','terminal_coords_A']:tt[k]=(np.asarray(t[k])@Q+offset).tolist()
    for a in pp['pocket_sites']:a['coords_A']=(np.asarray(a['coords_A'])@Q+offset).tolist()
    vv=MotifReward(pp,tt).evaluate(z@torch.tensor(Q)+torch.tensor(offset),logits,mask)[0]
    assert torch.allclose(v,vv,atol=1e-12)


def test_bounded_controller_uses_real_model_vjp_without_affinity_or_extra_forward():
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False);calls=0
        def __call__(self,curr,pocket,times,**kw):
            self.calls+=1;x=curr['coords'];return {'coords':1.3*x+.4,'atomics':x[...,0,None].expand(-1,-1,5),'affinity':{'pic50':x.sum((1,2))}}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    p,t=inputs();p['controller']='bounded';control=MotifControl(Model(),p,t,torch.zeros((1,3),dtype=torch.double))
    curr={'coords':torch.tensor([t['coords_A']],dtype=torch.double),'mask':torch.ones((1,6),dtype=torch.bool)}
    control.forward(curr,{},[torch.tensor([.1])]*3,None,None,None,10,.01)
    assert control.model.calls==1 and 0<float(control.dose_gain[0])<1
    grad=control.cached[0];assert grad.abs().max()>0
    control.after(curr,10);assert control.rows[-1]['nonzero_molecules']==1


def test_unknown_references_are_rejected_in_all_reference_fields():
    import jsonschema
    for k in ['evidence_refs','counterevidence_refs']:
        a={'evidence_refs':['r/context'],'counterevidence_refs':[]};a[k]=['missing']
        with pytest.raises(jsonschema.ValidationError):references(a,{'evidence':{'r/context':{}}})


def test_inactive_parameters_do_not_create_fake_new_pointwise_functions():
    p,t=inputs();p['operator']='pointwise';q=copy.deepcopy(p);q['band_multiplier']=3.
    assert identity(p,t)==identity(q,t)
    q['dose']['native_rms_ratio']=.7;assert identity(p,t)!=identity(q,t)
    p['operator']='orientation';q=copy.deepcopy(p);q['band_multiplier']=3.
    assert identity(p,t)==identity(q,t)


def test_legal_exchange_interaction_keeps_fixed_N_count():
    layouts=[('0,1',1.),('1,2',2.),('0,3',3.),('2,3',5.)]
    f=pd.DataFrame([dict(N_sites=s,pic50_target=v,is_original=(i==0),eligible_sites='0,1,2,3',exhaustive=False) for i,(s,v) in enumerate(layouts)])
    x=contrasts(f);assert len(x['single_exchanges'])==2
    assert x['largest_signed_interactions'][0]['difference_of_differences']==1.


def test_grouped_restore_preserves_conditioning_and_strict_suffix_clocks():
    state={'step':30,'current':{'coords':torch.arange(18).reshape(3,2,3)},'prior':{'coords':torch.ones((3,2,3))},
        'cond':{'coords':torch.ones((3,2,3))*2},'times':[torch.full((3,),.3)]*3}
    x=GroupedSuffix(state,[2,0],100,'cpu');prior,current,cond,times=x.restore()
    assert torch.equal(current['coords'],state['current']['coords'][[2,0]]) and cond['coords'].shape[0]==2
    x.before(30,current,times);x.after(30,current,[torch.full((2,),.31)]*3)
    with pytest.raises(ValueError):x.before(32,current,times)


def test_branch_cache_tracks_active_math_without_invalidating_unrelated_new_operators():
    from evomolsteer.local_motif.semantic_contract import selected_branch_signature
    a="def f(operator,x):\n    if operator=='a':\n        v=x*x\n    else:\n        v=-x\n    return v\n"
    b="def f(operator,x):\n    if operator=='a':\n        v=x*x\n    elif operator=='b':\n        v=3*x\n    else:\n        v=-x\n    return v\n"
    assert selected_branch_signature(a,'a')==selected_branch_signature(b,'a')
    assert selected_branch_signature(a,'a')!=selected_branch_signature(a.replace('x*x','x*x*x'),'a')


def test_explicit_parameter_paths_separate_physical_dose_from_curvature():
    from evomolsteer.local_motif.semantic_contract import variants
    p,t=inputs();p.update(operator='pointwise',region_id='R',template_id='T')
    r={k:p[k] for k in ['region_id','template_id','atom_slots','pocket_sites','orientation_triplets','base_band_A']}
    m={'learning_window':[0,.5],'allowed_starts':[0,.1,.3]}
    c=variants(p,{'factor':'dose.pointcloud_delta_A','observable_id':'pointwise'},r,t,m)
    assert any(q['dose']['pointcloud_delta_A']!=p['dose']['pointcloud_delta_A'] for q in c.values())
    assert all(q['dose']['native_rms_ratio']==p['dose']['native_rms_ratio'] for q in c.values())
    d=variants(p,{'factor':'dose.native_rms_ratio','observable_id':'pointwise'},r,t,m)
    assert all(q['dose']['pointcloud_delta_A']==p['dose']['pointcloud_delta_A'] for q in d.values())


def test_temporal_gain_changes_actual_step_after_normalization():
    class Model:
        coord_scale=1.;integrator=SimpleNamespace(use_cosine_scheduler=False)
        def __call__(self,curr,pocket,times,**kw):
            x=curr['coords'];return {'coords':x+.4,'atomics':x[...,0,None].expand(-1,-1,5)}
        def _get_predictions(self,out):return {'coords':out['coords']},None
    p,t=inputs();p['dose'].update(max_atom_step_A=10.,max_cumulative_rms_A=100.)
    curr={'coords':torch.tensor([t['coords_A']],dtype=torch.double),'mask':torch.ones((1,6),dtype=torch.bool)}
    rms=[]
    for schedule in ['constant','early']:
        q=copy.deepcopy(p);q['time_schedule']=schedule;c=MotifControl(Model(),q,t,torch.zeros((1,3),dtype=torch.double))
        c.forward(curr,{},[torch.tensor([.1],dtype=torch.double)]*3,None,None,None,10,.01);c.after(curr,10)
        rms.append(c.rows[-1]['injected_rms_A_mean'])
    assert rms[1]/rms[0]==pytest.approx(1.4875)


def test_path_target_fit_recovers_curves_and_enforces_clock_domain():
    from evomolsteer.local_motif.path_targets import fit_path,target_at
    times=np.linspace(0,.5,51);anchor=np.arange(18).reshape(6,3)/10
    points=anchor+(times[:,None,None]-.5)*2+(times[:,None,None]-.5)**2*3
    profile=fit_path(times,points,anchor,.5,[0,.5])
    assert profile['fit_rms_A']<1e-12
    assert np.allclose(target_at(profile,.2,[0,2]),points[20,[0,2]])
    assert np.allclose(target_at(profile,.5,[0,2]),anchor[[0,2]])
    with pytest.raises(ValueError):target_at(profile,.6,[0,2])


def test_moving_target_and_bridge_horizon_are_consistent():
    from evomolsteer.local_motif.path_targets import fit_path
    p,t=inputs();anchor=np.array(t['coords_A']);times=np.linspace(0,.5,51)
    t['historical_path']=fit_path(times,anchor+(times[:,None,None]-.5)*2,anchor,.5,[0,.5])
    p['target_schedule']='historical_path';r=MotifReward(p,t)
    assert r.target_clock(.1)==pytest.approx(.2)
    assert np.allclose(r.target,anchor[p['atom_slots']]-.6)
    assert r.target_clock(.49)==pytest.approx(.5)
    assert np.allclose(r.target,anchor[p['atom_slots']])


def test_novelty_actions_exclude_seen_numeric_programs():
    from evomolsteer.local_motif.actions import novel_actions
    p,t=inputs();p.update(operator='pointwise',region_id='R',template_id='T')
    r={k:p[k] for k in ['region_id','template_id','atom_slots','pocket_sites','orientation_triplets','base_band_A']}
    master={'learning_window':[0,.5],'allowed_starts':[0,.1,.3]}
    parents={'P':{'program':p,'legal_factors':['dose.native_rms_ratio']}}
    aa=novel_actions(parents,{'R':r},{'T':t},master)
    assert len(aa)==1 and len(aa[0]['options'])==2
    master['_seen_numeric_programs']=[identity(q,t) for q in aa[0]['options'].values()]
    assert novel_actions(parents,{'R':r},{'T':t},master)==[]


def test_full_state_clone_keys_preserve_equal_coordinates_with_different_bonds():
    from evomolsteer.local_motif.state_controls import full_state_controls
    class Trajectory:
        def read(self,key,step):
            if key=='current_charges':return np.zeros((3,2))
            return np.array([[[0,1],[1,0]],[[0,2],[2,0]],[[0,1],[1,0]]])
    result=full_state_controls(Trajectory(),0,np.zeros((3,2,3)),np.zeros((3,2)),0)
    assert result.tolist()==[1]
