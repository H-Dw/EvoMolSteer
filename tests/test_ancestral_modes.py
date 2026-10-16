"""Numerical/representation invariants of stage-matched local control."""
import copy
import numpy as np
import pytest
import torch
from evomolsteer.ancestral_modes.bank import lookup
from evomolsteer.ancestral_modes.reward import AncestorReward,horizon,temporal_gain
from evomolsteer.ancestral_modes.programs import numerical_identity,validate


def fixture():
    q=np.array([[0.,0.,0.],[1.,.1,0.],[2.,1.,0.],[3.,0.,1.],[4.,0.,0.]])
    probs=np.array([[.9,.08,.02],[.1,.8,.1],[.1,.1,.8],[.2,.7,.1],[.8,.1,.1]])
    m={'coords_A':q.tolist(),'terminal_coords_A':(q+.2).tolist(),'time':.6,'trajectory_sha256':'synthetic',
        'native_elements':['C','N','O','N','C'],'atom_vocabulary':{'C':0,'N':1,'O':2},'charge_vocabulary':{'0':0,'1':1,'-1':2},
        'path':{'times':[.2,.4,.6],'coords_A':[(q-.8).tolist(),(q-.4).tolist(),q.tolist()],
            'endpoint_A':[(q+.3).tolist(),(q+.25).tolist(),(q+.2).tolist()],
            'element_probs':[probs.tolist()]*3,'charge_probs':[[[.8,.1,.1]]*5]*3}}
    teacher={'time':.6,'modes':{'A':m}}
    p={'family':'ancestral_modes_v3','clock_policy':'fixed','learning_window':[.2,.6],'window':[.2,.6],'steps':20,'teacher_time':.6,
        'mode_specs':[{'template_id':'A','region_id':'A__local','atom_slots':[0,1],
            'orientation_triplets':[],'pocket_sites':[{'coords_A':[0.,-2.,0.]}]}],
        'routing':'single','mode_memory_t':.08,'switch_margin_A':.1,'lookahead_fraction':.2,
        'operator':'pointwise','dose':{'native_rms_ratio':.6,'max_atom_step_A':1.,'max_cumulative_rms_A':20.,'pointcloud_delta_A':1.},
        'dose_schedule':'constant','schedule_normalization':1.,'attainment_response':False,'assignment_element_cost':.5,
        'target_band_A':.3,'typed_weight':0.,'charge_weight':0.,'sensing':'detached','field_sigma_A':1.2,
        'typed_channels':'all','type_target':'predicted','density_scale_floor':[.03,.1,.1,.1],'advantage_cap':20.,
        'native_options':{},'smc':False,'affinity_head_gradient':False,'coordinate_updates_only':True}
    return p,teacher


def test_lookup_clock_and_representation():
    p,b=fixture();m=b['modes']['A']
    assert np.allclose(lookup(m,.3),np.asarray(m['coords_A'])-.6)
    assert not np.allclose(lookup(m,.3),lookup(m,.3,'endpoint_A'))
    with pytest.raises(ValueError):lookup(m,.1)
    with pytest.raises(ValueError):lookup(m,.7)
    p['clock_policy']='current';assert horizon(p,.3)==.3
    p['clock_policy']='lookahead';assert horizon(p,.3)==pytest.approx(.38)


@pytest.mark.parametrize('clock',['current','fixed','lookahead','endpoint_prediction'])
@pytest.mark.parametrize('sensing',['detached','full'])
def test_finite_difference_conditional_real_output_chain(clock,sensing):
    p,b=fixture();p.update(clock_policy=clock,sensing=sensing,typed_weight=1. if sensing=='full' else 0.)
    x=torch.tensor([[[.3,.1,.2],[1.4,.3,.1],[2.3,1.,.2],[3.2,.2,1.],[4.2,.2,0.]]],dtype=torch.double,requires_grad=True)
    assignment=torch.tensor([[0,1]]);mask=torch.ones(1,5,dtype=torch.bool)
    def f(x):
        y=x+.1*x.square();logits=torch.stack([x[...,0]*.2,x[...,1]*.3,-x[...,2]],-1)
        charge=torch.stack([x[...,0]*.1,-x[...,0]*.2,x[...,2]*.2],-1)
        return AncestorReward(p,b).evaluate(x,y,logits,charge,mask,.3,torch.tensor([0]),[assignment])[0].sum()
    g=torch.autograd.grad(f(x),x)[0];eps=1e-6
    for idx in ((0,0,0),(0,1,1),(0,4,0)):
        a=x.detach().clone();c=a.clone();a[idx]+=eps;c[idx]-=eps
        assert float(g[idx])==pytest.approx(float((f(a)-f(c))/(2*eps)),abs=2e-7)


def test_current_geometry_has_no_endpoint_pullback():
    p,b=fixture();p['clock_policy']='current';r=AncestorReward(p,b)
    x=torch.randn(1,5,3,dtype=torch.double,requires_grad=True);y=torch.randn_like(x,requires_grad=True)
    value,_=r.evaluate(x,y,torch.zeros(1,5,3,dtype=torch.double),torch.zeros(1,5,3,dtype=torch.double),torch.ones(1,5,dtype=torch.bool),.3)
    gx,gy=torch.autograd.grad(value.sum(),(x,y),allow_unused=True)
    assert gx is not None and gy is None


def test_type_probability_derivative_is_active_and_local():
    p,b=fixture();p.update(clock_policy='current',typed_weight=5.)
    x=torch.tensor([[[.3,.1,0.],[1.1,.1,.2],[2.,1.,0.],[3.,0.,1.],[4.,0.,0.]]],dtype=torch.double,requires_grad=True)
    mask=torch.ones(1,5,dtype=torch.bool);assignment=[torch.tensor([[0,1]])]
    def derivative(mode):
        pp=copy.deepcopy(p);pp['sensing']=mode
        logits=torch.stack([x[...,0],-x[...,0],x[...,1]],-1)
        v,d=AncestorReward(pp,b).evaluate(x,x+.2,logits,logits,mask,.3,torch.tensor([0]),assignment)
        return torch.autograd.grad(v.sum(),x,retain_graph=True)[0],d
    g0,d=derivative('detached');g1,_=derivative('full')
    assert not torch.allclose(g0,g1)
    assert d['direct_mask'].sum()==2 and torch.all(g1[:,2:]==0)


def test_seed_specific_modes_and_proper_support():
    p,b=fixture();p.update(clock_policy='fixed',routing='compatible',assignment_element_cost=0.)
    other=copy.deepcopy(b['modes']['A']);other['path']['coords_A']=(np.asarray(other['path']['coords_A'])+10).tolist()
    b['modes']['B']=other;s=copy.deepcopy(p['mode_specs'][0]);s.update(template_id='B',region_id='B__local');p['mode_specs'].append(s)
    q=np.asarray(b['modes']['A']['coords_A']);x=torch.tensor(np.stack([q+.05,q+10.05]),dtype=torch.double)
    logits=torch.zeros(2,5,3,dtype=torch.double);mask=torch.ones(2,5,dtype=torch.bool)
    _,d=AncestorReward(p,b).evaluate(x,x,logits,logits,mask,.3)
    assert d['mode'].tolist()==[0,1]
    assert d['direct_mask'].sum(1).tolist()==[2,2]
    p['mode_specs'][0]['atom_slots']=list(range(5))
    with pytest.raises(ValueError):AncestorReward(p,b)


def test_inactive_parameters_do_not_create_new_programs():
    p,b=fixture();q=copy.deepcopy(p);q.update(typed_channels='NC',sensing='full',type_target='terminal',field_sigma_A=9.)
    assert numerical_identity(p,b)==numerical_identity(q,b)
    q['typed_weight']=1.;assert numerical_identity(p,b)!=numerical_identity(q,b)
    p['operator']=q['operator']='pair_distance';q=copy.deepcopy(p);q['dose']['pointcloud_delta_A']=99.
    assert numerical_identity(p,b)==numerical_identity(q,b)


def test_time_gain_after_normalization_and_nonstandard_window():
    p,b=fixture();p['dose_schedule']='late'
    assert temporal_gain(p,.2,.02)<temporal_gain(p,.5,.02)
    p['dose_schedule']='early';assert temporal_gain(p,.2,.02)>temporal_gain(p,.5,.02)
    p=validate(p,b);assert p['expected_active_steps']==list(range(4,12))


def test_legacy_local_potential_value_and_gradient_parity():
    from evomolsteer.local_motif.reward import MotifReward
    p,b=fixture();t=copy.deepcopy(b['modes']['A']);t['time']=.6
    old={'family':'local_motif_v2','atom_slots':[0,1],'operator':'pointwise','teacher_time':.6,'window':[.2,.6],
        'coordinate_updates_only':True,'smc':False,'affinity_head_gradient':False,'representation':'bridge',
        'controller':'unit_rms','assignment_element_cost':.5,'dose':p['dose'],'band_multiplier':1.,'base_band_A':.3,'orientation_triplets':[]}
    x=torch.randn(2,5,3,dtype=torch.double,requires_grad=True);logits=torch.randn(2,5,3,dtype=torch.double);mask=torch.ones(2,5,dtype=torch.bool)
    legacy=MotifReward(old,t);v0,_=legacy.evaluate(x,logits,mask)
    v1,_=AncestorReward(p,b).evaluate(x,x,logits,logits,mask,.3)
    assert torch.allclose(v0,v1,atol=1e-12)
    assert torch.allclose(torch.autograd.grad(v0.sum(),x,retain_graph=True)[0],torch.autograd.grad(v1.sum(),x)[0],atol=1e-12)


def test_amide_and_carboxylate_are_distinct_local_groups():
    from evomolsteer.ancestral_modes.chemistry import groups_from_graph
    roles=[{'atom_index':i,'element':e,'formal_charge':q} for i,(e,q) in enumerate([('C',0),('O',0),('N',0),('O',-1)])]
    edges=[{'i':0,'j':1,'bond_type':'DOUBLE'},{'i':0,'j':2,'bond_type':'SINGLE'}]
    assert groups_from_graph(roles,edges)[0]['group']=='amide'
    edges[1]['j']=3
    value=groups_from_graph(roles,edges)[0]
    assert value['group']=='carboxylate' and value['formal_charge_sum']==-1


def test_attained_geometry_does_not_suppress_unmet_chemical_field():
    from evomolsteer.ancestral_modes.reward import attainment_gain
    p,b=fixture();error=torch.zeros(2)
    comp={'typed':torch.tensor([-1.,0.]),'charge':torch.zeros(2)}
    assert torch.all(attainment_gain(p,error,comp)==0)
    p['typed_weight']=2.
    gain=attainment_gain(p,error,comp)
    assert gain[0]>.9 and gain[1]==0


def test_fixed_space_advantage_uses_non_self_centered_feature_and_correct_gradient():
    p,b=fixture();p.update(operator='selection_advantage',clock_policy='current',assignment_element_cost=0.)
    centers=np.asarray(b['modes']['A']['terminal_coords_A'])[:2]
    profile={'times':[.2,.6],'feature_centers_A':centers.tolist(),'focal':[[.5,-.1,0.,.1]]*2,
        'background_mean':[[.4,.1,.1,0.]]*2,'background_sd':[[.2,.5,.5,.5]]*2}
    p['mode_specs'][0]['selection_profile']=profile
    x=torch.tensor([[[.3,.1,.2],[1.2,.3,.1],[2.,1.,0.],[3.,0.,1.],[4.,0.,0.]]],dtype=torch.double,requires_grad=True)
    mask=torch.ones(1,5,dtype=torch.bool);logits=torch.zeros(1,5,3,dtype=torch.double)
    def f(z):return AncestorReward(p,b).evaluate(z,z,logits,logits,mask,.3,torch.tensor([0]),[torch.tensor([[0,1]])])[0].sum()
    g=torch.autograd.grad(f(x),x)[0];a=x.detach().clone();c=a.clone();a[0,0,0]+=1e-6;c[0,0,0]-=1e-6
    assert float(g[0,0,0])==pytest.approx(float((f(a)-f(c))/2e-6),abs=1e-7)
    historical=np.asarray(b['modes']['A']['path']['coords_A'][0])[:2]
    mass=np.exp(-np.square(historical-centers).sum(-1)/(2*p['field_sigma_A']**2)).mean()
    assert mass<.9  # It is not identically one merely because this is the focal path.


def test_response_router_uses_real_leaf_and_can_prefer_farther_controllable_mode():
    p,b=fixture();p.update(routing='response',assignment_element_cost=0.)
    other=copy.deepcopy(b['modes']['A']);b['modes']['B']=other
    s=copy.deepcopy(p['mode_specs'][0]);s.update(template_id='B',region_id='B__local');p['mode_specs'].append(s)
    r=AncestorReward(p,b)
    x=torch.ones(1,5,3,dtype=torch.double,requires_grad=True);mask=torch.ones(1,5,dtype=torch.bool)
    logits=torch.zeros(1,5,3,dtype=torch.double)
    def synthetic(current,endpoint,atom_logits,charge_logits,mask,t,index,assignment=None):
        # Mode 1 is farther but its real coordinate response is much larger.
        value=-(.1 if index==0 else 2.)*x[:,0,0].square()
        error=x.new_tensor([.1 if index==0 else 2.])
        return value,{'error':error,'assignment':torch.tensor([[0,1]]),'direct_mask':torch.tensor([[True,True,False,False,False]]),
            'clock':t,'base':value,'typed':value*0,'charge':value*0}
    # Different offsets make the local response/gap ratios different at this point.
    original=synthetic
    def mode(*args,**kwargs):
        value,d=original(*args,**kwargs)
        if args[6]==0:value=value-1.;d['base']=value
        return value,d
    r.evaluate_mode=mode
    with pytest.raises(ValueError):r.evaluate(x,x,logits,logits,mask,.3)
    value,d=r.evaluate(x,x,logits,logits,mask,.3,derivative_input=x)
    assert d['mode'].tolist()==[1] and d['routing_VJPs']==2
    assert torch.autograd.grad(value.sum(),x)[0].abs().sum()>0


def test_response_parameters_only_enter_active_numerical_identity():
    p,b=fixture();q=copy.deepcopy(p);q['response_refresh_fraction']=.1
    assert numerical_identity(p,b)==numerical_identity(q,b)
    p['routing']=q['routing']='response'
    assert numerical_identity(p,b)!=numerical_identity(q,b)


def test_authoritative_delta_distinguishes_route_change_from_metadata():
    from evomolsteer.ancestral_modes.programs import program_delta
    p,b=fixture();q=copy.deepcopy(p);q.update(program_id='new',routing='compatible',analyst_sha256='new')
    assert program_delta(p,q)=={'routing':{'before':'single','after':'compatible'}}


def test_class_balanced_layout_detects_majority_element_shortcut():
    from evomolsteer.ancestral_modes.layout import class_scores
    s=class_scores([0,0,0,0,1],[0,0,0,0,0],{'C':0,'N':1,'O':2})
    assert s['C_recall']==1 and s['N_recall']==0 and s['NC_balanced_recall']==.5
    assert s['N_precision'] is None and s['O_recall'] is None


def test_advantage_rejects_forecast_current_mismatch_and_zero_contrast_has_zero_dose_gain():
    p,b=fixture();p.update(operator='selection_advantage',clock_policy='endpoint_prediction',assignment_element_cost=0.,advantage_evidence_gain=True)
    profile={'times':[.2,.6],'feature_centers_A':b['modes']['A']['terminal_coords_A'][:2],
        'focal':[[.5,0.,0.,0.]]*2,'background_mean':[[.5,0.,0.,0.]]*2,'background_sd':[[.2,.5,.5,.5]]*2}
    p['mode_specs'][0]['selection_profile']=profile
    with pytest.raises(ValueError,match='representation'):AncestorReward(p,b)
    profile['feature_representation']='endpoint_prediction'
    r=AncestorReward(p,b);x=torch.randn(1,5,3,dtype=torch.double,requires_grad=True);logits=torch.zeros_like(x)
    value,d=r.evaluate(x,x,logits,logits,torch.ones(1,5,dtype=torch.bool),.3)
    assert torch.all(d['evidence_gain']==0) and torch.all(value==0)


def test_designer_scalar_options_cannot_escape_analyst_clock_or_type_channels():
    from evomolsteer.ancestral_modes.programs import scientific_group_key
    p,b=fixture();a={'parent_id':'parent','region_ids':['region'],'phase':'probe','program':p};q=copy.deepcopy(a)
    q['program']['clock_policy']='endpoint_prediction'
    assert scientific_group_key(a)!=scientific_group_key(q)


def test_balanced_element_field_full_coordinate_finite_difference():
    p,b=fixture();p.update(typed_weight=2.,sensing='full',type_balance='target_mass',type_mass_floor=.05)
    x=torch.randn(1,5,3,dtype=torch.double,requires_grad=True);mask=torch.ones(1,5,dtype=torch.bool)
    def f(z):
        logits=torch.stack([z[...,0],z[...,1],-z[...,2]],-1)
        return AncestorReward(p,b).evaluate(z,z+.1*z.square(),logits,logits,mask,.3,torch.tensor([0]),[torch.tensor([[0,1]])])[0].sum()
    g=torch.autograd.grad(f(x),x)[0];a=x.detach().clone();c=a.clone();a[0,1,0]+=1e-6;c[0,1,0]-=1e-6
    assert float(g[0,1,0])==pytest.approx(float((f(a)-f(c))/2e-6),abs=1e-6)
    q=copy.deepcopy(a);q['program']['dose']['native_rms_ratio']=1.2
    assert scientific_group_key(a)==scientific_group_key(q)
    a['program']['typed_weight']=q['program']['typed_weight']=1.
    q['program']['typed_channels']='NC'
    assert scientific_group_key(a)!=scientific_group_key(q)
