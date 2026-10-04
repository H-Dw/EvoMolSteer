import copy
import numpy as np
import torch
import pytest
from evomolsteer.geometry import measure
from evomolsteer.reward import RewardProgram,NotApplicable
from evomolsteer.statistics import bh,cluster_summary,stage_effect_rates,lineage_intervals
import pandas as pd

def fixture():
    cat={'atom_vocabulary':{'C':3,'N':4,'O':5,'S':9},'regions':{'p:A:GLU1':{'points_A':[[0.,0.,0.],[0.,1.,0.]]}},
         'features':{'p:A:GLU1::hetero_distance_softmin':{'region':'p:A:GLU1','kind':'hetero_distance_softmin',
             'elements':['N','O','S'],'temperature_A':.25,'differentiable_supported':True}}}
    term={'term_id':'t','rule_id':'r','target_id':'z','feature':'p:A:GLU1::hetero_distance_softmin',
          'lower':2.,'upper':3.,'scale':1.,'weight':1.,'stage_start':0.,'stage_end':1.,'gate_width':.02,
          'evidence_ids':['e'],'parameter_status':'exploratory_pilot'}
    p={'version':'2.0','selection_window':{'start':0.,'end':1.,'time_axis':'score_time'},'representation':'predicted_endpoint_world_A','direction':'reward_ascent',
       'operator':'negative_smooth_window_sum','terms':[term],
       'constraints':{'max_atom_displacement_A':.05,'max_rms_displacement_A':.05,'max_pair_distance_change_A':.1,
                      'clash_threshold_A':1.2,'max_new_clash_pairs':0,'preserve_fixed_atoms':True}}
    return p,cat

def test_batch_inference_not_particle_count():
    cfg={'minimum_inference_batches':4,'seed':42,'permutations':256}
    result=cluster_summary([100.,100.],cfg,'two_batches')
    assert result['n_batches']==2 and np.isnan(result['p_value'])
    np.testing.assert_allclose(bh([.01,.04,.03,np.nan])[:3],[.03,.04,.04])

def test_rates_use_nonuniform_time_and_do_not_bridge_missing_stage():
    df=pd.DataFrame([{'feature':'x','batch':0,'stage':i,'time':t,'value':v} for i,t,v in [(0,.05,1),(1,.20,4),(3,.8,9)]])
    rates=stage_effect_rates(df,['feature'],'value')
    assert len(rates)==1 and rates.iloc[0]['rate']==pytest.approx(20.)

def test_reward_direction_finite_difference_and_fixed_mask():
    p,c=fixture();r=RewardProgram(p,c)
    x=torch.tensor([[[5.,.5,0.],[6.,1.,0.]]],dtype=torch.float64,requires_grad=True)
    atom=torch.tensor([[4,3]]);mask=torch.ones((1,2),dtype=torch.bool)
    v,_=r(x,atom,mask,.5);grad=torch.autograd.grad(v.sum(),x)[0]
    assert grad[0,0,0]<0 # pull a distant eligible nitrogen toward supported window
    eps=1e-5;a=x.detach().clone();b=a.clone();a[0,0,0]+=eps;b[0,0,0]-=eps
    numeric=(r(a,atom,mask,.5)[0]-r(b,atom,mask,.5)[0])/(2*eps)
    assert numeric.item()==pytest.approx(grad[0,0,0].item(),rel=1e-6)
    editable=mask.clone();editable[:,1]=False
    y,audit=r.guarded_step(x.detach(),atom,mask,.5,editable)
    assert torch.equal(y[:,1],x.detach()[:,1])
    assert r(y,atom,mask,.5)[0]>v.detach()

def test_undefined_objective_and_unapproved_operator_are_rejected():
    p,c=fixture();r=RewardProgram(p,c)
    with pytest.raises(NotApplicable):r(torch.ones((1,2,3)),torch.tensor([[3,3]]),torch.ones((1,2),dtype=torch.bool),.5)
    bad=copy.deepcopy(p);bad['operator']='arbitrary_python'
    with pytest.raises(Exception):RewardProgram(bad,c)

def test_selection_decomposition_is_not_a_generation_effect():
    phi=np.array([1.,4.,9.]);freq=np.array([0.,0.,1.]);next_population=np.array([10.,10.,10.])
    selection=freq@phi-phi.mean();within=next_population.mean()-freq@phi
    assert selection+within==pytest.approx(next_population.mean()-phi.mean())
    assert within==1. and selection>4.

def test_proposal_lineage_uses_advanced_nonuniform_geometry_times():
    # score times .1,.2 have dt .1; proposals at .2,.5 instead span .3.
    np.testing.assert_allclose(lineage_intervals([.2],[.5]),[.3])
    with pytest.raises(ValueError):lineage_intervals([.5],[.2])

def test_nonfinite_reward_parameters_and_coordinates_are_rejected():
    p,c=fixture();bad=copy.deepcopy(p);bad['terms'][0]['lower']=float('nan')
    with pytest.raises(ValueError):RewardProgram(bad,c)
    with pytest.raises(NotApplicable):
        RewardProgram(p,c)(torch.full((1,2,3),float('nan')),torch.tensor([[4,3]]),torch.ones((1,2),dtype=torch.bool),.5)
