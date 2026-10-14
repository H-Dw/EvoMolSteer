import copy
import numpy as np
import pandas as pd
import pytest
import torch
from evomolsteer.continuous.batch_split import split_batches
from evomolsteer.continuous.terminal_regions import select_terminal_elites,partial_rank_correlation,_integrate
from evomolsteer.continuous.elite_path_credit import terminal_ancestors
from evomolsteer.generation.teacher_region_reward import TeacherRegionReward


def test_dynamic_split_uses_available_identifiers_and_preserves_exact_coverage():
    available=[5,23,101,900,3,11,33,77]
    a=split_batches(available,{'discovery':.75,'validation':.25},42)
    assert a==split_batches(available[::-1],{'discovery':.75,'validation':.25},42)
    assert len(a['discovery'])==6 and len(a['validation'])==2
    assert sorted(a['discovery']+a['validation'])==sorted(available)
    assert not set(a['discovery'])&set(a['validation'])


def test_topk_is_configured_and_uses_final_scores_instead_of_intermediate_scores():
    labels=pd.DataFrame({'batch':[1,1,3,3],'slot':[0,1,0,1],'valid_connected':[True,True,True,False],
        'smiles':['CC','CC','CN','CO'],'pic50_on_rescore':[8.,9.,8.5,20.],'intermediate_score':[100.,1.,10.,200.]})
    chosen=select_terminal_elites(labels,{'top_k':2,'deduplicate_smiles':True})
    assert chosen.path_id.tolist()==['1:1','3:0']
    assert len(select_terminal_elites(labels,{'top_k':1}))==1
    with pytest.raises(ValueError):select_terminal_elites(labels,{'top_k':4})


def test_nonidentity_parent_indices_reverse_real_branches():
    selected=np.array([[2,0,1],[1,1,0],[0,2,1]])
    ancestors=terminal_ancestors(selected)
    assert ancestors[:,1].tolist()==[2,0,2,1]
    assert not np.array_equal(ancestors[:,1],np.ones(4))


def test_integration_does_not_bridge_missing_stages():
    values=np.array([[0.],[1.],[np.nan],[100.],[101.]])
    mean,coverage=_integrate(values,np.arange(5.))
    np.testing.assert_allclose(mean,[50.5]);np.testing.assert_allclose(coverage,[.5])


def test_current_score_adjustment_removes_a_pure_instantaneous_score_proxy():
    current=np.arange(12.);future=current+np.sin(current)
    feature=np.c_[current,np.sin(current)]
    correlation=partial_rank_correlation(feature,future,current,np.full(12,1/12))
    assert np.isnan(correlation[0])


def reward_fixture():
    fields=['f']*20
    ref={'schema_version':'affinity-endpoint-library-1.0','window':[0.,.5],'times':[0.,.1,.2,.3,.4],
        'landmarks_A':[[0.,0.,0.],[20.,0.,0.]],'origin_A':[0.,0.,0.],
        'features':fields,'feature_scale':[1.]*20,
        'frames':[{'time':t,'teacher_endpoint_A':[[[0.,0.,0.],[1.,0.,0.],[20.,0.,0.]]],
            'teacher_scores':[0.],'teacher_base_log_weight':[0.]} for t in [0.,.1,.2,.3,.4]]}
    p={'reward_view':'endpoint_teacher_region','derivative_path':'flowr_endpoint_vjp','window':[0.,.5],
        'coordinate_region_weights':[1.,0.],'coordinate_region_radius_A':3.,'coordinate_background_weight':0,
        'teacher_neighbors':4,'teacher_endpoint_temperature_A2':4.,'teacher_score_beta':0,'mixture_temperature':.5}
    return p,ref


def test_local_loss_has_zero_direct_force_outside_teacher_region_and_correct_derivative():
    p,ref=reward_fixture();reward=TeacherRegionReward(p,ref)
    x=torch.tensor([[[.2,.1,0.],[1.2,.1,0.],[20.2,.1,0.]]],dtype=torch.float64,requires_grad=True)
    anchor=x.detach().clone();mask=torch.ones((1,3),dtype=torch.bool);atoms=torch.zeros((1,3),dtype=torch.long)
    value,detail=reward(x,atoms,mask,.1,anchor);g,=torch.autograd.grad(value.sum(),x)
    torch.testing.assert_close(g[:,2],torch.zeros_like(g[:,2]),rtol=0,atol=0)
    assert bool(g[:,:2].abs().sum()>0) and detail['regional_core_fraction'].item()==pytest.approx(2/3)
    direction=g/g.norm();epsilon=1e-5
    a,_=reward(x.detach()+epsilon*direction,atoms,mask,.1,anchor)
    b,_=reward(x.detach()-epsilon*direction,atoms,mask,.1,anchor)
    torch.testing.assert_close((a-b).sum()/(2*epsilon),(g*direction).sum(),rtol=1e-7,atol=1e-8)
    moved=x.detach().clone();moved[:,2]+=3
    c,_=reward(moved,atoms,mask,.1,anchor)
    torch.testing.assert_close(c,value.detach(),rtol=0,atol=0)


def test_delayed_window_is_dynamic_and_not_normalized_to_a_new_early_phase():
    p,ref=reward_fixture();p['window']=[.1,.5];ref['window']=[.1,.5]
    ref['times']=ref['times'][1:];ref['frames']=ref['frames'][1:]
    reward=TeacherRegionReward(p,ref)
    assert not reward.active(.09,.1) and reward.active(.1,.11)
    assert reward.active(.49,.5) and not reward.active(.5,.51)
