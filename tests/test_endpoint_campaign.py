import copy
from evomolsteer.generation.endpoint_campaign import propose_endpoint
from evomolsteer.generation.affinity_campaign import update_program,select_parent


def row(n,delta,family='endpoint_direction',valid=0):
    return dict(round=n,all_head_change_vs_native=delta,reward_view=family,valid_rate_change=valid,head_coverage=1.,
                negative_MMFF_relative_change=0.,negative_MMFF_p90_relative_change=0.)


def test_endpoint_weak_and_wrong_directions_require_different_interventions():
    programs={1:dict(reward_view='motif_mixture',native_rms_ratio=.3),5:dict(reward_view='endpoint_direction',native_rms_ratio=.3)}
    flat=[row(1,.06,'motif_mixture'),row(5,.001)]
    p=propose_endpoint(6,flat,programs)
    assert p['parent_round']==1 and p['template_round']==5 and p['native_rms_ratio']==.51
    bad=copy.deepcopy(flat);bad[-1]['all_head_change_vs_native']=-.06
    p=propose_endpoint(6,bad,programs)
    assert p['reward_view']=='endpoint_landmark' and p['parent_round']==1


def test_endpoint_bad_yield_keeps_report_and_global_parent():
    rows=[row(1,.06,'motif_mixture'),row(5,.2,valid=-.5)]
    p=propose_endpoint(6,rows,{5:dict(reward_view='endpoint_direction',native_rms_ratio=.3)})
    assert p['parent_round']==1 and p['native_rms_ratio']==.1 and p['reward_view']=='endpoint_landmark'


def test_endpoint_compiled_weights_survive_update_and_frozen_global_validation():
    parent=dict(reward_view='endpoint_direction',native_rms_ratio=.3,geometry_block_weights=[0,1,2,1])
    p=update_program(parent,{'geometry_block_weights':[0,1,2,0]},[.1,.4],'sha')
    assert p['geometry_block_weights']==[0,1,2,0] and p['derivative_path']=='flowr_endpoint_vjp'
    rows=[row(1,.08,'motif_mixture'),dict(row(26,.03),frozen_winner=1)]
    assert propose_endpoint(27,rows,{})['template_round']==1


def test_affinity_selection_handles_missing_secondary_energy_without_inventing_it():
    rows=[row(1,.07),row(2,.075)]
    rows[-1]['negative_MMFF_relative_change']=None
    assert select_parent(rows,True)['round']==1


def test_public_factory_uses_forecast_geometry_and_dynamic_support():
    import numpy as np,torch
    from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
    from evomolsteer.continuous.affinity_geometry import names
    points=np.zeros((1,3));target=np.array([[-1.,0,0],[1.,0,0]])
    ref={'schema_version':'affinity-endpoint-library-1.0','window':[.1,.4],'times':[.1],
         'features':names(points),'feature_scale':[1.]*len(names(points)),'landmarks_A':points.tolist(),'origin_A':[0,0,0],
         'frames':[{'teacher_endpoint_A':[target.tolist()],'teacher_scores':[8.]}]}
    p={'window':[.1,.4],'reward_view':'endpoint_pointcloud','derivative_path':'flowr_endpoint_vjp'}
    reward=make_coordinate_reward(p,ref);x=torch.tensor(target*.6)[None].requires_grad_(True)
    value,_=reward(x,torch.zeros((1,2),dtype=torch.int64),torch.ones((1,2),dtype=torch.bool),.1,x.detach())
    g,=torch.autograd.grad(value.sum(),x)
    assert (g*(torch.tensor(target)[None]-x.detach())).sum()>0
    assert not reward.active(.39,.41)
