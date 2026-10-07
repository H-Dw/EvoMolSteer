import numpy as np
import torch
from evomolsteer.continuous.affinity_endpoint import summarize

def test_whole_window_endpoint_effect_uses_independent_batches():
    rng=np.random.default_rng(42);times=np.arange(50)/100
    hi=rng.normal(size=(14,50,2))*.03+np.array([.8,0]);lo=np.zeros_like(hi);var=np.ones_like(hi)
    scale,effects,functions=summarize(times,hi,lo,var,hi*0,['known','null'])
    assert effects[0]['q']<.05 and effects[0]['high_low_integrated_z']>.75
    assert abs(effects[1]['high_low_integrated_z'])<.01
    np.testing.assert_array_equal(scale,np.ones(2))
    assert set(functions)=={'known','null'}

def test_live_endpoint_vjp_uses_one_forward_and_never_differentiates_head():
    from evomolsteer.generation.endpoint_controller import endpoint_pullback
    x=torch.tensor([[[.2,-.3,.4],[.1,.6,-.2]]],dtype=torch.float64)
    a=x.new_tensor([2.,3.,4.]);calls=[]
    def predict(z):
        calls.append(1);y=torch.sin(z*a)
        head=z.square().sum((1,2));head.register_hook(lambda g:(_ for _ in ()).throw(AssertionError('Affinity branch differentiated')))
        return {'coords':y,'atomics':torch.ones((1,2,1)), 'affinity':{'pic50':head}}, {'coords':y}
    def reward(y,atoms,mask,time,anchor):
        return y.sum((1,2)),{'dose_gate':y.new_ones(len(y))}
    pred,cond,g,_,_,_,_=endpoint_pullback(predict,reward,x,torch.ones((1,2),dtype=torch.bool),5.,x.new_zeros((1,3)),.2)
    torch.testing.assert_close(g,5*a*torch.cos(a*x))
    assert len(calls)==1 and not pred['coords'].requires_grad and not cond['coords'].requires_grad
    assert not pred['affinity']['pic50'].requires_grad

def test_endpoint_registered_pointcloud_gradient_is_on_endpoint_not_proposal():
    from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
    from evomolsteer.continuous.affinity_geometry import names
    points=np.zeros((1,3));target=np.array([[-1.,0,0],[1.,0,0]])
    ref={'schema_version':'affinity-endpoint-library-1.0','window':[.1,.4],'times':[.1],
         'features':names(points),'feature_scale':[1.]*len(names(points)),'landmarks_A':points.tolist(),'origin_A':[0,0,0],
         'frames':[{'teacher_endpoint_A':[target.tolist()],'teacher_scores':[8.]}]}
    p={'window':[.1,.4],'reward_view':'endpoint_pointcloud','derivative_path':'flowr_endpoint_vjp'}
    reward=EndpointGeometryReward(p,ref);x=torch.tensor(target*.6)[None].requires_grad_(True)
    value,_=reward(x,torch.zeros((1,2),dtype=torch.int64),torch.ones((1,2),dtype=torch.bool),.1,x.detach())
    g,=torch.autograd.grad(value.sum(),x)
    assert (g*(torch.tensor(target)[None]-x.detach())).sum()>0
    assert not reward.active(.39,.41) and reward.active(.1,.11)

def test_data_only_designer_compiler_binds_dynamic_evidence_support():
    from evomolsteer.continuous.endpoint_design import compile_endpoint
    response={'primary_objective':'predicted_affinity','affinity_head_gradient':False,
        'production_extra_forward_calls_per_step':0,'full_zero_validation_required':True,
        'reward_design':{'reward_view':'endpoint_direction','derivative_path':'flowr_endpoint_vjp','window':[.1,.4],
            'coordinate_representation':'predicted_endpoint_world_A','native_rms_ratio':.3,'time_ramp_power':0.,'geometry_block_weights':[.1,1.,2.,1.]}}
    ref={'schema_version':'affinity-endpoint-library-1.0','window':[.1,.4]}
    p=compile_endpoint(response,{},ref,'example_sha')
    assert p['window']==[.1,.4] and p['affinity_head_gradient'] is False
    response['reward_design']['window']=[0.,.5]
    import pytest
    with pytest.raises(ValueError):compile_endpoint(response,{},ref,'example_sha')

def test_literal_endpoint_agent_behavior_and_design_compile():
    from pathlib import Path
    from evomolsteer.io import read_json,digest
    from evomolsteer.generation.window_reference import load_reference
    from evomolsteer.continuous.endpoint_design import audit_endpoint_behavior,compile_endpoint
    root=Path(__file__).resolve().parents[1];folder=root/'docs/experiments/ck2_affinity_geometry30_20261007/endpoint_skill_test'
    response=read_json(folder/'response.json');ref=root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz'
    audit=audit_endpoint_behavior(root/'docs/experiments/skill_ablation_20261007/legacy_skills/affinity-endpoint-pullback.md',folder/'input.json',folder/'prompt.txt',response,digest(ref))
    program=compile_endpoint(response,{},load_reference(ref),digest(ref))
    assert audit['passed'] and audit['representation_intervention']
    assert program['reward_view']=='endpoint_direction' and program['geometry_block_weights']==[0,1,2,1]
