import torch
from evomolsteer.generation.coordinate_evaluation import regional_attention_summary
from test_regional_point_reward import inputs
from evomolsteer.generation.regional_point_reward import RegionalPointReward


def test_actual_reward_attention_is_retained_without_inventing_native_values():
    p,ref=inputs();reward=RegionalPointReward(p,ref)
    x=torch.tensor(ref['frames'][25]['teacher_endpoint_A'][:2],dtype=torch.float64)+.1
    mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    _,detail=reward(x,atoms,mask,.25,x)
    trace={key:value.tolist() for key,value in detail.items() if key.startswith('regional_')}
    summary=regional_attention_summary(trace)
    assert summary['regional_attention_min_mean']>=p['coordinate_background_weight']
    assert summary['regional_attention_max_mean']>summary['regional_attention_min_mean']
    assert 0<=summary['regional_core_fraction_mean']<=1
    assert summary['injection_noncore_fraction_mean'] is None
    assert all(v is None for v in regional_attention_summary({}).values())
