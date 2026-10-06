import copy
from pathlib import Path
import numpy as np
import torch
from evomolsteer.io import read_json
from evomolsteer.generation.window_reference import load_reference
from evomolsteer.generation.regional_point_reward import RegionalPointReward
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward


def inputs():
    root=Path(__file__).resolve().parents[1]
    p=read_json(root/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round07.json')
    ref=load_reference(root/'configs/experiments/ck2_affinity_geometry30_v1/endpoint_reference.json.gz')
    p.update(reward_view='endpoint_regional_pointcloud',coordinate_region_weights=[1.]+[0.]*19,coordinate_region_radius_A=5.,coordinate_background_weight=.25)
    return p,ref


def test_fixed_spatial_attention_gradient_matches_conditional_finite_difference():
    p,ref=inputs();reward=RegionalPointReward(p,ref)
    x=torch.tensor(ref['frames'][25]['teacher_endpoint_A'][:1],dtype=torch.float64)+.1;x.requires_grad_(True)
    anchor=x.detach().clone();mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    value,detail=reward(x,atoms,mask,.25,anchor);g,=torch.autograd.grad(value.sum(),x);d=g/g.norm();eps=1e-5
    a,_=reward(x.detach()+eps*d,atoms,mask,.25,anchor);b,_=reward(x.detach()-eps*d,atoms,mask,.25,anchor)
    torch.testing.assert_close((a-b).sum()/(2*eps),(g*d).sum(),rtol=1e-5,atol=1e-6)
    assert value.max()<=0 and detail['regional_attention_min'].min()>=.25


def test_uniform_attention_recovers_original_point_cloud_and_atom_types_are_unused():
    p,ref=inputs();p['coordinate_region_radius_A']=1e12
    reward=RegionalPointReward(p,ref);original=copy.deepcopy(p);original['reward_view']='endpoint_pointcloud'
    baseline=EndpointGeometryReward(original,ref);x=torch.tensor(ref['frames'][25]['teacher_endpoint_A'][:1],dtype=torch.float64)+.1
    mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    value,_=reward(x,atoms,mask,.25,x);expected,_=baseline(x,atoms,mask,.25,x)
    torch.testing.assert_close(value,expected,rtol=1e-10,atol=1e-12)
    changed,_=reward(x,atoms+4,mask,.25,x);torch.testing.assert_close(changed,value)
