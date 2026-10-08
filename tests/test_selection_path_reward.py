import copy
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import pytest
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward
from evomolsteer.generation.selection_path_reward import SelectionPathReward
from evomolsteer.continuous.selection_pressure import coordinate_niches

def fixture():
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/'configs/experiments/skill_ablation_v1/incumbent.json').read_text())
    r=json.loads(gzip.decompress((root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz').read_bytes()))
    r['frames']=r['frames'][:1];r['times']=r['times'][:1]
    f=r['frames'][0];g,m,_=coordinate_niches(f['teacher_endpoint_A'],f['teacher_batches'])
    f['teacher_niche']=g.tolist();f['teacher_precision']=m.tolist();r['selection_path']={}
    x=torch.tensor(np.asarray(f['teacher_endpoint_A'])[:2]+.1,dtype=torch.float64)
    return p,r,x

def test_empty_mechanism_exact_incumbent_scalar_and_gradient():
    p,r,x=fixture();x.requires_grad_(True);mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    base=EndpointGeometryReward(p,r);new=SelectionPathReward(p,r)
    a,_=base(x,atoms,mask,0.,x.detach());b,_=new(x,atoms,mask,0.,x.detach())
    ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x)
    assert torch.equal(a,b) and torch.equal(ga,gb)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_inactive_floor_exact_incumbent_arithmetic(dtype):
    p,r,x=fixture();x=x.detach().to(dtype).requires_grad_(True);mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    base=EndpointGeometryReward(p,r);p['selection_path']={'ess_fraction':0.};new=SelectionPathReward(p,r)
    a,_=base(x,atoms,mask,0.,x.detach());b,_=new(x,atoms,mask,0.,x.detach())
    ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x)
    assert torch.equal(a,b) and torch.equal(ga,gb)


@pytest.mark.parametrize('spec',[{'niche_balance':.5},{'precision_mix':.5}])
def test_no_geometric_information_reduces_to_incumbent(spec):
    p,r,x=fixture();f=r['frames'][0];m,n=np.asarray(f['teacher_endpoint_A']).shape[:2]
    f['teacher_niche']=[0]*m;f['teacher_precision']=np.broadcast_to(np.eye(3),(m,n,3,3)).tolist()
    x=x.detach().float().requires_grad_(True);mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long)
    base=EndpointGeometryReward(p,r);p['selection_path']=spec;new=SelectionPathReward(p,r)
    a,_=base(x,atoms,mask,0.,x.detach());b,_=new(x,atoms,mask,0.,x.detach())
    ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x)
    assert torch.equal(a,b) and torch.equal(ga,gb)

def test_conditional_coordinate_gradient_not_time_derivative():
    p,r,x=fixture();p['selection_path']={'precision_mix':.25,'ess_fraction':.75}
    reward=SelectionPathReward(p,r);x.requires_grad_(True)
    mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long);anchor=x.detach().clone()
    value,detail=reward(x,atoms,mask,0.,anchor);g,=torch.autograd.grad(value.sum(),x)
    v=g/g.norm();eps=1e-5
    a,_=reward(x.detach()+eps*v,atoms,mask,0.,anchor);b,_=reward(x.detach()-eps*v,atoms,mask,0.,anchor)
    assert torch.allclose((a.sum()-b.sum())/(2*eps),(g*v).sum(),rtol=1e-5,atol=1e-8)
    assert detail['selection_pressure_audit'][:,0].min()>=3-1e-8
    assert detail['selection_pressure_audit'].shape==(2,4)
    assert detail['selection_pressure_audit'][:,3].min()>=1-1e-8
