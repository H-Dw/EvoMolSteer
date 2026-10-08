import copy,gzip,json
from pathlib import Path
import numpy as np
import pytest
import torch
from evomolsteer.generation.innovation_reward import InnovationReward
from evomolsteer.generation.endpoint_reward import EndpointGeometryReward

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_null_is_exact_and_bounded_shift_has_conditional_gradient(dtype):
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/'configs/experiments/skill_ablation_v1/incumbent.json').read_text())
    r=json.loads(gzip.decompress((root/'configs/experiments/skill_ablation_v1/endpoint_reference.json.gz').read_bytes()))
    r['times']=r['times'][:1];r['frames']=r['frames'][:1]
    f=r['frames'][0];shape=np.asarray(f['teacher_endpoint_A']).shape
    f.update(teacher_contrast_direction_unit=np.ones(shape).tolist(),teacher_contrast_confidence=np.ones(shape[:1]).tolist(),
        teacher_contrast_atom_weight=np.ones(shape[:2]).tolist())
    x=torch.tensor(np.asarray(f['teacher_endpoint_A'])[:2]+.2,dtype=dtype,requires_grad=True)
    mask=torch.ones(x.shape[:2],dtype=torch.bool);atoms=torch.zeros_like(mask,dtype=torch.long);anchor=x.detach()
    a,_=EndpointGeometryReward(p,r)(x,atoms,mask,0,anchor)
    b,_=InnovationReward(p,r)(x,atoms,mask,0,anchor)
    ga,=torch.autograd.grad(a.sum(),x);gb,=torch.autograd.grad(b.sum(),x)
    assert torch.equal(a,b) and torch.equal(ga,gb)
    p['innovation']={'field_strength_A':.1};reward=InnovationReward(p,r)
    value,detail=reward(x,atoms,mask,0,anchor);g,=torch.autograd.grad(value.sum(),x)
    assert float(detail['contrast_teacher_shift_rms_A'].min())>0
    assert not torch.equal(g,ga)
    if dtype==torch.float64:
        v=g/g.norm();eps=1e-5
        a,_=reward(x.detach()+eps*v,atoms,mask,0,anchor);b,_=reward(x.detach()-eps*v,atoms,mask,0,anchor)
        assert torch.allclose((a.sum()-b.sum())/(2*eps),(g*v).sum(),rtol=1e-5,atol=1e-8)
