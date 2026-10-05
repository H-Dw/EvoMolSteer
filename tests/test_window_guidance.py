import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from evomolsteer.generation.window_reward import WindowReward
from evomolsteer.generation.window_controller import no_selection_source


def fixture():
    coords=[[0.,0.,0.],[1.2,.1,0.],[2.,1.,.2]]
    frame={'time':.3,'mass':1.,'coords':coords,'atomics':[3,4,5],
           'bonds':[[0,1,0],[1,0,2],[0,2,0]]}
    reference={'window':[.2,.4],'times':[.2,.3,.4],
               'frames':[dict(frame,time=t) for t in [.2,.3,.4]]}
    program={'window':[.2,.4],'native_rms_ratio':.1,'typed_weight':.3,'bond_weight':.2,
             'temperature':.03,'sigmas_A':[.7,1.4,2.8]}
    x=torch.tensor([coords],dtype=torch.float64)+torch.tensor([.4,.2,.1])
    return WindowReward(program,reference),x,torch.tensor([[3,4,5]]),torch.tensor([frame['bonds']])


def test_dynamic_gate_exact_state_time():
    reward,*_=fixture()
    assert reward.active(.2,.3) and reward.active(.3,.4)
    assert not reward.active(.1,.2) and not reward.active(.4,.5)
    assert reward.active(.39,.40000001)
    with pytest.raises(ValueError):WindowReward(dict(reward.program,window=[0,.5]),reward.reference)
    with pytest.raises(ValueError):reward.bank(.31,torch.zeros(1,3,3))


def test_gradient_finite_difference_permutation_and_no_atom_gradient_claim():
    reward,x,a,b=fixture();x.requires_grad_(True)
    value=reward(x,a,b,.3)[0];g,=torch.autograd.grad(value.sum(),x)
    assert torch.isfinite(g).all() and g.norm()>0
    direction=g/g.norm();eps=1e-5
    numeric=(reward(x+eps*direction,a,b,.3)[0]-reward(x-eps*direction,a,b,.3)[0])/(2*eps)
    torch.testing.assert_close(numeric.sum(),(g*direction).sum(),rtol=1e-6,atol=1e-8)
    perm=torch.tensor([2,0,1])
    torch.testing.assert_close(value,reward(x[:,perm],a[:,perm],b[:,perm][:,:,perm],.3)[0])
    assert not a.requires_grad and not b.requires_grad


def test_joint_frame_transform_and_shape_identity():
    reward,x,a,b=fixture();other=copy.deepcopy(reward.reference)
    shift=torch.tensor([3.,-2.,4.],dtype=x.dtype)
    for f in other['frames']:f['coords']=(np.array(f['coords'])+shift.numpy()).tolist()
    r2=WindowReward(reward.program,other)
    torch.testing.assert_close(reward(x,a,b,.3)[0],r2(x+shift,a,b,.3)[0])
    y=x.new_tensor([other['frames'][0]['coords']])
    assert r2(y,a,b,.3)[0].abs().max()<1e-12


def test_source_adapter_disables_upstream_selection_assertion():
    path=Path('data/multistage_comparison_generated/results/comparison_v1/provenance/upstream_generate_selective.py')
    if not path.exists():pytest.skip('Local preserved upstream source not installed')
    src=no_selection_source(path.read_text())
    assert 'assert not apply_guidance' in src
    assert 'self._gradient.after_native(curr)' in src
    assert 'if (\n                apply_guidance' in src
