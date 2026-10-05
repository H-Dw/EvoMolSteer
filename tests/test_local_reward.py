import copy
import numpy as np
import torch
import pytest
from evomolsteer.generation.local_reward import LocalIntervalReward,bounded_local_step
from evomolsteer.generation.terminal_evaluation import energy_relaxation
from rdkit import Chem
from rdkit.Chem import AllChem


def fixture():
    features=['a','b'];catalog={'atom_vocabulary':{'C':0,'N':1,'O':2,'S':3},
       'features':{k:{'region':k,'elements':['N','O','S'],'temperature_A':.5} for k in features},
       'regions':{'a':{'points_A':[[0,0,0]]},'b':{'points_A':[[1,0,0]]}}}
    ref={'window':[.2,.4],'times':[.2,.3,.4],'features':features,'catalog':catalog,
         'frames':[{'center_A':[2,2],'covariance_A2':[[1,.2],[.2,1]],'radius_squared':1} for t in range(3)]}
    reward=LocalIntervalReward({'window':[.2,.4],'native_rms_ratio':.1},ref)
    return reward


def test_acceptable_set_and_dynamic_window():
    r=fixture();z=torch.tensor([[2.,2.],[5.,5.]],dtype=torch.float64,requires_grad=True)
    value,detail=r.from_features(z,torch.tensor([True,True]),.3)
    g,=torch.autograd.grad(value.sum(),z)
    assert value[0]==0 and bool((g[0]==0).all()) and detail['dose_gate'][0]==0
    assert value[1]<0 and bool((g[1]<0).all()) and 0<detail['dose_gate'][1]<1
    assert r.active(.2,.3) and not r.active(.1,.2) and not r.active(.4,.5)
    with pytest.raises(ValueError):r.reference_at(.31,z)


def test_gradient_and_absent_types():
    r=fixture();x=torch.tensor([[[5.,1.,.2],[6.,.4,.1]]],dtype=torch.float64,requires_grad=True)
    atoms=torch.tensor([[1,2]]);mask=torch.ones((1,2),dtype=torch.bool)
    v,_=r(x,atoms,mask,.3);g,=torch.autograd.grad(v.sum(),x);d=g/g.norm();e=1e-5
    numeric=(r(x+e*d,atoms,mask,.3)[0]-r(x-e*d,atoms,mask,.3)[0])/(2*e)
    torch.testing.assert_close(numeric.sum(),(g*d).sum(),rtol=1e-6,atol=1e-8)
    v,detail=r(x,torch.zeros_like(atoms),mask,.3)
    assert not bool(detail['available'][0]) and v[0]==0 and detail['dose_gate'][0]==0
    perm=torch.tensor([1,0]);torch.testing.assert_close(r(x,atoms,mask,.3)[0],r(x[:,perm],atoms[:,perm],mask,.3)[0])


def test_dose_vanishes_at_target_and_honors_cap():
    g=torch.ones(2,3,3);native=torch.ones_like(g)*.1;mask=torch.ones(2,3,dtype=torch.bool)
    step,detail=bounded_local_step(g,native,mask,torch.tensor([0.,.5]),.1,1.,.025,torch.ones(2))
    assert step[0].count_nonzero()==0 and float(step.norm(dim=-1).max())<=.02500001
    assert detail['requested_rms_A'][0]==0


def test_energy_does_not_modify_generated_coordinates():
    m=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(m,randomSeed=42)
    before=m.GetConformer().GetPositions().copy()
    result=energy_relaxation(m,np.array([True,False,False]))
    np.testing.assert_array_equal(before,m.GetConformer().GetPositions())
    assert result['energy_status'] in ('converged','not_converged')
    assert result['mmff_relief_kcal_mol']>=-1e-6


def test_structural_check_rejects_broken_bond_geometry():
    from posebusters import PoseBusters
    m=Chem.AddHs(Chem.MolFromSmiles('CCO'));AllChem.EmbedMolecule(m,randomSeed=42)
    conf=m.GetConformer();position=conf.GetAtomPosition(0);position.x+=50;conf.SetAtomPosition(0,position)
    checks=PoseBusters(config='mol_fast').bust(mol_pred=m)
    assert not bool(checks.to_numpy(dtype=bool).all())
