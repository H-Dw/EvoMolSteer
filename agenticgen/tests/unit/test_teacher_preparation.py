import copy
from rdkit import Chem
import numpy as np
import pytest
from agenticgen.uniguide import teacher_preparation as teacher


def molecule(smiles):
    mol=Chem.MolFromSmiles(smiles)
    mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()));mol.GetConformer().Set3D(True)
    for i in range(mol.GetNumAtoms()):mol.GetConformer().SetAtomPosition(i,(i,.1*i*i,.02*i))
    return mol


@pytest.mark.parametrize('smiles',['O=c1[nH]c2cc(O)c(O)cc2c2ccccc12','c1cc[nH]c1','c1ccncc1','c1ncc[nH]1',
    'c1ccc2[nH]ccc2c1','[NH4+]','C[NH2+]C','[O-]C(=O)c1ccccc1','CCCO'])
def test_promoted_teacher_roundtrip_retains_valence_hydrogens_and_coordinates(smiles):
    mol=molecule(smiles);before=Chem.MolToMolBlock(mol);controlled=['atom_%04d'%i for i in range(min(3,mol.GetNumAtoms()))]
    features,audit=teacher.features(mol,controlled)
    assert features['controlled_atom_ids']==controlled
    assert np.array_equal(np.asarray(features['coordinates_angstrom']),mol.GetConformer().GetPositions())
    assert audit['source']==audit['roundtrip'] and audit['hydrogen_count_preserved']
    assert all(b['order']!='aromatic' for b in features['bonds'])
    assert Chem.MolToMolBlock(mol)==before


def test_unsupported_chemical_roundtrip_fails_instead_of_changing_a_teacher():
    mol=molecule('[13CH3]C')
    with pytest.raises(ValueError,match='serialization changed'):teacher.features(mol,['atom_0000'])


def test_promoted_condition_and_encoder_both_receive_the_same_loss_checked_graph(monkeypatch):
    from agenticgen.uniguide import absolute_coordinate,geometry_chemistry
    seen=[]
    def geometry(f,protein):seen.append(copy.deepcopy(f));return dict(f,condition_sha256='condition')
    def encode(model,condition):
        seen.append({k:v for k,v in condition.items() if k!='condition_sha256'})
        return {},[],{}
    monkeypatch.setattr(absolute_coordinate,'prepare',geometry);monkeypatch.setattr(geometry_chemistry,'condition_features',encode)
    monkeypatch.setattr(teacher,'native_channels',lambda *a:{})
    features,condition,audit=teacher.prepare(None,molecule('c1cc[nH]c1'),['atom_0000'],'protein')
    assert seen==[features,features] and condition['condition_sha256']=='condition'


def test_independent_rerun_seeds_do_not_repeat_consumed_scientific_or_engineering_slots():
    from agenticgen.uniguide.iterative_affinity import seeds
    old={s for r in range(6) for s in seeds(r)}
    new=[s for r in range(1,6) for s in seeds(r,7)]
    assert len(set(new))==25 and not old.intersection(new)
    assert seeds(0,7)==seeds(0)
