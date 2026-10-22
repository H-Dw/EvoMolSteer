import copy
import pytest
from agenticgen.adaptive_schedule.io import write, sha
from agenticgen.adaptive_schedule.teacher_evidence import teacher_molecule


def fixture(tmp_path):
    Chem=pytest.importorskip('rdkit.Chem')
    mol=Chem.MolFromSmiles('CO');conf=Chem.Conformer(2)
    conf.SetAtomPosition(0,(0,0,0));conf.SetAtomPosition(1,(1.4,0,0));mol.AddConformer(conf)
    sdf=tmp_path/'teacher.sdf'
    with Chem.SDWriter(str(sdf)) as w:w.write(mol)
    write(tmp_path/'historical_references.json',{str(sdf):sha(sdf)})
    write(tmp_path/'PREPARED.json',{'historical_references_sha256':sha(tmp_path/'historical_references.json')})
    template={'sdf':str(sdf),'source_sdf_sha256':'original_batch_not_extracted_file',
        'coords_A':[[0,0,0],[1.4,0,0]],'atom_vocabulary':{'C':0,'O':1},
        'charge_vocabulary':{'0':0},'bond_vocabulary':{'none':0,'single':1},
        'atom_labels':[0,1],'charge_labels':[0,0],'bond_labels':[[0,1],[1,0]],
        'native_to_sdf':[0,1],'sdf_to_native':[0,1]}
    return template


def test_extracted_reference_is_distinct_from_source_batch(tmp_path):
    t=fixture(tmp_path)
    mol,digest=teacher_molecule(tmp_path/'wave',t)
    assert mol.GetNumAtoms()==2 and digest==sha(t['sdf'])!=t['source_sdf_sha256']
    with open(t['sdf'],'a') as f:f.write('\n')
    with pytest.raises(ValueError,match='Extracted teacher SDF changed'):teacher_molecule(tmp_path,t)


def test_reference_manifest_and_native_graph_remain_checked(tmp_path):
    t=fixture(tmp_path)
    bad=copy.deepcopy(t);bad['bond_labels']=[[0,0],[0,0]]
    with pytest.raises(ValueError,match='covalent graph'):teacher_molecule(tmp_path,bad)
    bad=copy.deepcopy(t);bad['coords_A'][1][0]=2
    with pytest.raises(ValueError,match='coordinates'):teacher_molecule(tmp_path,bad)
    write(tmp_path/'historical_references.json',{})
    with pytest.raises(ValueError,match='not bound'):teacher_molecule(tmp_path,t)


def test_full_diagnostic_accepts_native_float32_coordinates(tmp_path,monkeypatch):
    import sys, types
    import numpy as np
    from agenticgen.adaptive_schedule.wave_retention import chemical_diagnostics
    t=fixture(tmp_path);t['core_atoms']=[0,1]
    protein=tmp_path/'protein.pdb';protein.write_text('fixed protein test reference')
    t['protein']=str(protein)
    class Tensor:
        def __init__(self,value):self.array=np.asarray(value)
        def __getitem__(self,key):return Tensor(self.array[key])
        def __iter__(self):return iter(self.array)
        def __int__(self):return int(self.array)
        def argmax(self,axis):return Tensor(self.array.argmax(axis))
        def numpy(self):return self.array
    output={'coords':Tensor(np.asarray([t['coords_A']],dtype=np.float32)),
        'atomics':Tensor([np.eye(2)]),'charges':Tensor(np.ones((1,2,1))),
        'bonds':Tensor(np.eye(2)[np.asarray([t['bond_labels']])]),'mask':Tensor([[True,True]])}
    stub=types.ModuleType('agenticgen.adaptive_schedule.flowr');stub.load_pt=lambda path:output
    monkeypatch.setitem(sys.modules,'agenticgen.adaptive_schedule.flowr',stub)
    mol,digest=teacher_molecule(tmp_path,t)
    result=chemical_diagnostics(tmp_path/'wave',t,{0:mol},1)[0]
    assert result['status']=='available' and result['pose_rmsd_A']<1e-6
    assert result['teacher_sdf_sha256']==digest
    assert result['source_batch_sdf_sha256']==t['source_sdf_sha256']
    assert result['shape_rmsd_A'] is None
