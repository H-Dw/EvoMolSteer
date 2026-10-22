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
