import copy
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[2]/'docs/experiments/20261029_chemical_interact_formal/prepare_run.py'
spec=importlib.util.spec_from_file_location('chemical_interact_inputs',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture():
    def atom(serial,xyz):
        return f'ATOM  {serial:5d}  CA  ALA A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C'
    text='\n'.join([atom(11,[0,0,2]),atom(12,[0,2,2]),atom(13,[99,99,99])])
    contact=dict(type='pistacking',ligand_atoms=[0,1],ligand_pdb_serials=[101,102],
        protein_pdb_serials=[11,12],details={'PROTCOO':[0,1,2]})
    teacher=dict(core_sdf_atoms=[0,1],plip_profile=dict(core_atoms=[0,1],
        pdb_serial_to_native={'101':0,'102':1},interactions=[contact,copy.deepcopy(contact)]))
    return teacher,['a','b','c'],text


def test_same_atom_sets_only_recorded_partners_and_unique_pairs():
    teacher,ids,text=fixture();features,audit=module.interaction_features(teacher,ids,text)
    assert {a['atom_id'] for a in features['interact_atoms']}=={'a','b'}
    assert {a['id'] for a in features['target_atoms']}=={'protein_serial_11','protein_serial_12'}
    assert audit['unique_ligand_target_pairs']==4 and len(audit['contacts'])==2
    assert all(len(a['target_atom_ids'])==2 for a in features['interact_atoms'])


@pytest.mark.parametrize('mutation', ['missing_target','wrong_coordinates','wrong_mapping','uncovered_control','duplicate_serial'])
def test_reject_ambiguous_or_inconsistent_source(mutation):
    teacher,ids,text=fixture()
    if mutation=='missing_target':teacher['plip_profile']['interactions'][0]['protein_pdb_serials']=[88]
    if mutation=='wrong_coordinates':teacher['plip_profile']['interactions'][0]['details']['PROTCOO']=[0,0,0]
    if mutation=='wrong_mapping':teacher['plip_profile']['pdb_serial_to_native']['101']=1
    if mutation=='uncovered_control':
        teacher['core_sdf_atoms'].append(2);teacher['plip_profile']['core_atoms'].append(2)
    if mutation=='duplicate_serial':text+='\n'+text.splitlines()[0]
    with pytest.raises(ValueError):module.interaction_features(teacher,ids,text)
