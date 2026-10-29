"""The online launcher must enforce the current protocol before starting work."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

pytest.importorskip('torch')
pytest.importorskip('rdkit')
from agenticgen.uniguide.absolute_coordinate import prepare

BASE=Path(__file__).resolve().parents[2]
path=BASE/'docs/experiments/20261030_uncapped_coordinate/run_campaign.py'
spec=importlib.util.spec_from_file_location('uncapped_campaign',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.fixture
def declared(tmp_path):
    config=json.loads((BASE/'configs/uniguide_absolute_coordinate.json').read_text(encoding='utf-8'))
    config.update(root=str(tmp_path/'run'),generation_context_root=str(tmp_path/'source'))
    protein=tmp_path/'protein.pdb'
    points=[(0.,0.,0.),(4.,0.,0.),(0.,5.,0.),(0.,0.,6.)]
    protein.write_text(''.join(f'ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n'
        for i,(x,y,z) in enumerate(points,1)))
    ids=[f'atom_{i:04d}' for i in range(9)];xyz=[[i*1.4,0.,0.] for i in range(9)]
    condition=prepare(dict(atoms=[dict(id=i,element='C') for i in ids],
        bonds=[dict(atom_ids=[ids[i],ids[i+1]],order='single') for i in range(8)],
        controlled_atom_ids=ids,coordinates_angstrom=xyz,coordinate_frame='generation_context_angstrom'),protein)
    subjects=[dict(teacher=dict(target=t,node_id=n,core_sdf_atoms=list(range(9)),core_A=xyz),
                   condition=condition) for t,n in module.TEACHERS.items()]
    return config,subjects


def test_accepts_five_round_current_uncapped_and_upper_tail_contract(declared):
    config,subjects=declared
    actual=module.validate_scope(config,subjects)
    assert actual['agent_tools']['skill_release']=='2.5.0'
    assert actual['rmsd_outlier_policy']['maximum_excluded_fraction']==.1
    assert 'maximum_single_atom_guidance_increment_angstrom' not in actual


@pytest.mark.parametrize('mutation',['rounds','chemistry','recovery','teacher','coordinates','old_skills','cap'])
def test_rejects_scope_drift_before_running_any_stage(declared,mutation):
    config,subjects=copy.deepcopy(declared)
    if mutation=='rounds':config['rounds']=4
    if mutation=='chemistry':config['chemical_inpainting']['enabled']=False
    if mutation=='recovery':config['sampling_recovery']={'authorization_path':'old'}
    if mutation=='teacher':subjects[0]['teacher']['node_id']='wrong'
    if mutation=='coordinates':subjects[0]['teacher']['core_A'][0][0]+=1
    if mutation=='old_skills':config['agent_tools']['skill_release']='2.4.0'
    if mutation=='cap':config['maximum_single_atom_guidance_increment_angstrom']=3.
    with pytest.raises(ValueError):module.validate_scope(config,subjects)
