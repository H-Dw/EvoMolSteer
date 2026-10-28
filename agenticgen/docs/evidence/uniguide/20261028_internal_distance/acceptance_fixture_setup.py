import json,sys,datetime,subprocess
from pathlib import Path
from rdkit import Chem
from agenticgen.uniguide.geometry_condition import from_molecule
from agenticgen.uniguide.io import sha,write

checkout=Path(sys.argv[1]).resolve()
commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
base=checkout.parent/(checkout.name+'-inputs')
base.mkdir(exist_ok=True)
source=Path('/workspace/agenticgen_uniguide_flowr_20261026')
config=json.loads((checkout/'agenticgen/configs/uniguide_geometry.json').read_text())
config.update(root='/workspace/agenticgen_geometry_acceptance_20261028_'+commit[:8],rounds=2)
conditions=[];protected=[source/'attempts.json',source/'teachers.json',source/'config.json']
for teacher in json.loads((source/'teachers.json').read_text()):
    mol=next(m for m in Chem.SDMolSupplier(teacher['sdf'],removeHs=True) if m is not None)
    atoms=[str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d'%a.GetIdx() for a in mol.GetAtoms()]
    controls=[atoms[i] for i in teacher['core_sdf_atoms']]
    mol.RemoveAllConformers()
    features=from_molecule(mol,controls)
    assert 'coordinates_angstrom' not in features
    path=base/(teacher['target']+'.json')
    write(path,features)
    conditions.append(dict(teacher_id=teacher['target'],features_path=str(path)))
    protected.extend([Path(teacher['sdf']),Path(teacher['native_config']),source/'R0'/teacher['target']/'checkpoint_010.pt.gz'])
    protected.extend(sorted((source/'initials'/teacher['target']).glob('batch_*.pt.gz')))
config['conditions']=conditions
path=checkout.parent/(checkout.name+'-config.json')
write(path,config)
write(checkout.parent/(checkout.name+'-source-before.json'),dict(
    time=datetime.datetime.now().astimezone().isoformat(),checkout_commit=commit,
    setup_script_sha256=sha(__file__),source_hashes={str(p):sha(p) for p in protected},
    fixture='two original molecular graphs with all conformers removed; independent ETKDG reference generation',
    config_sha256=sha(path),no_llm_molecule_editing=True))
print(json.dumps(dict(config=str(path),root=config['root'],teachers=[c['teacher_id'] for c in conditions],protected_files=len(protected))))
