"""One-off input preparation for the user-authorized two-teacher formal run."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rdkit import Chem
from agenticgen.uniguide.geometry_condition import from_molecule
from agenticgen.uniguide.geometry_runtime import prepare, validate_config
from agenticgen.uniguide.io import frozen, read, sha, write


def main():
    repo, root, input_dir, config_path = map(Path, sys.argv[1:5])
    if root.exists() or input_dir.exists() or config_path.exists():
        raise ValueError("Fresh experiment/input/config paths required; inspect existing state explicitly")
    source = Path('/workspace/agenticgen_uniguide_flowr_20261026')
    catalog = {t['target']: t for t in read(source/'teachers.json')}
    expected = {'6pyy_H7S_A_402': 'single_b004_final_015', '3w54_RNB_A_504': 'single_b000_final_003'}
    config = read(repo/'agenticgen/configs/uniguide_geometry.json')
    config.update(root=str(root), rounds=5, generation_context_root=str(source))
    config['plip_dependencies'] = read(source/'config.json')['plip_dependencies']
    protected = {str(p): sha(p) for p in [source/'teachers.json', source/'config.json', source/'attempts.json']}
    inputs = []
    input_dir.mkdir(parents=True)
    for item in config['conditions']:
        teacher = catalog[item['teacher_id']]
        assert teacher['node_id'] == expected[teacher['target']]
        mol = next(m for m in Chem.SDMolSupplier(teacher['sdf'], removeHs=True) if m is not None)
        assert mol.GetNumConformers() == 1 and mol.GetConformer().Is3D()
        ids = [str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d' % a.GetIdx() for a in mol.GetAtoms()]
        controls = [ids[i] for i in teacher['core_sdf_atoms']]
        assert len(controls) == 9 and len(set(controls)) == 9
        features = from_molecule(mol, controls, interact_atoms=[], coordinate_frame='generation_context_angstrom')
        assert features['coordinates_angstrom'] == mol.GetConformer().GetPositions().tolist()
        destination = input_dir/(teacher['target']+'.json')
        frozen(destination, features)
        item['features_path'] = str(destination)
        paths = [Path(teacher[k]) for k in ['sdf', 'protein', 'native_config']]
        paths += sorted((source/'initials'/teacher['target']).glob('batch_*.pt.gz'))
        paths += [source/'R0'/teacher['target']/'checkpoint_010.pt.gz']
        protected.update({str(p): sha(p) for p in paths})
        inputs.append(dict(teacher=teacher['target'], node_id=teacher['node_id'], core_sdf_atoms=teacher['core_sdf_atoms'],
                           heavy_atoms=mol.GetNumAtoms(), source_sdf=teacher['sdf'], features_sha256=sha(destination),
                           source='original teacher SDF conformer', conformer_reembedded=False, interact_atoms=[]))
    config = validate_config(config)
    frozen(config_path, config)
    subjects = prepare(config)
    frozen(root/'SOURCE_PRESERVATION.json', protected)
    frozen(root/'INPUT_AUDIT.json', dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        inputs=inputs, config_sha256=sha(config_path), independent_teacher_calibration=True,
        copied_historical_scores=False, requested_rounds_per_teacher=5,
        loss_weights=[s['condition']['guidance_loss_weights'] for s in subjects]))
    print(json.dumps({'prepared': str(root), 'config': str(config_path), 'inputs': inputs}, ensure_ascii=True))


if __name__ == '__main__':
    main()
