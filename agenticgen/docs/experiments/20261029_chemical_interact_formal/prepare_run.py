"""Freeze the same PLIP ligand sites as condition atoms and interaction atoms."""
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rdkit import Chem
from agenticgen.uniguide.geometry_condition import from_molecule
from agenticgen.uniguide.geometry_runtime import prepare, validate_config
from agenticgen.uniguide.io import frozen, read, sha, value_sha


def interaction_features(teacher, atom_ids, protein_text):
    """Use original PDB serials; expand only atom sets in a recorded PLIP contact."""
    atoms = {}
    for line in protein_text.splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        serial = int(line[6:11])
        if serial in atoms:
            raise ValueError('Protein serial must identify one atom')
        xyz = [float(line[i:i+8]) for i in (30, 38, 46)]
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError('Nonfinite protein coordinate')
        atoms[serial] = dict(id=f'protein_serial_{serial}', coordinates_angstrom=xyz)
    profile = teacher['plip_profile']
    controls = set(teacher['core_sdf_atoms'])
    if controls != set(profile['core_atoms']):
        raise ValueError('Frozen PLIP core differs from teacher controls')
    selected, contacts = {}, []
    for number, contact in enumerate(profile['interactions']):
        ligands, partners = contact['ligand_atoms'], contact['protein_pdb_serials']
        if not ligands or not partners or not set(ligands) <= controls:
            raise ValueError('PLIP contact is empty or outside the approved core')
        if any(i not in atoms for i in partners):
            raise ValueError('PLIP target serial missing from original protein')
        if [profile['pdb_serial_to_native'][str(s)] for s in contact['ligand_pdb_serials']] != ligands:
            raise ValueError('PLIP ligand identity mapping differs')
        expected = contact['details'].get('PROTCOO')
        centroid = [sum(atoms[p]['coordinates_angstrom'][j] for p in partners)/len(partners) for j in range(3)]
        if expected is None or math.dist(expected, centroid) > 1e-6:
            raise ValueError('PLIP target coordinates differ from original PDB serials')
        for i in ligands:
            if type(i) is not int or not 0 <= i < len(atom_ids):
                raise ValueError('PLIP ligand index outside template')
            selected.setdefault(atom_ids[i], set()).update(partners)
        contacts.append(dict(contact_index=number, type=contact['type'],
            ligand_atom_ids=[atom_ids[i] for i in ligands], protein_serials=partners,
            pair_rule='Cartesian product within this recorded interaction; not independent PLIP events'))
    if set(selected) != {atom_ids[i] for i in controls}:
        raise ValueError('condition_atom and interact_atom sets must be identical')
    used = sorted({p for group in selected.values() for p in group})
    interact = [dict(atom_id=a, target_atom_ids=[atoms[p]['id'] for p in sorted(selected[a])]) for a in sorted(selected)]
    return dict(interact_atoms=interact, target_atoms=[atoms[p] for p in used],
        coordinate_frame='generation_context_angstrom'), dict(contacts=contacts,
        unique_ligand_target_pairs=sum(len(v) for v in selected.values()),
        protein_serials=used, duplicate_pairs_removed=True, expanded_beyond_recorded_contact=False)


def main():
    repo, root, inputs, config_path = map(Path, sys.argv[1:5])
    if any(p.exists() for p in (root, inputs, config_path)):
        raise ValueError('Fresh experiment, input and config paths required')
    source = Path('/workspace/agenticgen_uniguide_flowr_20261026')
    catalog = {t['target']: t for t in read(source/'teachers.json')}
    expected = {'6pyy_H7S_A_402': 'single_b004_final_015', '3w54_RNB_A_504': 'single_b000_final_003'}
    config = read(repo/'agenticgen/configs/uniguide_geometry.json')
    config.update(root=str(root), rounds=5, generation_context_root=str(source), chemical_inpainting={'enabled': True})
    config['plip_dependencies'] = read(source/'config.json')['plip_dependencies']
    protected = {str(p): sha(p) for p in [source/'teachers.json', source/'config.json', source/'attempts.json']}
    inputs.mkdir(parents=True)
    audits = []
    for item in config['conditions']:
        teacher = catalog[item['teacher_id']]
        assert teacher['node_id'] == expected[teacher['target']]
        assert sha(teacher['protein']) == teacher['protein_sha256'] == teacher['plip_profile']['protein_sha256']
        mols = [m for m in Chem.SDMolSupplier(teacher['sdf'], removeHs=True) if m is not None]
        assert len(mols) == 1 and mols[0].GetProp('node_id') == teacher['node_id']
        mol = mols[0]
        assert mol.GetNumConformers() == 1 and mol.GetConformer().Is3D()
        ids = [str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d' % a.GetIdx() for a in mol.GetAtoms()]
        controls = [ids[i] for i in teacher['core_sdf_atoms']]
        assert len(set(controls)) == 9
        options, audit = interaction_features(teacher, ids, Path(teacher['protein']).read_text())
        features = from_molecule(mol, controls, **options)
        assert features['coordinates_angstrom'] == mol.GetConformer().GetPositions().tolist()
        path = inputs/(teacher['target']+'.json')
        frozen(path, features);item['features_path'] = str(path)
        paths = [Path(teacher[k]) for k in ('sdf', 'protein', 'native_config')]
        paths += sorted((source/'initials'/teacher['target']).glob('batch_*.pt.gz'))
        paths += [source/'R0'/teacher['target']/'checkpoint_010.pt.gz']
        protected.update({str(p): sha(p) for p in paths})
        audits.append(dict(teacher=teacher['target'], node_id=teacher['node_id'],
            condition_atom_ids=sorted(controls), interact_atom_ids=[x['atom_id'] for x in options['interact_atoms']],
            teacher_profile_sha256=value_sha(teacher['plip_profile']), features_sha256=sha(path),
            conformer_reembedded=False, interaction_mapping=audit))
    old = Path('/workspace/agenticgen_template_distance_20261029_777bfa36')
    protected.update({str(p): sha(p) for p in old.rglob('*') if p.is_file()})
    config = validate_config(config);frozen(config_path, config)
    subjects = prepare(config)
    for s in subjects:
        c = s['condition']
        assert set(c['controlled_atom_ids']) == {i['atom_id'] for i in c['interact_atoms']}
        assert c['guidance_loss_weights'] == {'condition_intra_weight': 1, 'condition_outer_weight': 0, 'interact_weigh': 1}
    frozen(root/'SOURCE_PRESERVATION.json', protected)
    frozen(root/'INPUT_AUDIT.json', dict(timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        inputs=audits, protected_files=len(protected), requested_rounds_per_teacher=5,
        initial_loss_weights=[s['condition']['guidance_loss_weights'] for s in subjects],
        interaction_pair_counts={s['teacher']['target']: len(s['condition']['interaction_pairs']) for s in subjects},
        old_calibration_reused=False, chemical_state_inpainting=True, coordinate_inpainting=False))
    import json
    print(json.dumps(read(root/'INPUT_AUDIT.json'), ensure_ascii=True))


if __name__ == '__main__':main()
