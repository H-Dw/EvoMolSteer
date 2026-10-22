"""Verify the extracted teacher against its frozen file reference and native graph."""
from pathlib import Path
from .io import read, sha
from .protocol_state import prepared_path


def teacher_molecule(folder, template):
    from rdkit import Chem
    from .geometry import audit_molecule_state
    folder = Path(folder).resolve()
    root = next((p for p in [folder, *folder.parents]
                 if (p/'historical_references.json').is_file()), None)
    if root is None:
        raise ValueError('Missing experiment historical references')
    prepared = read(prepared_path(root))
    references = root/'historical_references.json'
    if prepared.get('historical_references_sha256') != sha(references):
        raise ValueError('Historical references are not bound to current preparation')
    digest = read(references).get(template['sdf'])
    if not digest or sha(template['sdf']) != digest:
        raise ValueError('Extracted teacher SDF changed before pose diagnostic')
    molecules = list(Chem.SDMolSupplier(template['sdf'], removeHs=True))
    if len(molecules) != 1 or molecules[0] is None:
        raise ValueError('Expected one readable extracted teacher molecule')
    teacher = molecules[0]
    atoms = {v:k for k,v in template['atom_vocabulary'].items()}
    charges = {v:k for k,v in template['charge_vocabulary'].items()}
    bonds = {v:k for k,v in template['bond_vocabulary'].items()}
    mapping = audit_molecule_state(teacher, template['coords_A'],
        [atoms[i] for i in template['atom_labels']],
        [int(charges[i]) for i in template['charge_labels']], template['bond_labels'], bonds)
    if mapping != template['sdf_to_native'] or any(
            template['native_to_sdf'][native] != sdf for sdf,native in enumerate(mapping)):
        raise ValueError('Teacher SDF/native mapping disagrees with frozen template')
    for sdf,native in enumerate(mapping):
        teacher.GetConformer().SetAtomPosition(sdf, template['coords_A'][native])
    return teacher, digest
