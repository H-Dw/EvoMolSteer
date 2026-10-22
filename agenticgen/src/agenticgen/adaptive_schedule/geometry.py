"""Coordinate metrics with explicit frames and chemically constrained diagnostics."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def xyz(value):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Finite nonempty N by 3 coordinates required')
    return a


def protein_transform(moving, reference, atom_ids, *, max_residual_A=.01):
    """Backbone correspondence is supplied, never inferred from ligand geometry.

    .01 A is only a same-rigid-frame identity check, not a pose success tolerance.
    """
    a, b = xyz(moving), xyz(reference)
    if a.shape != b.shape or len(atom_ids) != len(a) or len(set(atom_ids)) != len(a):
        raise ValueError('Unique corresponding protein backbone atoms required')
    if any(str(i).split(':')[-1] not in {'N', 'CA', 'C', 'O'} for i in atom_ids):
        raise ValueError('Use predeclared corresponding protein backbone N/CA/C/O atoms')
    ac, bc = a-a.mean(0), b-b.mean(0)
    if len(a) < 3 or np.linalg.matrix_rank(ac) < 2 or np.linalg.matrix_rank(bc) < 2:
        raise ValueError('Protein fit requires noncollinear correspondence')
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.eye(3); correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    translation = b.mean(0)-a.mean(0) @ rotation
    residual = np.linalg.norm(a @ rotation+translation-b, axis=1)
    if residual.max() > max_residual_A:
        raise ValueError('Protein frames are not related by the declared proper rigid transform')
    return {'rotation_row_vectors': rotation.tolist(), 'translation_A': translation.tolist(),
            'protein_atom_ids': list(atom_ids), 'max_residual_A': float(residual.max()),
            'determinant': float(np.linalg.det(rotation)), 'fit_target': 'protein_backbone_only'}


def transform(points, frame):
    return xyz(points) @ np.asarray(frame['rotation_row_vectors']) + np.asarray(frame['translation_A'])


def audit_molecule_state(mol, coords, elements, charges, bonds, bond_names, tolerance_A=.001):
    """Verify SDF heavy atoms against native terminal coordinates and categorical graph.

    The small tolerance allows SDF decimal serialization only, not pose perturbation.
    Returns the native atom index for each SDF atom; never optimizes coordinates.
    """
    from rdkit import Chem
    x=xyz(coords); m=xyz(mol.GetConformer().GetPositions())
    if len(x)!=len(m) or len(elements)!=len(x) or len(charges)!=len(x):
        raise ValueError('SDF/native heavy atom count mismatch')
    mapping=site_geometry(x,m)['assignment']
    if np.linalg.norm(x[mapping]-m,axis=1).max()>tolerance_A:
        raise ValueError('SDF/native terminal coordinates disagree')
    # Native classes can encode Kekule single/double bonds while the SDF reader
    # perceives aromatic bonds. Normalize a separate native graph using the same
    # RDKit aromaticity model; coordinates and the generated molecule are untouched.
    native=Chem.RWMol()
    for element,charge in zip(elements,charges):
        atom=Chem.Atom(element);atom.SetFormalCharge(int(charge));native.AddAtom(atom)
    bond_types={'single':Chem.BondType.SINGLE,'double':Chem.BondType.DOUBLE,
                'triple':Chem.BondType.TRIPLE,'aromatic':Chem.BondType.AROMATIC}
    for i in range(len(elements)):
        for j in range(i):
            name=bond_names[int(bonds[i][j])]
            if name!='none':native.AddBond(i,j,bond_types[name])
    native=native.GetMol()
    try: Chem.SanitizeMol(native)
    except Exception as exc: raise ValueError('Native covalent graph cannot be sanitized for comparison') from exc
    if any(native.GetAtomWithIdx(i).GetFormalCharge()!=int(q) for i,q in enumerate(charges)):
        raise ValueError('Native graph sanitization changed formal charge')
    for i,j in enumerate(mapping):
        atom=mol.GetAtomWithIdx(i)
        if atom.GetSymbol()!=elements[j] or atom.GetFormalCharge()!=charges[j]:
            raise ValueError('SDF/native elements or formal charge disagree')
        for a,b in enumerate(mapping[:i]):
            bond=mol.GetBondBetweenAtoms(i,a)
            native_bond=native.GetBondBetweenAtoms(int(j),int(b))
            expected='none' if native_bond is None else str(native_bond.GetBondType()).lower()
            actual='none' if bond is None else str(bond.GetBondType()).lower()
            if expected!=actual: raise ValueError('SDF/native covalent graph disagrees')
    return mapping


def site_geometry(current, teacher, active_mask=None):
    """Injective geometry-only terminal match in a shared protein frame."""
    x, t = xyz(current), xyz(teacher)
    active = np.arange(len(x)) if active_mask is None else np.flatnonzero(active_mask)
    if active_mask is not None and len(active_mask) != len(x):
        raise ValueError('Mask length mismatch')
    if len(active) < len(t):
        raise ValueError('Insufficient active atoms for condition correspondence')
    cost = ((t[:, None]-x[active][None])**2).sum(-1)
    rows, columns = linear_sum_assignment(cost)
    if list(rows) != list(range(len(t))):
        raise ValueError('Incomplete condition assignment')
    matched = x[active[columns]]
    return {'assignment': active[columns].tolist(), 'matched_A': matched.tolist(),
            'rmsd_A': float(np.sqrt(np.mean(np.sum((matched-t)**2, axis=1)))),
            'alignment': 'protein_fixed_frame_no_ligand_fit', 'mapping': 'geometry_only_not_chemical_identity'}


def chemical_pose(mol, teacher, teacher_atoms):
    """Exact labeled graph (or specified teacher subgraph), no favorable MCS.

    All graph/symmetry mappings are enumerated up to a bounded limit; exceeding
    the limit returns N/A rather than a purported best among truncated mappings.
    """
    from rdkit import Chem
    if mol is None or teacher is None or not mol.GetNumConformers() or not teacher.GetNumConformers():
        return {'status': 'not_available', 'reason': 'Missing chemical structure or conformer'}
    ids = list(teacher_atoms)
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= teacher.GetNumAtoms() for i in ids):
        raise ValueError('Invalid predefined teacher subgraph')
    # RDKit molecule substructure matching preserves elemental/bond/stereo constraints.
    query = Chem.RWMol(teacher)
    for i in reversed(range(teacher.GetNumAtoms())):
        if i not in ids:
            query.RemoveAtom(i)
    query = query.GetMol(); ids = sorted(ids)
    matches = mol.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=10001)
    if len(matches) > 10000:
        return {'status': 'not_available', 'reason': 'Exact mapping enumeration limit exceeded'}
    # Explicit atomic number/charge checks guard aromatic or permissive query matches.
    matches = [m for m in matches if all(
        mol.GetAtomWithIdx(j).GetAtomicNum() == teacher.GetAtomWithIdx(i).GetAtomicNum()
        and mol.GetAtomWithIdx(j).GetFormalCharge() == teacher.GetAtomWithIdx(i).GetFormalCharge()
        for i, j in zip(ids, m))]
    def exact_induced(m):
        for a,i in enumerate(ids):
            for b,j in enumerate(ids[:a]):
                tb=teacher.GetBondBetweenAtoms(i,j); cb=mol.GetBondBetweenAtoms(m[a],m[b])
                if (tb is None) != (cb is None): return False
                if tb is not None and tb.GetBondType() != cb.GetBondType(): return False
        return True
    matches=[m for m in matches if exact_induced(m)]
    if not matches:
        return {'status': 'not_available', 'reason': 'No chemically legal predefined teacher subgraph mapping'}
    x = np.asarray(mol.GetConformer().GetPositions()); t = np.asarray(teacher.GetConformer().GetPositions())[ids]
    mapping = min(matches, key=lambda m: (float(np.sum((x[list(m)]-t)**2)), m))
    a = x[list(mapping)]
    out = {'status': 'available', 'teacher_atoms': ids, 'candidate_atoms': list(mapping),
           'pose_rmsd_A': float(np.sqrt(np.mean(np.sum((a-t)**2, axis=1)))),
           'alignment': 'protein_fixed_frame_no_ligand_fit', 'shape_rmsd_A': None}
    ac, tc = a-a.mean(0), t-t.mean(0)
    if len(ids) >= 3 and np.linalg.matrix_rank(ac) >= 2 and np.linalg.matrix_rank(tc) >= 2:
        u, _, vt = np.linalg.svd(ac.T @ tc); d = np.eye(3); d[-1, -1] = np.linalg.det(u @ vt)
        out['shape_rmsd_A'] = float(np.sqrt(np.mean(np.sum((ac @ u @ d @ vt-tc)**2, axis=1))))
        out['shape_note'] = 'Optional ligand-fit shape diagnostic using pose correspondence; never primary'
    else:
        out['shape_note'] = 'Unavailable: fewer than three noncollinear corresponding atoms'
    return out
