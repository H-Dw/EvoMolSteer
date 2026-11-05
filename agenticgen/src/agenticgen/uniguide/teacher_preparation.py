"""Loss-checked graph serialization for promoted, existing 3D teachers."""
import numpy as np
import torch
from rdkit import Chem
from .geometry_condition import from_molecule


def chemistry(mol):
    return dict(smiles=Chem.MolToSmiles(mol,isomericSmiles=False),
        atoms=[dict(element=a.GetSymbol(),charge=a.GetFormalCharge(),hydrogens=a.GetTotalNumHs(includeNeighbors=True),
            radicals=a.GetNumRadicalElectrons(),isotope=a.GetIsotope()) for a in mol.GetAtoms()])


def features(mol,controlled_atom_ids):
    """Keep atom order/coordinates; serialize a Kekule graph before sanitization.

    Aromatic bonds alone cannot distinguish pyridine N from pyrrole [nH] when
    rebuilding the legacy element/charge graph. Explicit single/double valences
    let RDKit recover that distinction. Verify the complete chemical round trip
    instead of dropping a valid candidate or silently changing its protonation.
    The shared inpainting encoder still applies the model's aromaticity policy.
    """
    source=Chem.RemoveHs(Chem.Mol(mol));expected=chemistry(source)
    serialized=Chem.Mol(source);Chem.Kekulize(serialized,clearAromaticFlags=True)
    result=from_molecule(source,controlled_atom_ids,coordinate_frame='generation_context_angstrom')
    # from_molecule's RemoveHs sanitizes and perceives aromaticity again. Read
    # bond orders directly from the separate Kekule copy after that adapter.
    ids=[a['id'] for a in result['atoms']]
    result['bonds']=[dict(atom_ids=[ids[b.GetBeginAtomIdx()],ids[b.GetEndAtomIdx()]],
        order=str(b.GetBondType()).lower()) for b in serialized.GetBonds()]
    rebuilt=Chem.RWMol();lookup={a['id']:i for i,a in enumerate(result['atoms'])}
    for atom in result['atoms']:
        item=Chem.Atom(atom['element']);item.SetFormalCharge(atom['formal_charge']);rebuilt.AddAtom(item)
    orders={'single':Chem.BondType.SINGLE,'double':Chem.BondType.DOUBLE,'triple':Chem.BondType.TRIPLE,'aromatic':Chem.BondType.AROMATIC}
    for bond in result['bonds']:rebuilt.AddBond(*(lookup[i] for i in bond['atom_ids']),orders[bond['order']])
    rebuilt=rebuilt.GetMol();Chem.SanitizeMol(rebuilt)
    if chemistry(rebuilt)!=expected:raise ValueError('Teacher feature serialization changed hydrogen, charge, radical, isotope or graph state')
    xyz=np.asarray(result['coordinates_angstrom'])
    if not np.array_equal(xyz,source.GetConformer().GetPositions()):raise ValueError('Teacher feature serialization moved coordinates')
    return result,dict(representation='loss-checked-kekule-valence-graph',source=expected,roundtrip=chemistry(rebuilt),
        atom_order_preserved=True,coordinates_unchanged=True,hydrogen_count_preserved=True,
        chemical_inpainting_policy_unchanged=True)


def prepare(model,mol,controlled_atom_ids,protein):
    from .absolute_coordinate import prepare as geometry
    from .geometry_chemistry import condition_features
    result,audit=features(mol,controlled_atom_ids)
    condition=geometry(result,protein)
    channels,_,_=condition_features(model,condition)
    expected=native_channels(model,mol,condition['controlled_atom_ids'])
    if set(channels)!=set(expected) or any(not torch.equal(v,expected[k]) for k,v in channels.items()):
        raise ValueError('Promoted teacher encoding differs from its native source molecule')
    audit['model_channels']={k:list(v.shape) for k,v in channels.items()}
    audit['native_source_encoding_matches']=True
    return result,condition,audit


def native_channels(model,mol,controlled_atom_ids):
    """Independent source-molecule reference to the shared inpainting encoder."""
    from flowr.util.rdkit import BOND_IDX_MAP
    source=Chem.RemoveHs(Chem.Mol(mol))
    if model.hparams.get('remove_aromaticity',False):Chem.Kekulize(source,clearAromaticFlags=True)
    ids={str(a.GetAtomMapNum()) if a.GetAtomMapNum() else 'atom_%04d'%a.GetIdx():a.GetIdx() for a in source.GetAtoms()}
    chosen=[source.GetAtomWithIdx(ids[key]) for key in controlled_atom_ids]
    values=dict(atomics=torch.tensor(model.vocab.indices_from_tokens([a.GetSymbol() for a in chosen],one_hot=True)),
        charges=torch.tensor(model.vocab_charges.indices_from_tokens([a.GetFormalCharge() for a in chosen],one_hot=True)))
    if model.add_feats:values['hybridization']=torch.tensor(model.vocab_hybridization.indices_from_tokens([a.GetHybridization() for a in chosen],one_hot=True))
    bonds=torch.zeros((len(chosen),len(chosen)),dtype=torch.long)
    for i,left in enumerate(chosen):
        for j,right in enumerate(chosen):
            bond=source.GetBondBetweenAtoms(left.GetIdx(),right.GetIdx()) if i!=j else None
            if bond is not None:bonds[i,j]=BOND_IDX_MAP[bond.GetBondType()]
    values['bonds']=torch.nn.functional.one_hot(bonds,5)
    return values
