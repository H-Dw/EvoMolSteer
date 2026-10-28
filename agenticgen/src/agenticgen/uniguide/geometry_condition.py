"""Protein-independent, immutable internal-distance conditions from molecular features."""
import math
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from .io import value_sha
from .condition import finite

POLICY = "intramolecular-distance-fixed-correspondence-v1"
DISTANCE = "intramolecular_pair_distance_root_mean_square_deviation_angstrom"
CONTROL_DISTANCE = "controlled_atom_pair_distance_root_mean_square_deviation_angstrom"
NEIGHBOR_DISTANCE = "controlled_to_neighborhood_distance_root_mean_square_deviation_angstrom"
MAX_ERROR = "maximum_atom_pair_distance_error_angstrom"


def from_molecule(mol, controlled_atom_ids, **options):
    """Adapter for an existing optimized RDKit molecule; atom-map numbers are identities."""
    mol = Chem.RemoveHs(Chem.Mol(mol))
    ids = [str(a.GetAtomMapNum()) if a.GetAtomMapNum() else "atom_%04d" % a.GetIdx() for a in mol.GetAtoms()]
    features = dict(atoms=[dict(id=i, element=a.GetSymbol(), formal_charge=a.GetFormalCharge())
        for i,a in zip(ids,mol.GetAtoms())], bonds=[dict(atom_ids=[ids[b.GetBeginAtomIdx()],ids[b.GetEndAtomIdx()]],
        order=str(b.GetBondType()).lower()) for b in mol.GetBonds()], controlled_atom_ids=controlled_atom_ids, **options)
    if mol.GetNumConformers():
        if not mol.GetConformer().Is3D():
            raise ValueError("A supplied conformer must be explicitly three dimensional")
        features["coordinates_angstrom"] = mol.GetConformer().GetPositions().tolist()
    return features


def prepare(features,valid_generated_slots=None):
    """No protein, archive, affinity or old generated coordinate frame is read here."""
    atoms = features["atoms"]
    ids = [a["id"] for a in atoms]
    if not ids or any(not isinstance(i,str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique nonempty string atom identities required")
    indices = {key:i for i,key in enumerate(ids)}
    mol = Chem.RWMol()
    for atom in atoms:
        a = Chem.Atom(atom["element"])
        if a.GetAtomicNum() <= 1: raise ValueError("Condition features must contain heavy atoms only")
        charge = atom.get("formal_charge",0)
        if type(charge) is not int: raise ValueError("Formal charge must be an integer")
        a.SetFormalCharge(charge); mol.AddAtom(a)
    types = {"single":Chem.BondType.SINGLE,"double":Chem.BondType.DOUBLE,
             "triple":Chem.BondType.TRIPLE,"aromatic":Chem.BondType.AROMATIC}
    edges=set()
    for bond in features["bonds"]:
        left,right = [indices[i] for i in bond["atom_ids"]]
        pair=tuple(sorted((left,right)))
        if left==right or pair in edges: raise ValueError("Self or duplicate covalent bond")
        edges.add(pair);mol.AddBond(left,right,types[bond["order"]])
    mol=mol.GetMol();Chem.SanitizeMol(mol)
    controlled=features["controlled_atom_ids"]
    if not controlled or len(set(controlled))!=len(controlled) or not set(controlled)<=set(ids):
        raise ValueError("Invalid controlled atom identities")
    coordinates=features.get("coordinates_angstrom")
    preparation=dict(rdkit_version=rdBase.rdkitVersion,source="provided_conformer",optimization_applied=False)
    if coordinates is None:
        # Hydrogen coordinates improve embedding, but are not guided atom identities.
        hydrated=Chem.AddHs(mol)
        params=AllChem.ETKDGv3();params.randomSeed=42;params.numThreads=1
        preparation["embedding_parameters"]={name:getattr(params,name) for name in dir(params)
            if not name.startswith("_") and type(getattr(params,name)) in (bool,int,float,str)}
        if AllChem.EmbedMolecule(hydrated,params)!=0: raise ValueError("RDKit ETKDGv3 embedding failed")
        coordinates=Chem.RemoveHs(hydrated).GetConformer().GetPositions().tolist()
        preparation.update(source="ETKDGv3",random_seed=42,num_threads=1)
    xyz=np.asarray(coordinates,dtype=np.float64)
    if xyz.shape!=(len(ids),3) or not np.isfinite(xyz).all(): raise ValueError("Finite N by 3 coordinates required")
    explicit=features.get("neighborhood_atom_ids")
    if explicit is not None and (set(explicit)!=set(controlled) or any(
            len(set(v))!=len(v) or not set(v)<=set(ids) or k in v for k,v in explicit.items())):
        raise ValueError("Neighborhood must map every control identity to unique other atom identities")
    graph_distance=Chem.GetDistanceMatrix(mol)
    pairs=set(); categories={}
    for left in controlled:
        neighbors=explicit[left] if explicit is not None else [right for right in ids
            if 0<graph_distance[indices[left],indices[right]]<=2]
        for right in set(neighbors)|set(controlled):
            if left==right: continue
            pair=tuple(sorted((left,right)));pairs.add(pair)
            categories[pair]="controlled_internal" if right in controlled and left in controlled else "controlled_neighborhood"
    if not pairs: raise ValueError("At least one nonself distance constraint required")
    mapping=features.get("atom_to_generated_slot")
    available=list(range(len(ids))) if valid_generated_slots is None else sorted(valid_generated_slots)
    if (any(type(v) is not int or v<0 for v in available) or len(set(available))!=len(available)
            or len(available)<len(ids)):raise ValueError("Insufficient unique valid generation slots")
    if mapping is None: mapping=dict(zip(sorted(ids),available))
    if set(mapping)!=set(ids) or any(type(v) is not int or v<0 for v in mapping.values()) or len(set(mapping.values()))!=len(ids):
        raise ValueError("Atom-to-slot mapping must be a complete nonnegative injection")
    if valid_generated_slots is not None and not set(mapping.values())<=set(available):
        raise ValueError("Explicit mapping references unavailable generation slots")
    rows=[]
    for left,right in sorted(pairs):
        distance=float(np.linalg.norm(xyz[indices[left]]-xyz[indices[right]]))
        if distance<=0: raise ValueError("Distinct reference atoms cannot coincide")
        rows.append(dict(atom_ids=[left,right],generated_slots=[mapping[left],mapping[right]],
            reference_distance_angstrom=distance,category=categories[(left,right)],
            bonded=tuple(sorted((indices[left],indices[right]))) in edges))
    body=dict(policy=POLICY,unit="angstrom",features_sha256=value_sha(features),atoms=atoms,bonds=features["bonds"],
        controlled_atom_ids=sorted(controlled),atom_to_generated_slot=mapping,pairs=rows,
        reference_coordinates_angstrom=xyz.tolist(),conformer_preparation=preparation,
        reference_distance_matrix_angstrom=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).tolist(),
        neighborhood_policy="explicit" if explicit is not None else "at_most_two_covalent_bonds",
        chemistry_locked=False,coordinates_replaced=False)
    return dict(body,condition_sha256=value_sha(body))


def validate(condition):
    body={k:v for k,v in condition.items() if k!="condition_sha256"}
    if condition.get("policy")!=POLICY or value_sha(body)!=condition.get("condition_sha256"):
        raise ValueError("Condition binding changed")
    pairs=condition["pairs"]
    keys=[tuple(sorted(p["atom_ids"])) for p in pairs]
    if not pairs or len(set(keys))!=len(keys) or any(a==b for a,b in keys):
        raise ValueError("Invalid or duplicate distance pairs")
    mapping=condition["atom_to_generated_slot"]
    ids=[a["id"] for a in condition["atoms"]]
    if condition.get("unit")!="angstrom" or not ids or len(ids)!=len(set(ids)) or set(ids)!=set(mapping):
        raise ValueError("Invalid condition identities or units")
    if len(set(mapping.values()))!=len(mapping) or any(type(v) is not int or v<0 for v in mapping.values()):
        raise ValueError("Invalid fixed correspondence")
    controlled=set(condition["controlled_atom_ids"])
    if not controlled or not controlled<=set(ids):raise ValueError("Invalid controls")
    xyz=np.asarray(condition["reference_coordinates_angstrom"],dtype=np.float64)
    if xyz.shape!=(len(ids),3) or not np.isfinite(xyz).all():raise ValueError("Invalid frozen geometry")
    matrix=np.asarray(condition["reference_distance_matrix_angstrom"],dtype=np.float64)
    if matrix.shape!=(len(ids),len(ids)) or not np.allclose(matrix,np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1),rtol=1e-12,atol=1e-12):
        raise ValueError("Invalid frozen reference distance matrix")
    index={key:i for i,key in enumerate(ids)}
    for p in pairs:
        d=p["reference_distance_angstrom"]
        if not isinstance(d,(int,float)) or isinstance(d,bool) or not math.isfinite(d) or d<=0:
            raise ValueError("Invalid reference distance")
        if p["generated_slots"]!=[mapping[i] for i in p["atom_ids"]]: raise ValueError("Pair correspondence differs")
        left,right=p["atom_ids"]
        category="controlled_internal" if {left,right}<=controlled else "controlled_neighborhood"
        if not {left,right}&controlled or p["category"]!=category:raise ValueError("Invalid pair category")
        actual=float(np.linalg.norm(xyz[index[left]]-xyz[index[right]]))
        if not math.isclose(actual,d,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("Reference distance differs from frozen conformer")
    return condition


def measurements(endpoint,condition,valid):
    validate(condition);finite("distance endpoint",endpoint)
    if endpoint.ndim!=3 or endpoint.shape[-1]!=3 or valid.shape!=endpoint.shape[:2]:
        raise ValueError("Invalid coordinate/mask layout")
    all_slots=torch.as_tensor(list(condition["atom_to_generated_slot"].values()),device=endpoint.device)
    if int(all_slots.max())>=endpoint.shape[1] or not bool(valid[:,all_slots].bool().all()):
        raise ValueError("Fixed condition mapping references masked or absent slots")
    pairs=torch.as_tensor([p["generated_slots"] for p in condition["pairs"]],device=endpoint.device)
    reference=endpoint.new_tensor([p["reference_distance_angstrom"] for p in condition["pairs"]]).detach()
    measured=torch.linalg.vector_norm(endpoint[:,pairs[:,0]]-endpoint[:,pairs[:,1]],dim=-1)
    errors=measured-reference
    finite("internal distance errors",errors)
    return errors,reference,pairs


def loss(endpoint,condition,valid):
    errors,reference,pairs=measurements(endpoint,condition,valid)
    slots=torch.unique(pairs.flatten(),sorted=True)
    assignment=slots[None].expand(len(endpoint),-1).detach()
    selected=torch.zeros_like(valid,dtype=torch.bool);selected[:,slots]=True
    return errors.square().mean(-1)/2,reference[None].expand(len(endpoint),-1),selected,assignment


def metrics(endpoint,condition,valid):
    errors,_,_=measurements(endpoint,condition,valid)
    result={DISTANCE:errors.square().mean(-1).sqrt(),MAX_ERROR:errors.abs().amax(-1)}
    for name,category in [(CONTROL_DISTANCE,"controlled_internal"),(NEIGHBOR_DISTANCE,"controlled_neighborhood")]:
        ids=[i for i,p in enumerate(condition["pairs"]) if p["category"]==category]
        result[name]=errors[:,ids].square().mean(-1).sqrt() if ids else None
    return result
