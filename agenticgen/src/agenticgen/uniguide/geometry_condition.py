"""Immutable template-distance conditions with optional explicit target atom pairs."""
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


def prepare(features,valid_generated_slots=None,*,reward=None):
    """Use only supplied template features and optional target coordinates, not archives or affinity."""
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
    if reward is not None and coordinates is None:raise ValueError("Component reward requires an input 3D template structure")
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
    component_mode=reward is not None or bool(features.get("interact_atoms"))
    if component_mode and explicit is not None:
        raise ValueError("Component reward uses all template-mapped outer atoms; remove neighborhood_atom_ids")
    graph_distance=Chem.GetDistanceMatrix(mol)
    pairs=set(); categories={}
    for left in controlled:
        neighbors=[right for right in ids if right not in controlled] if component_mode else explicit[left] if explicit is not None else [right for right in ids
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
    if component_mode:
        from .geometry_reward import interaction_pairs,normalization_policy,weights,WEIGHTS
        body.update(reward_policy="template-relative-three-component-v1",neighborhood_policy="all_template_mapped_outer_atoms",
            interaction_pairs=interaction_pairs(features,mapping,xyz.tolist()),
            interact_atoms=features.get("interact_atoms",[]),
            distance_normalization=normalization_policy(reward),coordinate_frame=features.get("coordinate_frame"))
        body[WEIGHTS]=weights(body)
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
    if condition.get("reward_policy"):
        from .geometry_reward import weights,normalization_policy,WEIGHTS
        if condition["reward_policy"]!="template-relative-three-component-v1":raise ValueError("Unknown component reward policy")
        weights(condition,condition[WEIGHTS]);normalization_policy(condition["distance_normalization"])
        expected={tuple(sorted((a,b))) for a in controlled for b in ids if a!=b}
        if set(keys)!=expected:raise ValueError("Component reward must include all template-mapped condition/outer pairs")
        seen=set()
        target_coordinates={}
        for p in condition.get("interaction_pairs",[]):
            key=(p["atom_id"],p["target_atom_id"])
            if key in seen or p["atom_id"] not in mapping or p["generated_slot"]!=mapping[p["atom_id"]]:
                raise ValueError("Invalid interaction correspondence")
            seen.add(key)
            target=np.asarray(p["target_coordinates_angstrom"],dtype=np.float64)
            if target.shape!=(3,) or not np.isfinite(target).all():raise ValueError("Invalid target coordinates")
            if p["target_atom_id"] in target_coordinates and target_coordinates[p["target_atom_id"]]!=target.tolist():
                raise ValueError("One target identity has conflicting coordinates")
            target_coordinates[p["target_atom_id"]]=target.tolist()
            d=float(np.linalg.norm(xyz[index[p["atom_id"]]]-target))
            if d<=0 or not math.isclose(d,p["reference_distance_angstrom"],rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("Interaction reference differs from template")
        specified=[(a["atom_id"],t) for a in condition.get("interact_atoms",[]) for t in a["target_atom_ids"]]
        if len(specified)!=len(set(specified)) or set(specified)!=seen:
            raise ValueError("Interaction pairs differ from explicit interact_atoms")
        if seen and condition.get("coordinate_frame")!="generation_context_angstrom":raise ValueError("Interaction coordinate frame missing")
    return condition


def measurements(endpoint,condition,valid,check=None):
    validate(condition)
    if check is None:finite("distance endpoint",endpoint)
    else:endpoint=check("distance endpoint",endpoint)
    if endpoint.ndim!=3 or endpoint.shape[-1]!=3 or valid.shape!=endpoint.shape[:2]:
        raise ValueError("Invalid coordinate/mask layout")
    all_slots=torch.as_tensor(list(condition["atom_to_generated_slot"].values()),device=endpoint.device)
    if int(all_slots.max())>=endpoint.shape[1] or not bool(valid[:,all_slots].bool().all()):
        raise ValueError("Fixed condition mapping references masked or absent slots")
    pairs=torch.as_tensor([p["generated_slots"] for p in condition["pairs"]],device=endpoint.device)
    reference=endpoint.new_tensor([p["reference_distance_angstrom"] for p in condition["pairs"]]).detach()
    measured=torch.linalg.vector_norm(endpoint[:,pairs[:,0]]-endpoint[:,pairs[:,1]],dim=-1)
    errors=measured-reference
    if check is None:finite("internal distance errors",errors)
    else:errors=check("internal distance errors",errors)
    return errors,reference,pairs


def loss(endpoint,condition,valid,loss_weights=None,*,current=None):
    if condition.get("reward_policy"):
        from .geometry_reward import evaluate
        return evaluate(endpoint,condition,valid,loss_weights,current=current)[0]
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
    if condition.get("reward_policy"):
        from .geometry_reward import INTERACTION_DISTANCE
        pairs=condition.get("interaction_pairs",[])
        if pairs:
            measured=torch.linalg.vector_norm(endpoint[:,[p["generated_slot"] for p in pairs]]-
                endpoint.new_tensor([p["target_coordinates_angstrom"] for p in pairs]),dim=-1)
            result[INTERACTION_DISTANCE]=(measured-endpoint.new_tensor([p["reference_distance_angstrom"] for p in pairs])).square().mean(-1).sqrt()
        else:result[INTERACTION_DISTANCE]=endpoint.new_zeros(len(endpoint))
    return result
