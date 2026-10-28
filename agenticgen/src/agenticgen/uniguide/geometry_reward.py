"""Template-relative distance objectives and candidate-local binary controls.

Losses are minimized; the corresponding reward is their negative. Coordinates
are Angstrom in the template frame. No target coordinate is optimized/replaced.
"""
import math
import torch

WEIGHTS = "guidance_loss_weights"
KEYS = ("condition_intra_weight", "condition_outer_weight", "interact_weigh")
LOSSES = ("condition_intra_loss", "condition_outer_loss", "interact_loss")
INTERACTION_DISTANCE = "interaction_atom_pair_distance_root_mean_square_deviation_angstrom"
DEFAULT_NORMALIZATION = dict(mode="bounded_step_mean_detached", minimum_length_angstrom=1., maximum_scale_ratio=4.)


def normalization_policy(value=None):
    result=dict(DEFAULT_NORMALIZATION, **(value or {}))
    if set(result)!=set(DEFAULT_NORMALIZATION) or result["mode"] not in ("none", "template_mean", "bounded_step_mean_detached"):
        raise ValueError("Unknown distance normalization policy")
    for key in ("minimum_length_angstrom", "maximum_scale_ratio"):
        x=result[key]
        if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
            raise ValueError("Positive finite normalization settings required")
    if result["maximum_scale_ratio"]<1:raise ValueError("Normalization maximum ratio must be at least one")
    return result


def weights(condition, value=None):
    result=dict(zip(KEYS,(1,0,int(bool(condition.get("interaction_pairs")))))) if value is None else dict(value)
    if set(result)!=set(KEYS) or any(type(result[k]) is not int or result[k] not in (0,1) for k in KEYS):
        raise ValueError("Loss weights must be exact binary integers 0 or 1")
    if result["interact_weigh"] and not condition.get("interaction_pairs"):
        raise ValueError("Cannot enable interaction loss without explicit interact_atoms")
    if result["condition_outer_weight"] and not any(p["category"]=="controlled_neighborhood" for p in condition["pairs"]):
        raise ValueError("No template-mapped outer pairs available")
    return result


def interaction_pairs(features, mapping, coordinates):
    interactions=features.get("interact_atoms", [])
    if not isinstance(interactions,list):raise ValueError("interact_atoms must be a list")
    if not interactions:return []
    if features.get("coordinates_angstrom") is None or features.get("coordinate_frame")!="generation_context_angstrom":
        raise ValueError("Interaction guidance requires template coordinates in the declared generation_context_angstrom frame")
    targets=features.get("target_atoms",[])
    if not isinstance(targets,list):raise ValueError("target_atoms must be a list")
    target_map={}
    for target in targets:
        if set(target)!={"id","coordinates_angstrom"} or not isinstance(target["id"],str) or not target["id"] or target["id"] in target_map:
            raise ValueError("Unique target atom identity and coordinates required")
        xyz=target["coordinates_angstrom"]
        if not isinstance(xyz,list) or len(xyz)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in xyz):
            raise ValueError("Finite target atom coordinates required")
        target_map[target["id"]]=xyz
    reference={a["id"]:coordinates[i] for i,a in enumerate(features["atoms"])}
    rows=[];seen=set();ligands=set()
    for item in interactions:
        if not isinstance(item,dict) or set(item)!={"atom_id","target_atom_ids"}:
            raise ValueError("interact_atoms entries require atom_id and target_atom_ids")
        atom=item["atom_id"];targets=item["target_atom_ids"]
        if atom not in mapping or atom in ligands or not isinstance(targets,list) or not targets:
            raise ValueError("Unknown/repeated interact atom or empty target correspondence")
        ligands.add(atom)
        for target in targets:
            if target not in target_map or (atom,target) in seen:raise ValueError("Unknown/repeated interaction pair")
            seen.add((atom,target));xyz=target_map[target]
            distance=math.dist(reference[atom],xyz)
            if distance<=0:raise ValueError("Template interaction atoms cannot coincide")
            rows.append(dict(atom_id=atom,target_atom_id=target,generated_slot=mapping[atom],
                target_coordinates_angstrom=xyz,reference_distance_angstrom=distance))
    return rows


def denominator(current,valid,condition,policy):
    policy=normalization_policy(policy)
    matrix=condition["reference_distance_matrix_angstrom"]
    values=[matrix[i][j] for i in range(len(matrix)) for j in range(i+1,len(matrix))]
    baseline=max(sum(values)/len(values) if values else 0.,policy["minimum_length_angstrom"])
    if policy["mode"]=="none":return current.new_ones(len(current))
    if policy["mode"]=="template_mean":return current.new_full((len(current),),baseline)
    # Every valid ligand atom at the CURRENT step, not the predicted endpoint.
    # Detachment avoids gradients through an expanding/contracting denominator.
    xyz=torch.where(valid.bool()[...,None],current.detach(),torch.zeros_like(current))
    pairs=torch.triu_indices(xyz.shape[1],xyz.shape[1],1,device=xyz.device)
    mask=valid[:,pairs[0]].bool() & valid[:,pairs[1]].bool()
    distances=torch.linalg.vector_norm(xyz[:,pairs[0]]-xyz[:,pairs[1]],dim=-1)
    mean=(distances*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    return mean.clamp(min=baseline,max=baseline*policy["maximum_scale_ratio"]).detach()


def evaluate(endpoint,condition,valid,loss_weights=None,*,current=None,normalization=None,check=None):
    """Independent per-candidate component means; check can quarantine bad rows."""
    from .geometry_condition import measurements
    errors,reference,pairs=measurements(endpoint,condition,valid,check=check)
    check=check or (lambda name,x:x)
    groups=[[i for i,p in enumerate(condition["pairs"]) if p["category"]==category]
        for category in ("controlled_internal","controlled_neighborhood")]
    # Keep graph-connected zeros so a completely disabled objective has a valid zero VJP.
    zero=endpoint.sum((1,2))*0
    raw=[errors[:,ids].square().mean(-1)/2 if ids else zero for ids in groups]
    interaction=condition.get("interaction_pairs",[])
    if interaction:
        slots=[p["generated_slot"] for p in interaction]
        target=endpoint.new_tensor([p["target_coordinates_angstrom"] for p in interaction]).detach()
        target_distance=endpoint.new_tensor([p["reference_distance_angstrom"] for p in interaction]).detach()
        delta=check("interaction distance errors",torch.linalg.vector_norm(endpoint[:,slots]-target,dim=-1)-target_distance)
        raw.append(delta.square().mean(-1)/2)
    else:raw.append(zero)
    policy=normalization_policy(normalization or condition.get("distance_normalization"))
    if current is None and policy["mode"]=="bounded_step_mean_detached":
        raise ValueError("Current-step coordinates required for step normalization")
    scale=check("distance normalization denominator",denominator(endpoint if current is None else current,valid,condition,policy))
    values=torch.stack(raw,-1)/scale[:,None].square()
    if isinstance(loss_weights,list):
        if len(loss_weights)!=len(endpoint):raise ValueError("Per-row loss weight count differs")
        checked=[weights(condition,w) for w in loss_weights]
    else:checked=[weights(condition,loss_weights)]*len(endpoint)
    coefficients=endpoint.new_tensor([[w[k] for k in KEYS] for w in checked])
    total=check("normalized distance loss",(values*coefficients).sum(-1))
    all_slots=torch.as_tensor(sorted(condition["atom_to_generated_slot"].values()),device=endpoint.device)
    selected=torch.zeros_like(valid,dtype=torch.bool);selected[:,all_slots]=True
    diagnostics={name:values[:,i] for i,name in enumerate(LOSSES)}
    diagnostics.update({name+"_unnormalized_squared_angstrom":raw[i] for i,name in enumerate(LOSSES)})
    diagnostics["distance_normalization_denominator_angstrom"]=scale
    diagnostics["guidance_total_loss"]=total
    return (total,reference[None].expand(len(endpoint),-1),selected,all_slots[None].expand(len(endpoint),-1).detach()),diagnostics
