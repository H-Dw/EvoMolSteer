"""Template-relative distance objectives and candidate-local binary controls.

Losses are minimized; the corresponding reward is their negative. Coordinates
are Angstrom in the template frame. No target coordinate is optimized/replaced.
"""

import math
import torch

WEIGHTS = "guidance_loss_weights"
KEYS = ("condition_intra_weight", "condition_outer_weight", "interact_weigh")
LOSSES = ("condition_intra_loss", "condition_outer_loss", "interact_loss")
INTERACTION_DISTANCE = (
    "interaction_atom_pair_distance_root_mean_square_deviation_angstrom"
)
DEFAULT_NORMALIZATION = dict(
    mode="bounded_step_mean_detached",
    minimum_length_angstrom=1.0,
    maximum_scale_ratio=4.0,
)


def normalization_policy(value=None):
    result = dict(DEFAULT_NORMALIZATION, **(value or {}))
    if set(result) != set(DEFAULT_NORMALIZATION) or result["mode"] not in (
        "none",
        "template_mean",
        "bounded_step_mean_detached",
    ):
        raise ValueError("Unknown distance normalization policy")
    for key in ("minimum_length_angstrom", "maximum_scale_ratio"):
        x = result[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError("Positive finite normalization settings required")
    if result["maximum_scale_ratio"] < 1:
        raise ValueError("Normalization maximum ratio must be at least one")
    return result


def weights(condition, value=None):
    result = (
        dict(zip(KEYS, (1, 0, int(bool(condition.get("interaction_pairs"))))))
        if value is None
        else dict(value)
    )
    if set(result) != set(KEYS) or any(
        type(result[k]) is not int or result[k] not in (0, 1) for k in KEYS
    ):
        raise ValueError("Loss weights must be exact binary integers 0 or 1")
    if result["interact_weigh"] and not condition.get("interaction_pairs"):
        raise ValueError(
            "Cannot enable interaction loss without explicit interact_atoms"
        )
    if result["condition_outer_weight"] and not any(
        p["category"] == "controlled_neighborhood" for p in condition["pairs"]
    ):
        raise ValueError("No template-mapped outer pairs available")
    return result


def interaction_pairs(features, mapping, coordinates):
    interactions = features.get("interact_atoms", [])
    if not isinstance(interactions, list):
        raise ValueError("interact_atoms must be a list")
    if not interactions:
        return []
    if (
        features.get("coordinates_angstrom") is None
        or features.get("coordinate_frame") != "generation_context_angstrom"
    ):
        raise ValueError(
            "Interaction guidance requires template coordinates in the declared generation_context_angstrom frame"
        )
    targets = features.get("target_atoms", [])
    if not isinstance(targets, list):
        raise ValueError("target_atoms must be a list")
    target_map = {}
    for target in targets:
        if (
            set(target) != {"id", "coordinates_angstrom"}
            or not isinstance(target["id"], str)
            or not target["id"]
            or target["id"] in target_map
        ):
            raise ValueError("Unique target atom identity and coordinates required")
        xyz = target["coordinates_angstrom"]
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in xyz)
        ):
            raise ValueError("Finite target atom coordinates required")
        target_map[target["id"]] = xyz
    reference = {a["id"]: coordinates[i] for i, a in enumerate(features["atoms"])}
    rows = []
    seen = set()
    ligands = set()
    for item in interactions:
        if not isinstance(item, dict) or set(item) != {"atom_id", "target_atom_ids"}:
            raise ValueError(
                "interact_atoms entries require atom_id and target_atom_ids"
            )
        atom = item["atom_id"]
        targets = item["target_atom_ids"]
        if (
            atom not in mapping
            or atom in ligands
            or not isinstance(targets, list)
            or not targets
        ):
            raise ValueError(
                "Unknown/repeated interact atom or empty target correspondence"
            )
        ligands.add(atom)
        for target in targets:
            if target not in target_map or (atom, target) in seen:
                raise ValueError("Unknown/repeated interaction pair")
            seen.add((atom, target))
            xyz = target_map[target]
            distance = math.dist(reference[atom], xyz)
            if distance <= 0:
                raise ValueError("Template interaction atoms cannot coincide")
            rows.append(
                dict(
                    atom_id=atom,
                    target_atom_id=target,
                    generated_slot=mapping[atom],
                    target_coordinates_angstrom=xyz,
                    reference_distance_angstrom=distance,
                )
            )
    return rows


def denominator(current, valid, condition, policy):
    policy = normalization_policy(policy)
    matrix = condition["reference_distance_matrix_angstrom"]
    values = [
        matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))
    ]
    baseline = max(
        sum(values) / len(values) if values else 0.0, policy["minimum_length_angstrom"]
    )
    if policy["mode"] == "none":
        return current.new_ones(len(current))
    if policy["mode"] == "template_mean":
        return current.new_full((len(current),), baseline)
    # Every valid ligand atom at the CURRENT step, not the predicted endpoint.
    # Detachment avoids gradients through an expanding/contracting denominator.
    xyz = torch.where(
        valid.bool()[..., None], current.detach(), torch.zeros_like(current)
    )
    pairs = torch.triu_indices(xyz.shape[1], xyz.shape[1], 1, device=xyz.device)
    mask = valid[:, pairs[0]].bool() & valid[:, pairs[1]].bool()
    distances = torch.linalg.vector_norm(xyz[:, pairs[0]] - xyz[:, pairs[1]], dim=-1)
    mean = (distances * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return mean.clamp(
        min=baseline, max=baseline * policy["maximum_scale_ratio"]
    ).detach()


def checked_tensor(name, value):
    """Validate without adding an identity-valued node to the autograd graph."""
    from .condition import finite

    finite(name, value)
    return value


def component_weights(condition, supplied, population):
    """Resolve scalar or per-candidate controls into exactly [candidate, 3]."""
    if not isinstance(supplied, list):
        return [weights(condition, supplied)] * population
    if len(supplied) != population:
        raise ValueError("Per-row loss weight count differs")
    return [weights(condition, item) for item in supplied]


def internal_components(errors, condition, zero):
    categories = ("controlled_internal", "controlled_neighborhood")
    groups = [
        [
            index
            for index, pair in enumerate(condition["pairs"])
            if pair["category"] == category
        ]
        for category in categories
    ]
    return [
        errors[:, indices].square().mean(-1) / 2 if indices else zero
        for indices in groups
    ]


def interaction_component(endpoint, condition, zero, check):
    pairs = condition.get("interaction_pairs", [])
    if not pairs:
        return zero
    slots = [pair["generated_slot"] for pair in pairs]
    targets = endpoint.new_tensor(
        [pair["target_coordinates_angstrom"] for pair in pairs]
    ).detach()
    references = endpoint.new_tensor(
        [pair["reference_distance_angstrom"] for pair in pairs]
    ).detach()
    errors = check(
        "interaction distance errors",
        torch.linalg.vector_norm(endpoint[:, slots] - targets, dim=-1) - references,
    )
    return errors.square().mean(-1) / 2


def evaluate(
    endpoint,
    condition,
    valid,
    loss_weights=None,
    *,
    current=None,
    normalization=None,
    check=None,
):
    """Per-candidate distance means and full endpoint gradients.

    Coordinates/references are Angstrom, weights have shape [candidate, 3],
    and the detached length denominator has shape [candidate]. Reductions
    never include the candidate axis. Keep component order and arithmetic
    grouping stable across independent and concurrent execution.
    """
    from .geometry_condition import measurements

    check = check or checked_tensor
    errors, reference, _ = measurements(endpoint, condition, valid, check=check)
    # Connected zero preserves a valid zero VJP for absent/disabled components.
    zero = endpoint.sum((1, 2)) * 0
    raw = internal_components(errors, condition, zero)
    raw.append(interaction_component(endpoint, condition, zero, check))

    policy = normalization_policy(
        normalization or condition.get("distance_normalization")
    )
    if current is None and policy["mode"] == "bounded_step_mean_detached":
        raise ValueError("Current-step coordinates required for step normalization")
    coordinates = endpoint if current is None else current
    if (
        coordinates.shape != endpoint.shape
        or coordinates.device != endpoint.device
        or coordinates.dtype != endpoint.dtype
    ):
        raise ValueError(
            "Normalization state must match endpoint layout, device and dtype"
        )
    scale = check(
        "distance normalization denominator",
        denominator(coordinates, valid, condition, policy),
    )
    values = torch.stack(raw, -1) / scale[:, None].square()
    resolved = component_weights(condition, loss_weights, len(endpoint))
    coefficients = endpoint.new_tensor([[row[key] for key in KEYS] for row in resolved])
    total = check("normalized distance loss", (values * coefficients).sum(-1))

    slots = torch.as_tensor(
        sorted(condition["atom_to_generated_slot"].values()), device=endpoint.device
    )
    selected = torch.zeros_like(valid, dtype=torch.bool)
    selected[:, slots] = True
    diagnostics = {name: values[:, index] for index, name in enumerate(LOSSES)}
    diagnostics.update(
        {
            name + "_unnormalized_squared_angstrom": raw[index]
            for index, name in enumerate(LOSSES)
        }
    )
    diagnostics.update(
        distance_normalization_denominator_angstrom=scale, guidance_total_loss=total
    )
    result = (
        total,
        reference[None].expand(len(endpoint), -1),
        selected,
        slots[None].expand(len(endpoint), -1).detach(),
    )
    return result, diagnostics
