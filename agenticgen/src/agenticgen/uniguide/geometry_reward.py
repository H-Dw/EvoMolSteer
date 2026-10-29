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
REWARD_POLICY = "template-relative-three-component-sum-v2"


def weights(condition, value=None):
    from .absolute_coordinate import is_absolute
    if is_absolute(condition):
        if value is not None:
            raise ValueError("Absolute coordinates have one objective, without component controls")
        return None
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
        errors[:, indices].square().sum(-1) / 2 if indices else zero
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
    return errors.square().sum(-1) / 2


def evaluate(
    endpoint,
    condition,
    valid,
    loss_weights=None,
    *,
    check=None,
):
    """Per-candidate half sums of squared distance errors and full endpoint gradients.

    Coordinates/references are Angstrom and losses are Angstrom squared.
    Weights have shape [candidate, 3]. Reductions never include the candidate
    axis. No current-state coordinates or length scale enter this objective.
    """
    from .geometry_condition import measurements

    check = check or checked_tensor
    errors, reference, _ = measurements(endpoint, condition, valid, check=check)
    # Connected zero preserves a valid zero VJP for absent/disabled components.
    zero = endpoint.sum((1, 2)) * 0
    components = internal_components(errors, condition, zero)
    components.append(interaction_component(endpoint, condition, zero, check))
    values = torch.stack(components, -1)
    resolved = component_weights(condition, loss_weights, len(endpoint))
    coefficients = endpoint.new_tensor([[row[key] for key in KEYS] for row in resolved])
    total = check("summed distance loss", (values * coefficients).sum(-1))

    slots = torch.as_tensor(
        sorted(condition["atom_to_generated_slot"].values()), device=endpoint.device
    )
    selected = torch.zeros_like(valid, dtype=torch.bool)
    selected[:, slots] = True
    diagnostics = {name: values[:, index] for index, name in enumerate(LOSSES)}
    diagnostics["guidance_total_loss"] = total
    result = (
        total,
        reference[None].expand(len(endpoint), -1),
        selected,
        slots[None].expand(len(endpoint), -1).detach(),
    )
    return result, diagnostics
