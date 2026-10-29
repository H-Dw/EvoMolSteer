"""Protein-anchored absolute coordinates with one initial assignment per trajectory.

Alignment is a change of measurement frame, never a sampling-state projection.
The protein is fixed input; its Kabsch transform has no ligand dependence.
"""
from pathlib import Path
import copy
from dataclasses import dataclass
import math
import numpy as np
import torch
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from .io import value_sha, sha, frozen, read, write

POLICY = "protein-anchored-absolute-coordinate-guidance-v2-uncapped-upper-tail"
RMSD = "condition_coordinate_root_mean_square_deviation_angstrom"
MAX_ERROR = "maximum_condition_atom_coordinate_error_angstrom"
METRICS = (RMSD, MAX_ERROR)
ASSIGNMENT = "initial_endpoint_hungarian_freeze"


def is_absolute(condition):
    return condition.get("policy") == POLICY


def protein_atoms(path):
    atoms = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM  ") or line[76:78].strip().upper() in ("H", "D"):
            continue
        key = "|".join((line[21:22], line[22:26].strip(), line[26:27], line[12:16].strip()))
        # PDB's primary conformer is the input representation; do not merge altlocs.
        if line[16:17] not in (" ", "A"):
            continue
        if key in atoms:
            raise ValueError("Ambiguous protein atom identity: " + key)
        atoms[key] = [float(line[i:i+8]) for i in (30, 38, 46)]
    return atoms


def kabsch(moving, reference):
    a, b = np.asarray(moving, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 3:
        raise ValueError("At least three corresponding protein atoms required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite protein coordinates")
    ac, bc = a-a.mean(0), b-b.mean(0)
    if min(np.linalg.matrix_rank(ac), np.linalg.matrix_rank(bc)) < 2:
        raise ValueError("Protein alignment anchors are collinear")
    if np.array_equal(a, b):
        return np.eye(3), np.zeros(3), 0.
    u, _, vt = np.linalg.svd(ac.T @ bc)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt  # Row-vector convention: x @ rotation + translation.
    translation = b.mean(0)-a.mean(0) @ rotation
    error = float(np.sqrt(np.mean(np.sum((a @ rotation+translation-b)**2, axis=-1))))
    return rotation, translation, error


def alignment(protein, reference_protein, maximum_rmsd=.001):
    left, right = protein_atoms(protein), protein_atoms(reference_protein)
    identities = sorted(set(left) & set(right))
    rotation, translation, error = kabsch([left[i] for i in identities], [right[i] for i in identities])
    if error > maximum_rmsd:
        raise ValueError("Protein frames are not the same rigid structure: " + str(error))
    return dict(policy="protein_anchored_proper_kabsch", atom_identities=identities,
                moving_protein_sha256=sha(protein), reference_protein_sha256=sha(reference_protein),
                rotation=rotation.tolist(), translation_angstrom=translation.tolist(),
                fitted_protein_rmsd_angstrom=error, maximum_protein_rmsd_angstrom=maximum_rmsd,
                ligand_participates_in_fit=False, sampling_coordinates_replaced=False)


def prepare(features, protein, reference_protein=None, valid_generated_slots=None):
    atoms, bonds = features["atoms"], features["bonds"]
    ids = [a["id"] for a in atoms]
    controlled = sorted(features["controlled_atom_ids"])
    if (not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids)
            or not controlled or len(set(controlled)) != len(controlled) or not set(controlled) <= set(ids)):
        raise ValueError("Unique atom and condition identities required")
    if features.get("coordinate_frame") != "generation_context_angstrom":
        raise ValueError("Absolute coordinates require the generation context frame")
    xyz = np.asarray(features["coordinates_angstrom"], dtype=np.float64)
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError("Finite input 3D reference required; no arbitrary conformer fallback")
    if valid_generated_slots is not None and len(valid_generated_slots) < len(controlled):
        raise ValueError("Too few valid generation slots")
    mol = Chem.RWMol(); lookup = {v:i for i,v in enumerate(ids)}; edges = set()
    types = {"single":Chem.BondType.SINGLE,"double":Chem.BondType.DOUBLE,
             "triple":Chem.BondType.TRIPLE,"aromatic":Chem.BondType.AROMATIC}
    for item in atoms:
        atom = Chem.Atom(item["element"])
        if atom.GetAtomicNum() <= 1 or type(item.get("formal_charge", 0)) is not int:
            raise ValueError("Heavy atom identity and integer charge required")
        atom.SetFormalCharge(item.get("formal_charge", 0)); mol.AddAtom(atom)
    for item in bonds:
        a, b = [lookup[v] for v in item["atom_ids"]]; pair = tuple(sorted((a,b)))
        if a == b or pair in edges:
            raise ValueError("Self or repeated chemical bond")
        edges.add(pair); mol.AddBond(a,b,types[item["order"]])
    Chem.SanitizeMol(mol.GetMol())
    body = dict(policy=POLICY, assignment_policy=ASSIGNMENT, unit="angstrom", atoms=atoms, bonds=bonds,
                controlled_atom_ids=controlled, reference_coordinates_angstrom=xyz.tolist(),
                coordinate_frame="generation_context_angstrom", loss_reduction="mean_over_condition_atoms",
                protein_alignment=alignment(protein, reference_protein or protein),
                features_sha256=value_sha(features), coordinates_replaced=False,
                old_component_fields_not_used=True)
    return dict(body, condition_sha256=value_sha(body))


def validate(condition):
    body = {k:v for k,v in condition.items() if k != "condition_sha256"}
    if (not is_absolute(condition) or value_sha(body) != condition.get("condition_sha256")
            or condition.get("assignment_policy") != ASSIGNMENT
            or condition.get("loss_reduction") != "mean_over_condition_atoms"):
        raise ValueError("Absolute coordinate condition binding changed")
    if any(k in condition for k in ("pairs", "interaction_pairs", "guidance_loss_weights", "atom_to_generated_slot")):
        raise ValueError("Old distance objectives or static mapping cannot enter this protocol")
    return condition


def aligned(endpoint, condition):
    frame = condition["protein_alignment"]
    return endpoint @ endpoint.new_tensor(frame["rotation"]).detach() + endpoint.new_tensor(frame["translation_angstrom"]).detach()


def reference(condition, endpoint):
    lookup = {a["id"]:i for i,a in enumerate(condition["atoms"])}
    return endpoint.new_tensor([condition["reference_coordinates_angstrom"][lookup[i]]
                               for i in condition["controlled_atom_ids"]]).detach()


def match(endpoint, condition, valid):
    points = aligned(endpoint.detach(), condition).double().cpu().numpy()
    targets = reference(condition, endpoint).double().cpu().numpy()
    assignments = []
    for xyz, mask in zip(points, valid.detach().cpu().numpy()):
        slots = np.flatnonzero(mask)
        if len(slots) < len(targets):
            raise ValueError("Too few valid slots for injective initial assignment")
        cost = np.sum((targets[:,None]-xyz[slots][None])**2, axis=-1)
        if not np.isfinite(cost).all():
            raise FloatingPointError("Nonfinite assignment cost")
        rows, cols = linear_sum_assignment(cost)
        assignments.append(slots[cols[np.argsort(rows)]].tolist())
    return torch.as_tensor(assignments, device=endpoint.device, dtype=torch.long)


@dataclass(frozen=True)
class FrozenAssignment:
    """Slot uniqueness is static; the current mask remains a dynamic check."""

    slots: torch.Tensor
    selected: torch.Tensor
    coordinate_shape: tuple

    def check_layout(self, endpoint, valid):
        if tuple(endpoint.shape) != self.coordinate_shape or valid.shape != endpoint.shape[:2]:
            raise ValueError("Coordinate/mask shape differs from frozen assignment")
        if endpoint.device != self.slots.device or valid.device != endpoint.device:
            raise ValueError("Coordinate/mask device differs from frozen assignment")
        if not bool(valid.bool().gather(1, self.slots).all()):
            raise ValueError("Assignment is not an injective valid selection")


class CoordinateObjective:
    """Validate one condition snapshot, then reuse constants without hashing in the VJP."""

    def __init__(self, condition):
        self.condition = copy.deepcopy(validate(condition))
        lookup = {atom["id"]: i for i, atom in enumerate(self.condition["atoms"])}
        self.target_coordinates = tuple(
            tuple(self.condition["reference_coordinates_angstrom"][lookup[atom]])
            for atom in self.condition["controlled_atom_ids"]
        )
        frame = self.condition["protein_alignment"]
        self.rotation = tuple(tuple(row) for row in frame["rotation"])
        self.translation = tuple(frame["translation_angstrom"])
        self._tensors = {}

    def tensors(self, endpoint):
        key = (endpoint.device, endpoint.dtype)
        if key not in self._tensors:
            self._tensors[key] = tuple(
                endpoint.new_tensor(value).detach()
                for value in (self.rotation, self.translation, self.target_coordinates)
            )
        return self._tensors[key]

    def bind(self, endpoint, valid, assignment):
        if endpoint.ndim != 3 or endpoint.shape[-1] != 3 or valid.shape != endpoint.shape[:2]:
            raise ValueError("Coordinate/mask shape differs")
        slots = torch.as_tensor(assignment, device=endpoint.device, dtype=torch.long).detach().clone()
        if slots.shape != (len(endpoint), len(self.target_coordinates)):
            raise ValueError("Frozen assignment population differs")
        if (bool((slots < 0).any()) or bool((slots >= endpoint.shape[1]).any())
                or any(len(set(row)) != len(row) for row in slots.cpu().tolist())):
            raise ValueError("Assignment is not an injective valid selection")
        selected = torch.zeros_like(valid, dtype=torch.bool).scatter(1, slots, True)
        return FrozenAssignment(slots, selected.detach(), tuple(endpoint.shape))

    def errors(self, endpoint, valid, assignment):
        assignment.check_layout(endpoint, valid)
        rotation, translation, target = self.tensors(endpoint)
        values = (endpoint @ rotation + translation).gather(
            1, assignment.slots[..., None].expand(-1, -1, 3)
        )
        return values - target[None], target

    def evaluate(self, endpoint, valid, assignment, *, check=None):
        errors, target = self.errors(endpoint, valid, assignment)
        value = errors.square().sum((1, 2)) / (2 * len(target))
        if check is not None:
            value = check("absolute coordinate loss", value)
        elif not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite absolute coordinate loss")
        return (value, target[None].expand(len(endpoint), -1, -1),
                assignment.selected, assignment.slots)


def loss(endpoint, condition, valid, assignment, *, check=None):
    """Untrusted one-shot API; sampling uses a prepared CoordinateObjective."""
    objective = CoordinateObjective(condition)
    bound = objective.bind(endpoint, valid, assignment)
    return objective.evaluate(endpoint, valid, bound, check=check)


def metrics(endpoint, condition, valid, assignment):
    value, target, _, slots = loss(endpoint,condition,valid,assignment)
    errors = aligned(endpoint,condition).gather(1,slots[...,None].expand(-1,-1,3))-target
    return {RMSD:(2*value).sqrt(), MAX_ERROR:errors.norm(dim=-1).amax(1)}


class AssignmentState:
    """Per-row state, derived without additional forwards or random draws."""
    def __init__(self, control):
        self.control = control
        self.objective = CoordinateObjective(control.geometry_condition)
        self.rows = {}
        self._assignments = {}

    def restore(self):
        saved = (self.control.resume or {}).get("absolute_assignment_state", {})
        if self.control.resume and value_sha(saved) != self.control.resume.get("absolute_assignment_sha256"):
            raise ValueError("Checkpoint assignment state checksum differs")
        self.rows = {int(k): copy.deepcopy(v) for k, v in saved.items()}
        self._assignments.clear()

    def evaluate(self, endpoint, valid, ids):
        control=self.control
        condition = self.objective.condition
        if control.active_step < 10:
            assignment = self.objective.bind(endpoint, valid, match(endpoint, condition, valid))
        else:
            missing=[j for j,i in enumerate(ids) if i not in self.rows]
            if missing:
                if control.active_step != 10:
                    raise ValueError("Missing initial assignment on recovery; cannot rematch")
                chosen=match(endpoint[missing],condition,valid[missing]).cpu().tolist()
                for j,slots in zip(missing,chosen):
                    self.rows[ids[j]]=dict(slots=slots,first_step=10,
                        endpoint_sha256=value_sha(endpoint[j].detach().double().cpu().tolist()),
                        valid_sha256=value_sha(valid[j].detach().cpu().tolist()))
            key = (tuple(ids), endpoint.device)
            if key not in self._assignments:
                self._assignments[key] = self.objective.bind(
                    endpoint, valid, [self.rows[i]["slots"] for i in ids]
                )
            assignment = self._assignments[key]
        return self.objective.evaluate(
            endpoint, valid, assignment,
            check=lambda name, value: control.check_rows(name, value, ids),
        )

    def save_cache(self):
        c=self.control
        if c.active_step != 10 or "sample_shards" not in c.folder.parts:
            return
        from .protocol import SEEDS
        root=Path(c.absolute_config["root"])/"assignments"/c.teacher["target"]
        for row,saved in self.rows.items():
            unit, local=divmod(row,10)
            slots=c.initial.get("candidate_slots",[c.initial.get("logical_slots",[0,1])])[unit]
            seed=SEEDS[local//2]; slot=slots[local%2]
            frozen(root/f"{seed}_{slot}.json",dict(saved,condition_sha256=c.geometry_condition["condition_sha256"],
                                                   logical_seed=seed,logical_slot=slot))
