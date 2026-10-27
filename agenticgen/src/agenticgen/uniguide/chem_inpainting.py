"""Chemical-only inpainting on the stopped-gradient geometric assignment.

Coordinates, masks and self-conditioning are never projected. The native
categorical transition runs before projection and consumes its original RNG.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .condition import condition_map
from .io import read, write, frozen, value_sha, sha
from .native_tools import save_pt, tensor_digest
from .sampler import Control

POLICY = "plip-core-chem-only-state-v1"
NODE_FEATURES = ("atomics", "charges", "hybridization")


def teacher_features(model, teacher):
    from flowr.util.rdkit import BOND_IDX_MAP
    source = Path(teacher["sdf"])
    mols = [m for m in Chem.SDMolSupplier(str(source), removeHs=True) if m is not None]
    if len(mols) != 1 or mols[0].GetProp("node_id") != teacher["node_id"]:
        raise ValueError("Chemical teacher identity differs")
    mol = Chem.Mol(mols[0])
    ids = teacher["core_sdf_atoms"]
    if (len(set(ids)) != len(ids) or not ids or mol.GetNumHeavyAtoms() != teacher["heavy_atoms"]
            or not np.allclose(mol.GetConformer().GetPositions()[ids], teacher["core_A"], atol=1e-7, rtol=0)):
        raise ValueError("Chemical core must be the original coordinate core in the same order")
    if model.hparams.get("remove_aromaticity", False):
        Chem.Kekulize(mol, clearAromaticFlags=True)
    atoms = [mol.GetAtomWithIdx(i) for i in ids]
    node = dict(atomics=torch.tensor(model.vocab.indices_from_tokens([a.GetSymbol() for a in atoms], one_hot=True)),
        charges=torch.tensor(model.vocab_charges.indices_from_tokens([a.GetFormalCharge() for a in atoms], one_hot=True)))
    if model.add_feats:
        node["hybridization"] = torch.tensor(model.vocab_hybridization.indices_from_tokens(
            [a.GetHybridization() for a in atoms], one_hot=True))
    bonds = torch.zeros((len(ids), len(ids)), dtype=torch.long)
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            bond = mol.GetBondBetweenAtoms(left, right) if left != right else None
            if bond is not None:
                bonds[i, j] = BOND_IDX_MAP[bond.GetBondType()]
    features = dict(node, bonds=torch.nn.functional.one_hot(bonds, 5))
    metadata = dict(policy=POLICY, teacher_sha256=value_sha(teacher), sdf_sha256=sha(source),
        node_id=teacher["node_id"], core_sdf_atoms=ids, fields=list(features),
        atomic_symbols=[a.GetSymbol() for a in atoms], formal_charges=[a.GetFormalCharge() for a in atoms],
        hybridization=[str(a.GetHybridization()) for a in atoms], induced_bond_classes=bonds.tolist(),
        remove_aromaticity=bool(model.hparams.get("remove_aromaticity", False)),
        coordinates_replaced=False, self_conditioning_replaced=False, covalent_expansion=0)
    return features, metadata


def project_chemistry(state, assignment, features):
    """Copy chemical channels only. Assignment order is the teacher core order."""
    if not set(features).issubset(set(NODE_FEATURES) | {"bonds"}) or "bonds" not in features:
        raise ValueError("Only native chemical features may be injected")
    mask = state["mask"].bool()
    ids = assignment.to(device=mask.device, dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != len(mask) or ids.shape[1] != len(features["bonds"]):
        raise ValueError("Chemical assignment shape differs from core")
    if any(len(set(row.tolist())) != len(row) for row in ids) or not bool(mask.gather(1, ids).all()):
        raise ValueError("Chemical assignment must be injective and valid")
    result = dict(state)
    batch = torch.arange(len(mask), device=mask.device)[:, None]
    report = dict(coordinates_replaced=False, selected_atoms_per_molecule=ids.shape[1], before={}, after={})
    for key in NODE_FEATURES:
        if key not in features:
            continue
        target = features[key].to(state[key])
        if target.shape != (ids.shape[1], state[key].shape[-1]):
            raise ValueError("Chemical vocabulary mismatch: " + key)
        old = state[key][batch, ids]
        report["before"][key] = float((old.argmax(-1) == target.argmax(-1)).float().mean())
        value = state[key].clone()
        value[batch, ids] = target[None]
        result[key] = value
        report["after"][key] = float((value[batch, ids].argmax(-1) == target.argmax(-1)).float().mean())
    target = features["bonds"].to(state["bonds"])
    if target.shape[-1] != state["bonds"].shape[-1] or not torch.equal(target, target.transpose(0, 1)):
        raise ValueError("Induced chemical bonds must use the native symmetric vocabulary")
    k = ids.shape[1]
    left, right = torch.where(~torch.eye(k, dtype=torch.bool, device=mask.device))
    b = torch.arange(len(mask), device=mask.device)[:, None]
    i, j = ids[:, left], ids[:, right]
    expected = target[left, right]
    report["before"]["bonds"] = float((state["bonds"][b, i, j].argmax(-1) == expected.argmax(-1)).float().mean())
    result["bonds"] = state["bonds"].clone()
    result["bonds"][b, i, j] = expected[None]
    report["after"]["bonds"] = float((result["bonds"][b, i, j].argmax(-1) == expected.argmax(-1)).float().mean())
    if result["coords"] is not state["coords"] or result["mask"] is not state["mask"]:
        raise AssertionError("Chemical projection touched coordinates or mask")
    report["coordinates_sha256"] = tensor_digest(state["coords"])
    return result, report


class ChemicalControl(Control):
    def __init__(self, *args, chemical_features, baseline_trace, **kwargs):
        super().__init__(*args, **kwargs)
        self.chemical_features = chemical_features
        self.baseline_trace = Path(baseline_trace)
        self.chemical_audits = []

    def after(self, curr, step):
        result = super().after(curr, step)
        if step == 10:
            import h5py
            # The first coordinate update precedes the first chemical injection.
            comparisons = {}
            with h5py.File(self.baseline_trace, "r") as reference:
                for key in ("current_A", "guidance_delta_A", "native_noise", "guided_proposal", "assignment"):
                    a, b = self.trace[key][step], reference[key][step]
                    comparisons[key] = dict(equal=bool(np.array_equal(a, b)), max_abs=float(np.max(np.abs(a-b))))
            write(self.folder/"FIRST_STEP_PAIRED_ACCEPTANCE.json", comparisons)
            if not all(v["equal"] for v in comparisons.values()):
                raise ValueError("First pre-inpainting update differs from the saved paired control")
        if step >= 10:
            before_rng = tensor_digest(self.random_streams.states)
            projected, audit = project_chemistry(result, self.last_assignment, self.chemical_features)
            if before_rng != tensor_digest(self.random_streams.states):
                raise ValueError("Chemical projection consumed native RNG")
            audit.update(step=step, assignment=self.last_assignment.detach().cpu().tolist())
            self.chemical_audits.append(audit)
            self.store("chemical_atomics", step, projected["atomics"].argmax(-1))
            self.store("chemical_charges", step, projected["charges"].argmax(-1))
            self.rows[-1]["chemical_inpainting"] = {k:v for k,v in audit.items() if k != "assignment"}
            if step % 10 == 9:
                self.trace.flush();write(self.folder/"steps.json", self.rows)
                write(self.folder/"chemical_steps.json", self.chemical_audits)
            return projected
        return result

    def close(self):
        super().close()
        write(self.folder/"chemical_steps.json", self.chemical_audits)


def terminal_projection(output, teacher, features, folder):
    # The native final corrector's coordinates are retained byte-for-byte.
    folder = Path(folder)
    save_pt(folder/"terminal_native_before_chemical_projection.pt.gz", output)
    assignment = condition_map(output["coords"], teacher["core_A"], output["mask"].bool())[2]
    result, audit = project_chemistry(output, assignment, features)
    audit.update(assignment=assignment.tolist(), stage="after_native_final_corrector",
                 upstream_affinity="computed before final chemical projection; use endpoint rescore for comparisons")
    write(folder/"FINAL_CHEMICAL_PROJECTION.json", audit)
    return result


class PilotBudget:
    """Separate user-authorized 100-attempt pilot; never reset UG accounting."""
    def __init__(self, root):
        self.path = Path(root)/"pilot_attempts.json"

    def reserve(self, run_id, stage, count, binding):
        if stage != "formal" or count != 50 or not run_id.startswith("groups/"):
            raise ValueError("Only two full fifty-attempt treatment groups are authorized")
        ledger = read(self.path) if self.path.exists() else {}
        item = dict(stage="chemical_only_inpainting_pilot", count=count, binding=binding)
        if run_id in ledger:
            if ledger[run_id] != item:
                raise ValueError("Pilot attempt identity changed")
            return
        if sum(r["count"] for r in ledger.values()) + count > 100:
            raise ValueError("Pilot cap is 100 new attempts; failures remain charged")
        ledger[run_id] = item
        write(self.path, ledger)
