"""Supplementary read-only chemical identity audit, separate from PoseB validity."""
import hashlib
import json
from pathlib import Path
from rdkit import Chem

ROOT = Path('/workspace/agenticgen_chemical_interact_sum_20261029')
OUT = Path('/workspace/uniguide_diagnostics/chemical-interact-sum-poseb-20261029/results/CHEMISTRY.json')


def bonds(mol, indices):
    selected = set(indices)
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): str(b.GetBondType())
            for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in selected and b.GetEndAtomIdx() in selected}


def main():
    subjects = json.loads((ROOT/'SUBJECTS.json').read_text())
    groups = json.loads(next(ROOT.glob('calibration/frozen/*/coarse_results.json')).read_text())
    result = []
    for group in groups:
        teacher = next(s['teacher'] for s in subjects if s['teacher']['target'] == group['teacher_id'])
        reference = next(m for m in Chem.SDMolSupplier(teacher['sdf']) if m is not None)
        core = teacher['core_native_slots']
        ref_bonds = bonds(reference, core)
        folder = ROOT/group['sampling_provenance'][0]['source_group']
        path = folder/'molecules_all_built.sdf'
        for mol in Chem.SDMolSupplier(str(path)):
            if mol is None:
                raise ValueError('Saved built molecule did not load')
            row = dict(scale=group['effective_guidance_scale_schedule'][0],
                       logical_seed=int(mol.GetProp('logical_seed')),
                       logical_slot=int(mol.GetProp('logical_slot')),
                       node_id=mol.GetProp('node_id'),
                       source_sdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            row['core_element_matches'] = sum(mol.GetAtomWithIdx(i).GetAtomicNum() == reference.GetAtomWithIdx(i).GetAtomicNum() for i in core)
            row['core_charge_matches'] = sum(mol.GetAtomWithIdx(i).GetFormalCharge() == reference.GetAtomWithIdx(i).GetFormalCharge() for i in core)
            row['core_bond_graph_matches'] = bonds(mol, core) == ref_bonds
            row['full_graph_smiles_matches'] = Chem.MolToSmiles(mol, isomericSmiles=False) == Chem.MolToSmiles(reference, isomericSmiles=False)
            row['core_element_and_charge_matches'] = row['core_element_matches'] == len(core) and row['core_charge_matches'] == len(core)
            row['core_chemistry_matches'] = row['core_element_and_charge_matches'] and row['core_bond_graph_matches']
            result.append(row)
    OUT.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
