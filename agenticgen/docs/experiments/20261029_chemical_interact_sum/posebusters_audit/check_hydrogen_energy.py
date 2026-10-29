"""Energy sensitivity: optimize only added hydrogens, freeze all source heavy atoms.

This is a diagnostic on copies; it never changes the official PoseB verdict,
source pose, sampling output, or saved scientific candidate population.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path('/workspace/uniguide_diagnostics/chemical-interact-sum-poseb-20261029/results')


def measure(row):
    if row['kind'] == 'generated':
        mol = next(m for m in Chem.SDMolSupplier(str(Path(row['source_group'])/'molecules_all_built.sdf'))
                   if m is not None and int(m.GetProp('logical_seed')) == row['logical_seed']
                   and int(m.GetProp('logical_slot')) == row['logical_slot'])
    else:
        mol = next(m for m in Chem.SDMolSupplier(row['sdf']) if m is not None)
    original = np.asarray(mol.GetConformer().GetPositions()).copy()
    enriched = Chem.AddHs(mol, addCoords=True)
    forcefield = AllChem.UFFGetMoleculeForceField(enriched)
    before = forcefield.CalcEnergy()
    for index in range(mol.GetNumAtoms()):
        forcefield.AddFixedPoint(index)
    forcefield.Initialize()
    status = forcefield.Minimize(maxIts=2000)
    after = forcefield.CalcEnergy()
    preserved = np.array_equal(original, np.asarray(enriched.GetConformer().GetPositions())[:mol.GetNumAtoms()])
    if not preserved:
        raise ValueError('Hydrogen-only optimization moved source heavy atoms')
    # RDKit's success code alone is insufficient: reject a numerically
    # worsening optimization as an unusable sensitivity result, retain it.
    usable = bool(status == 0 and np.isfinite(after) and after <= before + 1e-6)
    return dict(candidate_key=row['candidate_key'], old_ratio=row['raw_report']['energy_ratio'],
                old_energy=row['raw_report']['mol_pred_energy'],
                unrelaxed_copy_energy=before, hydrogen_relaxed_energy=after,
                hydrogen_relaxed_ratio=after/row['raw_report']['ensemble_avg_energy'],
                convergence_code=status, usable=usable,
                sensitivity_status='usable' if usable else 'unusable_optimization',
                all_heavy_coordinates_exactly_preserved=preserved)


def main():
    rows = json.loads((BASE/'RESULTS.json').read_text())
    selected = [r for r in rows if r['kind']=='generated' and r['raw_report']['energy_ratio']>7]
    subjects = json.loads(Path('/workspace/agenticgen_chemical_interact_sum_20261029/SUBJECTS.json').read_text())
    for row in rows:
        if row['kind']=='teacher_reference':
            row['sdf'] = next(s['teacher']['sdf'] for s in subjects if s['teacher']['target']==row['target'])
            selected.append(row)
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(measure, selected))
    (BASE/'HYDROGEN_ENERGY_SENSITIVITY.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
