"""Read-only, CPU-only PoseB audit of the 130 saved coarse-screen attempts.

Writes only to --out, never to the frozen sampling directories. No sampler,
LLM, PLIP, geometry minimization of submitted poses, or automatic retries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from rdkit import Chem, rdBase
from posebusters import PoseBusters
import posebusters


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return None


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def expected_checks(config):
    return [m.get("rename_outputs", {}).get(k, k + m.get("rename_suffix", ""))
            .lower().replace(" ", "_")
            for m in config["modules"] for k in m.get("chosen_binary_test_output", [])]


def scope_status(checks, columns):
    values = [checks[k] for k in columns]
    if False in values:
        return "failed"
    return "unknown" if None in values else "passed"


def evaluate(task):
    start = time.monotonic()
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromMolBlock(task["molblock"], removeHs=True, sanitize=True)
    out = {k: v for k, v in task.items() if k != "molblock"}
    out["pid"] = __import__("os").getpid()
    try:
        if mol is None:
            raise ValueError("Saved built SDF could not be sanitized")
        coords = np.asarray(mol.GetConformer().GetPositions())
        coords_before = coords.copy()
        buster = PoseBusters("dock")
        frame = buster.bust(mol, mol_cond=task["protein"], full_report=True)
        if len(frame) != 1:
            raise ValueError(f"Expected one result row, got {len(frame)}")
        row = frame.iloc[0]
        columns = expected_checks(buster.config)
        checks = {k: (bool(row[k]) if isinstance(row.get(k), (bool, np.bool_))
                       else None) for k in columns}
        # A failed force-field calculation is not a measured energy failure.
        energy_available = all(isinstance(row.get(k), (float, int, np.number))
                               and math.isfinite(float(row[k]))
                               for k in ["mol_pred_energy", "ensemble_avg_energy", "energy_ratio"])
        if not energy_available:
            checks["internal_energy"] = None
        intramolecular = columns[:columns.index("protein-ligand_maximum_distance")]
        intramolecular = [k for k in intramolecular if k != "mol_cond_loaded"]
        intermolecular = [k for k in columns if k not in intramolecular
                          and k not in ["mol_pred_loaded", "mol_cond_loaded"]]
        legacy_checks = {k: bool(row[k]) for k in columns
                         if isinstance(row.get(k), (bool, np.bool_))}
        bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        lengths = [float(np.linalg.norm(coords[i] - coords[j])) for i, j in bonds]
        out.update(
            status="evaluated", checks=checks,
            failed_checks=[k for k, v in checks.items() if v is False],
            unknown_checks=[k for k, v in checks.items() if v is None],
            posebusters_status=scope_status(checks, columns),
            intramolecular_status=scope_status(checks, intramolecular),
            protein_context_status=scope_status(checks, intermolecular),
            original_bool_only_pass=all(legacy_checks.values()) if legacy_checks else None,
            original_selected_boolean_count=len(legacy_checks),
            expected_check_count=len(columns),
            raw_report={str(k): v for k, v in row.items()},
            bond_length_minimum_A=min(lengths) if lengths else None,
            bond_length_maximum_A=max(lengths) if lengths else None,
            generated_smiles=Chem.MolToSmiles(mol),
            connected=len(Chem.GetMolFrags(mol)) == 1,
            submitted_heavy_coordinates_unchanged=bool(np.array_equal(
                coords_before, np.asarray(mol.GetConformer().GetPositions()))),
        )
    except Exception as e:
        out.update(status="evaluation_error", error={"type": type(e).__name__,
                   "message": str(e), "traceback": traceback.format_exc()})
    out["elapsed_seconds"] = time.monotonic() - start
    return clean(out)


def verify_original(root, audit):
    bad = []
    for relative, digest in audit["artifact_sha256"].items():
        path = Path(relative) if Path(relative).is_absolute() else root / relative
        if not path.is_file() or sha(path) != digest:
            bad.append(str(path))
    return {"files": len(audit["artifact_sha256"]), "changed": bad}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = args.root
    audit = json.loads((root / "monitoring/terminal_boundary_audit_20261029.json").read_text())
    initial = verify_original(root, audit)
    if initial["changed"]:
        raise ValueError(f"Frozen artifacts changed: {initial['changed']}")
    subjects = json.loads((root / "SUBJECTS.json").read_text())
    coarse = json.loads(next(root.glob("calibration/frozen/*/coarse_results.json")).read_text())
    tasks, attempts, hashes = [], [], {}
    import torch
    for group in coarse:
        source = root / group["sampling_provenance"][0]["source_group"]
        for name in ["molecules_all_built.sdf", "terminal.pt.gz", "records.json", "steps.json"]:
            hashes[str(source / name)] = sha(source / name)
        teacher = next(s["teacher"] for s in subjects if s["teacher"]["target"] == group["teacher_id"])
        steps = json.loads((source / "steps.json").read_text())
        records = json.loads((source / "records.json").read_text())
        with gzip.open(source / "terminal.pt.gz", "rb") as f:
            terminal = torch.load(f, map_location="cpu", weights_only=False)
        coords = terminal["coords"].double().numpy()
        core = teacher["core_native_slots"]
        reference = np.asarray(teacher["core_A"])
        molecules = {}
        for mol in Chem.SDMolSupplier(str(source / "molecules_all_built.sdf"), removeHs=True):
            if mol is None:
                raise ValueError("Unparseable entry in saved all-built SDF")
            key = (int(mol.GetProp("logical_seed")), int(mol.GetProp("logical_slot")))
            if key in molecules:
                raise ValueError("Duplicate SDF candidate identity")
            molecules[key] = mol
        for record in records:
            i = record["tensor_row"]
            key = f"{group['effective_guidance_scale_schedule'][0]:g}:{record['logical_seed']}:{record['logical_slot']}"
            events = [s["step"] for s in steps if s.get("guidance_increment_limited", [False] * 10)[i]]
            entry = dict(
                candidate_key=key, target=group["teacher_id"],
                scale=group["effective_guidance_scale_schedule"][0],
                node_id=record["node_id"], logical_seed=record["logical_seed"],
                logical_slot=record["logical_slot"], tensor_row=i,
                source_group=str(source), build_success=record["build_success"],
                sampling_failure_reason=record["failure_reason"],
                limiter_steps=events, limiter_events=len(events),
                limiter_fraction=len(events) / 90,
                maximum_unconstrained_increment_A=max(
                    s.get("unconstrained_max_atom_increment_A", [0] * 10)[i] for s in steps),
                condition_coordinate_rmsd_A=float(np.sqrt(np.mean(np.sum(
                    (coords[i, core] - reference) ** 2, axis=1)))),
                condition_pair_distance_rmsd_A=record["controlled_atom_pair_distance_root_mean_square_deviation_angstrom"],
                outer_pair_distance_rmsd_A=record["controlled_to_neighborhood_distance_root_mean_square_deviation_angstrom"],
                interaction_pair_distance_rmsd_A=record["interaction_atom_pair_distance_root_mean_square_deviation_angstrom"],
            )
            attempts.append(entry)
            mol = molecules.get((record["logical_seed"], record["logical_slot"]))
            if record["build_success"] != (mol is not None):
                raise ValueError("Build record/SDF identity mismatch")
            if mol is None:
                continue
            discrepancy = float(np.max(np.abs(np.asarray(mol.GetConformer().GetPositions()) - coords[i])))
            if discrepancy > 5.1e-5:
                raise ValueError(f"SDF is not rounded raw output: {discrepancy}")
            tasks.append(dict(entry, molblock=Chem.MolToMolBlock(mol),
                              protein=teacher["protein"], sdf_raw_coordinate_max_error_A=discrepancy,
                              kind="generated"))
    # Teachers are measured as-is, not presumed valid. The intentionally broken
    # control verifies geometric detection, not a new scientific sample.
    for subject in subjects:
        teacher = subject["teacher"]
        for path in [teacher["sdf"], teacher["protein"]]:
            hashes[path] = sha(path)
        mol = next(m for m in Chem.SDMolSupplier(teacher["sdf"], removeHs=True) if m is not None)
        for kind in ["teacher_reference", "distorted_control"]:
            control = Chem.Mol(mol)
            if kind == "distorted_control":
                p = np.asarray(control.GetConformer().GetAtomPosition(0)) + np.asarray([5., 0., 0.])
                control.GetConformer().SetAtomPosition(0, p)
            tasks.append(dict(candidate_key=f"{teacher['target']}:{kind}", target=teacher["target"],
                              kind=kind, molblock=Chem.MolToMolBlock(control), protein=teacher["protein"]))
    buster = PoseBusters("dock")
    pkg = Path(posebusters.__file__).parent
    provenance = dict(timestamp=now(), source_root=str(root), workers=args.workers,
                      new_candidates=0, model_forwards=0, llm_calls=0,
                      scope="all saved 13 coarse groups, each 10/10; no 50-attempt population exists",
                      posebusters_version=getattr(posebusters, "__version__", "unavailable"),
                      posebusters_path=str(pkg), rdkit_version=rdBase.rdkitVersion,
                      config=buster.config, check_columns=expected_checks(buster.config),
                      package_source_sha256={str(p.relative_to(pkg)): sha(p) for p in pkg.rglob("*")
                                             if p.is_file() and p.suffix in [".py", ".yaml", ".yml"]},
                      source_sha256=hashes, frozen_artifacts_before=initial,
                      runner_sha256=sha(__file__))
    write(args.out / "PROVENANCE.json", provenance)
    write(args.out / "ATTEMPTS.json", attempts)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(evaluate, task): task["candidate_key"] for task in tasks}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            write(args.out / f"result_{len(results):03d}.json", result)
            print(json.dumps({"completed": len(results), "total": len(tasks),
                              "candidate": result["candidate_key"], "status": result["status"]}), flush=True)
    results.sort(key=lambda r: r["candidate_key"])
    write(args.out / "RESULTS.json", results)
    final = verify_original(root, audit)
    changed = [p for p, digest in hashes.items() if sha(p) != digest]
    write(args.out / "COMPLETE.json", dict(timestamp=now(), attempts=len(attempts),
          built=sum(r["build_success"] for r in attempts), evaluated_tasks=len(results),
          statuses=dict(Counter(r["status"] for r in results)),
          frozen_artifacts_after=final, candidate_and_reference_sources_changed=changed,
          new_candidates=0, model_forwards=0, llm_calls=0))
    if final["changed"] or changed:
        raise ValueError("Sources changed during audit")


if __name__ == "__main__":
    main()
