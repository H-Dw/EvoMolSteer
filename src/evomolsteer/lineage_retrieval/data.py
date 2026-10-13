"""Build the frozen, auditable evidence package for lineage retrieval studies."""
from __future__ import annotations

import hashlib
import json
import math
import secrets
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FORMAT = "evomolsteer.ck2_lineage_evidence.v1"
_VISIBLE_FORBIDDEN = ("batch", "slot", "root", "alias_count", "offspring",
                      "selection_probability", "weight", "final_score", "node_id")


def _visible_source(row: dict) -> dict:
    allow = {
        "online_pic50", "radius_gyration_A", "centroid_pocket_centroid_A", "centroid_nearest_pocket_A",
        "shape_thickness_A", "shape_planarity_fraction", "shape_asphericity",
        "nearest_pocket_A_min", "nearest_pocket_A_p05", "nearest_pocket_A_median",
        "nearest_pocket_A_mean", "nearest_pocket_A_p95", "ligand_contacts_3A",
        "ligand_contacts_4A", "ligand_contacts_5A", "ligand_contact_fraction_4A",
        "pocket_contacts_4A", "residue_contacts_4A", "contact_pairs_4A",
        "clash_fraction_1p2A", "clash_pairs_1p2A", "aromatic_atom_fraction",
        "aromatic_rings", "rotatable_bonds", "hbd", "hba", "tpsa", "logp",
        "fraction_sp3", "n_active_heavy", "nearest_pocket_A_mean",
        "ck2:A:ASN117::hetero_distance_softmin", "ck2:A:VAL116::hetero_distance_softmin",
    }
    return {k: _scalar(v) for k, v in row.items()
            if k in allow and not any(token in str(k).lower() for token in _VISIBLE_FORBIDDEN)}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _artifact_manifest(study: Path) -> dict[str, dict[str, Any]]:
    """Hash only immutable prepared inputs; runtime outputs may be added later."""
    study = Path(study)
    fixed = [study / "evidence/features.parquet", study / "evidence/teacher.json",
             study / "oracle/top30.csv", study / "oracle/teacher_map.json",
             study / "oracle/ancestry.parquet"]
    fixed.extend((study / "evidence/pools").glob("t*.json"))
    fixed.extend((study / "oracle/pools").glob("t*.json"))
    paths = sorted({p for p in fixed if p.is_file()})
    return {p.relative_to(study).as_posix(): {"sha256": _sha(p), "bytes": p.stat().st_size}
            for p in paths}


def _verify_frozen_study(study: Path, source_hashes: dict, analysis_hashes: dict,
                         code_sha256: str) -> dict | None:
    protocol_path = study / "protocol.json"
    if not protocol_path.exists():
        if _artifact_manifest(study):
            raise FileExistsError(f"Study directory is a partial preparation without protocol.json: {study}")
        return None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("source_hashes") != source_hashes or
            protocol.get("analysis_hashes") != analysis_hashes or
            protocol.get("preparation_code_sha256") != code_sha256):
        raise RuntimeError("Frozen study inputs or preparation code changed; refusing to rebuild in place")
    expected = protocol.get("artifact_manifest")
    actual = _artifact_manifest(study)
    if expected != actual:
        raise RuntimeError("Frozen study artifacts changed or are incomplete; refusing to rebuild in place")
    return protocol


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def _feature_table(records: list[dict]) -> pa.Table:
    """Build a union-column Arrow table for interleaved current/forecast rows."""
    return pa.Table.from_pandas(pd.DataFrame.from_records(records), preserve_index=False)


def _scalar(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _sig(arrays, step: int, slot: int) -> str:
    h = hashlib.sha256()
    # Include all channels that are visible to the Scientist and any model
    # probabilities from the trajectory, preserving their exact stored bits.
    for key in ("current_coords", "predicted_coords", "mask", "current_atomics",
                "current_bonds", "current_charges", "predicted_atomics",
                "predicted_bonds", "predicted_charges", "predicted_atomics_probs_f16",
                "predicted_charges_probs_f16"):
        if key in arrays:
            a = np.ascontiguousarray(arrays[key][step, slot])
            h.update(key.encode() + b"\0" + str(a.dtype).encode() + b"\0")
            h.update(a.tobytes())
    # On-target score is part of the visible current observation and therefore
    # must participate in exact-card equivalence.
    h.update(np.asarray(arrays["pic50_on"][step, slot]).tobytes())
    return h.hexdigest()


def _geom(xyz: np.ndarray, pocket: np.ndarray, residues: np.ndarray, mask: np.ndarray) -> dict:
    xyz = np.asarray(xyz, dtype=float)[np.asarray(mask, dtype=bool)]
    if not len(xyz):
        return {}
    center = xyz.mean(axis=0)
    centered = xyz - center
    cov = centered.T @ centered / max(len(xyz), 1)
    eig, vec = np.linalg.eigh(cov)
    eig = np.maximum(eig, 0.)
    d = np.linalg.norm(xyz[:, None, :] - pocket[None, :, :], axis=2)
    hits = d < 4.0
    res_counts = {}
    for r in np.unique(residues[np.any(hits, axis=0)]):
        res_counts[str(r)] = int(np.count_nonzero(np.any(hits[:, residues == r], axis=1)))
    rg2 = float(eig.sum())
    return {
        "centroid_x_A": float(center[0]), "centroid_y_A": float(center[1]), "centroid_z_A": float(center[2]),
        "cov_xx_A2": float(cov[0, 0]), "cov_xy_A2": float(cov[0, 1]), "cov_xz_A2": float(cov[0, 2]),
        "cov_yy_A2": float(cov[1, 1]), "cov_yz_A2": float(cov[1, 2]), "cov_zz_A2": float(cov[2, 2]),
        "principal_axis": vec[:, 2].tolist(), "eigenvalues_A2": eig.tolist(),
        "radius_gyration_A": float(np.sqrt(rg2)),
        "planarity_fraction": float(eig[0] / max(eig.sum(), 1e-12)),
        "asphericity": float(eig[2] - .5 * (eig[1] + eig[0])),
        "nearest_pocket_A": float(d.min()), "ligand_contacts_4A": int(np.count_nonzero(np.any(hits, axis=1))),
        "pocket_contacts_4A": int(np.count_nonzero(np.any(hits, axis=0))),
        "contact_pairs_4A": int(hits.sum()), "clash_pairs_1p2A": int(np.count_nonzero(d < 1.2)),
        "residue_contact_footprint": sorted(res_counts, key=lambda r: (-res_counts[r], r))[:8],
    }


def _loc(root: Path, p: Path, sha: str, step: int, slot: int) -> dict:
    return {"path": p.relative_to(root).as_posix(), "sha256": sha, "step": int(step), "slot": int(slot)}


def _candidate_map(outputs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    term = pd.read_csv(outputs / "tables" / "terminal_all_features.csv")
    quality = pd.read_csv(outputs / "tables" / "high_quality_candidates.csv")
    eligible, p75 = _rank_top30(term)
    eligible.attrs["relief_p75"] = p75
    return eligible, quality


def _rank_top30(term: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    valid = term[(term.valid_connected == True) & np.isfinite(term.mmff_relief_per_heavy)]
    p75 = float(valid.mmff_relief_per_heavy.quantile(.75))
    eligible = term[(term.valid_connected == True) & (term.pb_full_pass == True) &
                    (term.energy_status == "converged") &
                    np.isfinite(term.mmff_relief_per_heavy) & np.isfinite(term.pic50_on_rescore) &
                    (term.mmff_relief_per_heavy <= p75)].copy()
    eligible = eligible.sort_values(["pic50_on_rescore", "mmff_relief_per_heavy", "node_id"],
                                    ascending=[False, True, True], kind="mergesort").head(30)
    return eligible, p75


def _validate_trajectory(arr: dict, batch: int, final_roots: np.ndarray | None = None) -> None:
    """Validate slot ancestry, propagated copies, root IDs, and the score clock."""
    ids = np.asarray(arr["selected_indices"], dtype=int)
    roots = np.asarray(arr["root_slot"], dtype=int)
    if ids.shape != (100, 50) or np.any(ids < 0) or np.any(ids >= 50):
        raise ValueError(f"batch {batch}: malformed selected_indices")
    if not np.isfinite(arr["current_coords"]).all() or not np.isfinite(arr["predicted_coords"]).all():
        raise ValueError(f"batch {batch}: nonfinite trajectory coordinates")
    if not np.allclose(arr["score_time"][:, 0], np.arange(100) / 100., rtol=0, atol=1e-7):
        raise ValueError(f"batch {batch}: score clock does not match 0..0.99")
    if not np.allclose(arr["score_time"][:, 0] + arr["step_size"], arr["state_time"], rtol=0, atol=1e-7):
        raise ValueError(f"batch {batch}: score/state clocks disagree")
    if not np.all(np.asarray(arr["offspring_count"]).sum(axis=1) == 50):
        raise ValueError(f"batch {batch}: offspring counts do not conserve 50 slots")
    for t in range(99):
        if not np.array_equal(arr["current_coords"][t + 1], arr["proposal_coords"][t, ids[t]]):
            raise ValueError(f"batch {batch}: coordinate copy mismatch at step {t}")
        if not np.array_equal(roots[t + 1], roots[t, ids[t]]):
            raise ValueError(f"batch {batch}: root copy mismatch at step {t}")
        for channel in ("atomics", "bonds", "charges"):
            if not np.array_equal(arr[f"current_{channel}"][t + 1], arr[f"proposal_{channel}"][t, ids[t]]):
                raise ValueError(f"batch {batch}: {channel} copy mismatch at step {t}")
    if final_roots is not None and not np.array_equal(roots[-1, ids[-1]], np.asarray(final_roots, dtype=int)):
        raise ValueError(f"batch {batch}: final readout roots disagree with terminal records")


def _ancestor_at(selected: np.ndarray, final_slot: int, cutoff: int) -> int:
    """Return the slot at ``cutoff`` on a final slot's exact selected lineage."""
    slot = int(final_slot)
    for t in range(selected.shape[0] - 1, int(cutoff) - 1, -1):
        slot = int(selected[t, slot])
    return slot


def _pool_groups(by_slot: dict[int, str]) -> dict[str, list[int]]:
    """Group slots by recursive exact history hash, within one batch."""
    out: dict[str, list[int]] = {}
    for slot, history_hash in by_slot.items():
        out.setdefault(history_hash, []).append(int(slot))
    return out


def _bond_edge_count(bonds: np.ndarray) -> int:
    """Count nonzero upper-triangle edges, independent of bond-type code."""
    return int(np.count_nonzero(np.triu(np.asarray(bonds), 1)))


def _charged_atom_count(labels: np.ndarray, mask: np.ndarray, charge_vocab: dict) -> int:
    labels, mask = np.asarray(labels), np.asarray(mask, dtype=bool)
    return int(np.count_nonzero(mask & (labels != charge_vocab["<PAD>"]) & (labels != charge_vocab["0"])))


def _compact_pool_feature(row: dict) -> dict:
    """Keep only fields used in current/forecast pool and teacher cards."""
    keys = {
        "time", "step", "representation", "online_pic50", "structure_hash", "history_hash", "locator", "channel",
        "radius_gyration_A", "centroid_pocket_centroid_A", "centroid_nearest_pocket_A",
        "shape_thickness_A", "shape_planarity_fraction", "shape_asphericity",
        "nearest_pocket_A_min", "nearest_pocket_A_p05", "nearest_pocket_A_median", "nearest_pocket_A_mean",
        "nearest_pocket_A_p95", "ligand_contacts_3A", "ligand_contacts_4A", "ligand_contacts_5A",
        "ligand_contact_fraction_4A", "pocket_contacts_4A", "residue_contacts_4A", "contact_pairs_4A",
        "clash_fraction_1p2A", "clash_pairs_1p2A", "aromatic_atom_fraction", "aromatic_rings",
        "rotatable_bonds", "hbd", "hba", "tpsa", "logp", "fraction_sp3", "n_active_heavy",
        "ck2:A:ASN117::hetero_distance_softmin", "ck2:A:VAL116::hetero_distance_softmin",
        "centroid_x_A", "centroid_y_A", "centroid_z_A", "cov_xx_A2", "cov_xy_A2", "cov_xz_A2",
        "cov_yy_A2", "cov_yz_A2", "cov_zz_A2", "principal_axis", "eigenvalues_A2", "planarity_fraction",
        "asphericity", "nearest_pocket_A", "ligand_contacts_4A", "pocket_contacts_4A", "contact_pairs_4A",
        "clash_pairs_1p2A", "residue_contact_footprint", "atom_count", "bond_edge_count",
        "charge_nonzero_count", "atom_N_count", "atom_O_count", "atom_F_count", "atom_Cl_count", "atom_Br_count",
        "delta_current_online_pic50", "delta_current_radius_gyration_A", "delta_current_nearest_pocket_A",
        "delta_current_ligand_contacts_4A", "delta_current_pocket_contacts_4A", "delta_current_contact_pairs_4A",
        "delta_current_atom_count", "delta_current_bond_edge_count", "delta_current_centroid_xyz_A",
        "delta_forecast_online_pic50", "delta_forecast_radius_gyration_A", "delta_forecast_nearest_pocket_A",
        "delta_forecast_ligand_contacts_4A", "delta_forecast_pocket_contacts_4A", "delta_forecast_contact_pairs_4A",
        "delta_forecast_atom_count", "delta_forecast_bond_edge_count", "delta_forecast_centroid_xyz_A",
        "forecast_revision_rms_A",
    }
    return {k: v for k, v in row.items() if k in keys}


def prepare(study: Path, workspace: Path) -> dict:
    """Prepare frozen source, lineage, teacher, and time-indexed evidence artifacts."""
    study, workspace = Path(study), Path(workspace)
    campaign = workspace / "data/raw/ck2_clk3_lineage_20261003/results/main1000_w050/single"
    output_candidates = [workspace / "outputs", workspace.parent / "outputs",
                         workspace.parent.parent / "outputs"]
    outputs = next((base / "ck2_original_steer_analysis_20261013"
                    for base in output_candidates
                    if (base / "ck2_original_steer_analysis_20261013").exists()),
                   output_candidates[1] / "ck2_original_steer_analysis_20261013")
    if not campaign.exists() or not outputs.exists():
        raise FileNotFoundError(f"CK2 source data not found (campaign={campaign}, outputs={outputs})")
    raw_paths = sorted(campaign.glob("batch_*/trajectory.npz"))
    if len(raw_paths) != 20:
        raise ValueError(f"Expected 20 trajectory batches, found {len(raw_paths)}")
    top30, _ = _candidate_map(outputs)
    terminal_source = pd.read_csv(outputs / "tables/terminal_all_features.csv")
    pocket_z = np.load(outputs / "features/pocket.npz", allow_pickle=False)
    pocket = np.asarray(pocket_z["xyz"], dtype=float)
    residues = np.asarray(pocket_z["residue_ids"]).astype(str)
    top_ids = set(top30.node_id.astype(str))

    sha_by_path = {p: _sha(p) for p in raw_paths}
    source_hashes = {p.relative_to(workspace).as_posix(): sha_by_path[p] for p in raw_paths}
    for batch_path in raw_paths:
        for name in ("final_prediction.pt.gz", "final_records.json", "molecules_all_built.sdf"):
            p = batch_path.parent / name
            source_hashes[p.relative_to(workspace).as_posix()] = _sha(p)
    for p in [campaign.parent / "config.json"] + [campaign.parent / f"frame_batch_{b:03d}.json" for b in range(20)]:
        source_hashes[p.relative_to(workspace).as_posix()] = _sha(p)
    feature_hashes = {"terminal_all_features.csv": _sha(outputs / "tables/terminal_all_features.csv"),
                      "high_quality_candidates.csv": _sha(outputs / "tables/high_quality_candidates.csv"),
                      "pocket.npz": _sha(outputs / "features/pocket.npz"),
                      "pocket_atoms.csv": _sha(outputs / "features/pocket_atoms.csv"),
                      "nodes": {f"nodes_{b:03d}.csv.gz": _sha(outputs / f"features/nodes_{b:03d}.csv.gz")
                                for b in range(20)}}
    code_sha256 = _sha(Path(__file__).resolve())
    frozen = _verify_frozen_study(study, source_hashes, feature_hashes, code_sha256)
    if frozen is not None:
        return frozen
    ancestry, terminal_records, target_rows = [], [], []
    histories = {}
    feature_lookup = {}
    raw_arrays = {}
    final_slot_ids = {}
    feature_path = study / "evidence/features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_writer = None
    for p in raw_paths:
        b = int(p.parent.name.split("_")[1])
        feature_rows = []
        node_frame = pd.read_csv(outputs / f"features/nodes_{b:03d}.csv.gz")
        batch_nodes = {(int(r["step"]), int(r["slot"]), str(r["representation"])): r
                       for r in node_frame.to_dict(orient="records")}
        z = np.load(p, allow_pickle=False)
        arr = {k: z[k] for k in z.files}
        raw_arrays[b] = arr
        final_roots = terminal_source[terminal_source.batch == b].sort_values("slot").root_slot.to_numpy()
        _validate_trajectory(arr, b, final_roots)
        final_records = json.loads((p.parent / "final_records.json").read_text(encoding="utf-8"))
        if len(final_records) != 50:
            raise ValueError(f"batch {b}: expected 50 final readout records")
        for rec in final_records:
            slot = int(rec["slot"])
            expected_parent = int(arr["selected_indices"][99, slot])
            expected_root = int(arr["root_slot"][99, expected_parent])
            if (rec.get("parent_step") != 99 or int(rec.get("parent_slot", -1)) != expected_parent or
                    int(rec.get("root_slot", -1)) != expected_root or
                    rec.get("node_id") != f"single_b{b:03d}_final_{slot:03d}"):
                raise ValueError(f"batch {b} slot {slot}: final readout ancestry mismatch")
            final_slot_ids[(b, slot)] = str(rec["node_id"])
        frames = json.loads((campaign.parent / f"frame_batch_{b:03d}.json").read_text())
        com = np.asarray(frames["target_com"], dtype=float)
        conf = json.loads((campaign.parent / "config.json").read_text())
        atom_vocab = conf["atom_vocabulary"]
        charge_vocab = conf["charge_vocabulary"]
        scale = float(conf.get("flowr_args", {}).get("coord_scale", conf.get("coord_scale", 1.0)))
        psha = sha_by_path[p]
        # Current and forecast rows are both emitted for every slot and step.
        for s in range(51):
            for slot in range(50):
                sig = _sig(arr, s, slot)
                parent = int(slot if s == 0 else arr["selected_indices"][s - 1, slot])
                hist_prev = "root" if s == 0 else histories[(b, s - 1, parent)]
                hist = hashlib.sha256((hist_prev + sig).encode()).hexdigest()
                histories[(b, s, slot)] = hist
                for representation, coord_key, atom_key, bond_key, charge_key in (
                    ("current", "current_coords", "current_atomics", "current_bonds", "current_charges"),
                    ("forecast", "predicted_coords", "predicted_atomics", "predicted_bonds", "predicted_charges")):
                    row = batch_nodes.get((s, slot, "predicted" if representation == "forecast" else "current"), {})
                    feat = _visible_source(row)
                    mask = arr["mask"][s, slot]
                    com_slot = com[slot] if com.ndim == 2 else com
                    xyz = arr[coord_key][s, slot].astype(float) * scale + com_slot
                    geom = _geom(xyz, pocket, residues, mask)
                    # Strictly numeric/channel features; no identity or selection metadata.
                    feat.update(geom)
                    feat.update({"time": float(s / 100), "step": s, "representation": representation,
                                 "online_pic50": float(arr["pic50_on"][s, slot]),
                                 "atom_count": int(mask.sum()), "bond_edge_count": _bond_edge_count(arr[bond_key][s, slot]),
                                 "charge_nonzero_count": _charged_atom_count(arr[charge_key][s, slot], mask, charge_vocab),
                                 "atom_N_count": int(np.count_nonzero((arr[atom_key][s, slot] == atom_vocab["N"]) & mask)),
                                 "atom_O_count": int(np.count_nonzero((arr[atom_key][s, slot] == atom_vocab["O"]) & mask)),
                                 "atom_F_count": int(np.count_nonzero((arr[atom_key][s, slot] == atom_vocab["F"]) & mask)),
                                 "atom_Cl_count": int(np.count_nonzero((arr[atom_key][s, slot] == atom_vocab["Cl"]) & mask)),
                                 "atom_Br_count": int(np.count_nonzero((arr[atom_key][s, slot] == atom_vocab["Br"]) & mask)),
                                 "structure_hash": sig, "history_hash": hist,
                                 "locator": _loc(workspace, p, psha, s, slot),
                                 "channel": representation})
                    feature_rows.append(feat)
                    feature_lookup[(b, s, slot, representation)] = feat
        # Attach signed changes against the actual selected predecessor, and revision
        # magnitude against the previous forecast along that same ancestry.
        for s in range(51):
            for slot in range(50):
                prev_slot = int(arr["selected_indices"][s - 1, slot]) if s else slot
                for rep in ("current", "forecast"):
                    curf = feature_lookup[(b, s, slot, rep)]
                    prevf = feature_lookup.get((b, s - 1, prev_slot, rep)) if s else None
                    for metric in ("online_pic50", "radius_gyration_A", "nearest_pocket_A", "ligand_contacts_4A",
                                   "pocket_contacts_4A", "contact_pairs_4A", "atom_count", "bond_edge_count"):
                        a, q = curf.get(metric), (prevf or {}).get(metric)
                        if a is not None and q is not None:
                            curf[f"delta_{rep}_{metric}"] = float(a) - float(q)
                        else:
                            curf[f"delta_{rep}_{metric}"] = None
                    if prevf and all(prevf.get(f"centroid_{axis}_A") is not None for axis in "xyz"):
                        curf[f"delta_{rep}_centroid_xyz_A"] = [
                            float(curf[f"centroid_{axis}_A"]) - float(prevf[f"centroid_{axis}_A"]) for axis in "xyz"]
                    else:
                        curf[f"delta_{rep}_centroid_xyz_A"] = None
                    if s:
                        mask = arr["mask"][s, slot].astype(bool)
                        parent_mask = arr["mask"][s - 1, prev_slot].astype(bool)
                        common_atoms = np.flatnonzero(mask & parent_mask)
                        old = arr["predicted_coords"][s - 1, prev_slot, common_atoms].astype(float)
                        new = arr["predicted_coords"][s, slot, common_atoms].astype(float)
                        curf["forecast_revision_rms_A"] = (float(np.sqrt(np.mean(np.sum(((new - old) * scale) ** 2, axis=1))))
                                                            if len(common_atoms) else None)
                    else:
                        curf["forecast_revision_rms_A"] = None
        # Stream one compact batch at a time to avoid holding a duplicate
        # Arrow table for all 102,000 observations in memory.
        output_rows = []
        for row in feature_rows:
            row = dict(row)
            prefix = "current_" if row["representation"] == "current" else "forecast_"
            common = {"time", "step", "representation", "online_pic50", "structure_hash", "history_hash", "locator", "channel"}
            for k in list(row):
                if k in common or k.startswith("delta_") or k == "forecast_revision_rms_A":
                    continue
                row[prefix + k] = row.pop(k)
            if "forecast_revision_rms_A" in row:
                row["delta_forecast_revision_rms_A"] = row.pop("forecast_revision_rms_A")
            output_rows.append(row)
        table = _feature_table(output_rows)
        if feature_writer is None:
            feature_writer = pq.ParquetWriter(feature_path, table.schema, compression="zstd")
        feature_writer.write_table(table)
        del table, output_rows
        # The pool compiler needs a smaller numeric projection, not the full
        # row duplicated in the feature cache.
        for key in feature_lookup:
            if key[0] == b:
                feature_lookup[key] = _compact_pool_feature(feature_lookup[key])
        del feature_rows
        # Exact final identity comes from terminal table node IDs; each terminal's path is the
        # selected parent chain, not a row-wise approximation.
        for slot in range(50):
            node = final_slot_ids[(b, slot)]
            chain = []
            cur = int(arr["selected_indices"][99, slot])
            for s in range(99, -1, -1):
                chain.append((s, cur))
                if s:
                    cur = int(arr["selected_indices"][s - 1, cur])
            chain.reverse()
            for s, sl in chain:
                ancestry.append({"node_id": node, "batch": b, "slot": slot, "step": s,
                                 "ancestor_slot": sl, "root_slot": int(arr["root_slot"][s, sl]),
                                 "time": float(arr["score_time"][s, 0]),
                                 "state_time": float(arr["state_time"][s]),
                                 "selected_parent_slot": int(arr["selected_indices"][s - 1, sl]) if s else sl,
                                 "locator": _loc(workspace, p, psha, s, sl)})
            ancestry.append({"node_id": node, "batch": b, "slot": slot, "step": 100,
                             "ancestor_slot": cur, "root_slot": int(arr["root_slot"][99, cur]),
                             "time": 1.0, "state_time": 1.0,
                             "locator": {"path": (p.parent / "final_prediction.pt.gz").relative_to(workspace).as_posix(),
                                         "sha256": source_hashes[(p.parent / "final_prediction.pt.gz").relative_to(workspace).as_posix()],
                                         "step": 100, "slot": slot, "parent_slot": cur,
                                         "channel": "final_readout"}, "readout": True})
        del batch_nodes, node_frame
    # Create one teacher per target candidate, with randomized opaque identity and exactly 51
    # observations along that candidate's true ancestry.
    rng = secrets.SystemRandom()
    teacher_rows, teacher_map = [], {}
    candidates = top30.sort_values("node_id")
    donor_ids = [f"T{i:02d}" for i in range(1, len(candidates) + 1)]
    rng.shuffle(donor_ids)
    for donor, (_, cand) in zip(donor_ids, candidates.iterrows()):
        b, slot = int(cand.batch), int(cand.slot)
        arr = raw_arrays[b]
        # The teacher presents evidence only through step 50; locate that
        # ancestor from the final readout and then walk its exact history.
        chain_slots = [_ancestor_at(arr["selected_indices"], slot, 50)]
        for s in range(50, 0, -1):
            chain_slots.append(int(arr["selected_indices"][s - 1, chain_slots[-1]]))
        chain_slots.reverse()
        observations = []
        for s, sl in enumerate(chain_slots):
            observations.append(_pool_view(feature_lookup[(b, s, sl, "current")],
                                           feature_lookup[(b, s, sl, "forecast")]))
        teacher_rows.append({"donor_id": donor, "observations": observations})
        teacher_map[donor] = {"node_id": str(cand.node_id), "batch": b, "slot": slot}
    # Build point-in-time pool snapshots. Group by recursive exact full-visible history hash;
    # group keys are freshly randomized on each preparation, while the oracle map remains private.
    oracle_pools = {}
    for s in range(5, 51, 5):
        per_batch = {}
        for b, arr in raw_arrays.items():
            per_batch[b] = _pool_groups({slot: histories[(b, s, slot)] for slot in range(50)})
        groups = {(b, h): slots for b, hmap in per_batch.items() for h, slots in hmap.items()}
        group_list, cards = [], []
        targets_for_step = []
        keys = {k: "g" + secrets.token_hex(8) for k in groups}
        for (b, h), slots in groups.items():
            rep = min(slots)
            key = keys[(b, h)]
            arr = raw_arrays[b]
            # features row per ancestry step as the exact visible history
            hist_feats = []
            cur = rep
            slots_by_step = []
            for t in range(s, -1, -1):
                slots_by_step.append((t, cur))
                if t:
                    cur = int(arr["selected_indices"][t - 1, cur])
            slots_by_step.reverse()
            for t, sl in slots_by_step:
                hist_feats.append(_pool_view(feature_lookup[(b, t, sl, "current")],
                                             feature_lookup[(b, t, sl, "forecast")]))
            cards.append({"key": key, "now": hist_feats[-1], "history": hist_feats})
            descendants = 0
            target_descendants = []
            for final_slot in range(50):
                ancestor = _ancestor_at(arr["selected_indices"], final_slot, s)
                if ancestor in slots:
                    descendants += 1
                    node_id = final_slot_ids[(b, final_slot)]
                    if node_id in top_ids:
                        target_descendants.append((node_id, final_slot, ancestor))
            targets = [v[0] for v in target_descendants]
            group_list.append({"key": key, "batch": b, "representative_slot": rep, "aliases": sorted(slots),
                               "targets": targets, "observed_descendants": descendants})
            for node_id, final_slot, ancestor in target_descendants:
                targets_for_step.append({"node_id": node_id, "batch": b, "slot": final_slot,
                                         "ancestor_slot": ancestor, "key": key})
        _json(study / "evidence" / "pools" / f"t{s:03d}.json", {"step": s, "time": s / 100, "cards": cards})
        oracle_pools[str(s)] = {"groups": group_list, "targets": targets_for_step}
    for s, obj in oracle_pools.items():
        _json(study / "oracle" / "pools" / f"t{int(s):03d}.json", obj)
    if feature_writer is not None:
        feature_writer.close()
    _records(study / "oracle/ancestry.parquet", ancestry)
    top30_records = []
    for rank, (_, r) in enumerate(top30.iterrows(), 1):
        report_fields = ("smiles", "heavy_atoms", "mw", "rings", "qed", "sa_score", "min_protein_distance_A",
                         "clash_pairs_1p2A", "radius_gyration_A", "hbd", "hba", "aromatic_rings")
        record = {"rank": rank, "node_id": str(r.node_id), "batch": int(r.batch), "slot": int(r.slot),
                              "pic50_on_rescore": float(r.pic50_on_rescore),
                                 "mmff_relief_per_heavy": float(r.mmff_relief_per_heavy),
                              "valid_connected": bool(r.valid_connected), "pb_full_pass": bool(r.pb_full_pass),
                              "energy_status": str(r.energy_status)}
        record.update({field: _scalar(r[field]) for field in report_fields if field in r.index})
        top30_records.append(record)
    pd.DataFrame(top30_records).to_csv(study / "oracle/top30.csv", index=False)
    _json(study / "oracle/teacher_map.json", teacher_map)
    _json(study / "evidence/teacher.json", {"donors": teacher_rows,
          "reference": {"pocket_atom_count": len(pocket), "pocket_sha256": feature_hashes["pocket.npz"],
                        "coordinate_frame": "x_world = coords * coord_scale + target_com"}})
    protocol = {"format": FORMAT, "source_hashes": source_hashes, "analysis_hashes": feature_hashes,
                "preparation_code_sha256": code_sha256,
                "top30_eligibility": "valid_connected & pb_full_pass & energy_status=converged & finite mmff_relief_per_heavy<=P75(valid_connected,finite relief) & finite pic50_on_rescore; sorted rescore descending, mmff_relief_per_heavy ascending, node_id ascending",
                "top30_relief_p75": float(top30.attrs["relief_p75"]),
                "feature_glossary": {
                    "planarity_fraction": "smallest covariance eigenvalue / eigenvalue sum; near 0 indicates a planar point cloud",
                    "shape_planarity_fraction": "cached shape summary with the same low-for-planar convention",
                    "principal_axis": "largest-eigenvalue eigenvector; its sign is arbitrary and sign flips do not indicate rotation",
                    "bond_edge_count": "nonzero upper-triangle raw bond-class entries; not a chemical-validity judgment",
                    "charge_nonzero_count": "active charge labels excluding PAD and neutral class 0",
                },
                "steps": list(range(51)), "pool_cutoffs": list(range(5, 51, 5)),
                "time_unit": "score_time, step/100", "batches": len(raw_paths),
                "slots_per_batch": 50, "coord_scale": "from campaign config", "coordinate_frame": "x_world=coords*coord_scale+target_com",
                "feature_rows": 20 * 51 * 50 * 2,
                "source_paths": {"campaign": campaign.relative_to(workspace).as_posix(), "analysis": str(outputs)}}
    protocol["artifact_manifest"] = _artifact_manifest(study)
    _json(study / "protocol.json", protocol)
    return protocol


def com_for_batch(campaign: Path, b: int) -> np.ndarray:
    frame = json.loads((campaign.parent / f"frame_batch_{b:03d}.json").read_text())
    return np.asarray(frame["target_com"], dtype=float)


def _pool_view(current: dict, forecast: dict) -> dict:
    """Combine contemporaneous channels while keeping geometry namespaces explicit."""
    view = {"time": current.get("time"), "step": current.get("step"),
            "online_pic50": current.get("online_pic50"), "graph_edges_are_raw_unvalidated": True}
    for rep, row in (("current", current), ("forecast", forecast)):
        for k, v in row.items():
            if k in ("time", "step", "representation", "online_pic50", "structure_hash", "history_hash", "locator", "channel"):
                continue
            if k.startswith("delta_") or k == "forecast_revision_rms_A":
                continue
            view[f"{rep}_{k}"] = v
    for rep, row in (("current", current), ("forecast", forecast)):
        view[f"delta_{rep}_Rg"] = row.get(f"delta_{rep}_radius_gyration_A")
        view[f"delta_{rep}_nearest_pocket_A"] = row.get(f"delta_{rep}_nearest_pocket_A")
        view[f"delta_{rep}_centroid_xyz_A"] = row.get(f"delta_{rep}_centroid_xyz_A")
        view[f"delta_{rep}_contact_pairs_4A"] = row.get(f"delta_{rep}_contact_pairs_4A")
        view[f"delta_{rep}_ligand_contacts_4A"] = row.get(f"delta_{rep}_ligand_contacts_4A")
    view["delta_forecast_revision_rms_A"] = current.get("forecast_revision_rms_A", forecast.get("forecast_revision_rms_A"))
    return view


def load_pool(study: Path, step: int) -> dict:
    if not 5 <= int(step) <= 50 or int(step) % 5:
        raise ValueError("pool step must be one of 5, 10, ..., 50")
    path = Path(study) / "evidence/pools" / f"t{int(step):03d}.json"
    return json.loads(path.read_text(encoding="utf-8"))
