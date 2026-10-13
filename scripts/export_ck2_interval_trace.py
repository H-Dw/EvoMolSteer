#!/usr/bin/env python
"""Export private TOP30 terminal-to-intermediate lineage provenance tables.

These exports are oracle-only and must never be passed to Scientist or
Selector workers. Intermediate states are identified by the exact source tuple
(batch, step, ancestor_slot) and the hashed trajectory locator; the frozen
ancestry table does not define separate intermediate node IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_STUDY = Path("docs/experiments/ck2_scientist_selector_20261013")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _int(value: Any, label: str) -> int:
    if value is None or pd.isna(value):
        raise ValueError(f"missing required integer {label}")
    return int(value)


def _locator_fields(row: dict) -> dict:
    loc = row.get("locator")
    if not isinstance(loc, dict):
        raise ValueError(f"ancestry row for {row.get('node_id')} step {row.get('step')} has no locator struct")
    required = ("path", "sha256", "step", "slot")
    if any(loc.get(k) is None for k in required):
        raise ValueError(f"incomplete source locator for {row.get('node_id')} step {row.get('step')}")
    return {
        "path": str(loc["path"]), "sha256": str(loc["sha256"]),
        "step": _int(loc["step"], "locator.step"), "slot": _int(loc["slot"], "locator.slot"),
        "parent_slot": None if loc.get("parent_slot") is None or pd.isna(loc.get("parent_slot")) else int(loc["parent_slot"]),
        "channel": loc.get("channel"),
    }


def build_trace_tables(top30_rows: list[dict], teacher_map: dict, ancestry_rows: list[dict],
                       oracle_pools: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    """Return (300-row interval trace, 1530-row exact step trace)."""
    if len(top30_rows) != 30 or len(teacher_map) != 30:
        raise ValueError("expected exactly 30 top30 candidates and teacher mappings")
    by_terminal = {}
    for row in ancestry_rows:
        by_terminal.setdefault(str(row["node_id"]), []).append(row)
    donors_by_node = {}
    for donor_id, item in teacher_map.items():
        node_id = str(item["node_id"])
        if node_id in donors_by_node:
            raise ValueError(f"duplicate teacher mapping for terminal {node_id}")
        donors_by_node[node_id] = (str(donor_id), item)

    interval_rows, full_rows = [], []
    seen_node_ids = set()
    for candidate in sorted(top30_rows, key=lambda r: _int(r.get("rank"), "rank")):
        node_id = str(candidate["node_id"])
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate TOP30 terminal node_id {node_id}")
        seen_node_ids.add(node_id)
        if node_id not in donors_by_node or node_id not in by_terminal:
            raise ValueError(f"TOP30 terminal lacks teacher or ancestry mapping: {node_id}")
        donor_id, mapped = donors_by_node[node_id]
        batch, final_slot = _int(candidate["batch"], "batch"), _int(candidate["slot"], "slot")
        if (_int(mapped["batch"], "teacher batch") != batch or
                _int(mapped["slot"], "teacher slot") != final_slot):
            raise ValueError(f"teacher mapping disagrees with TOP30 candidate {node_id}")
        rows = [r for r in by_terminal[node_id]
                if _int(r.get("batch"), "ancestry batch") == batch and
                _int(r.get("slot"), "ancestry final slot") == final_slot]
        by_step = {}
        for row in rows:
            step = _int(row.get("step"), "ancestry step")
            if step in by_step:
                raise ValueError(f"duplicate ancestry step {step} for {node_id}")
            by_step[step] = row
        if set(by_step) != set(range(101)):
            raise ValueError(f"{node_id}: expected one exact ancestry row for every step 0..100")
        final_locator = _locator_fields(by_step[100])
        if (final_locator["step"] != 100 or final_locator["slot"] != final_slot or
                _int(by_step[100].get("ancestor_slot"), "final readout ancestor_slot") != _int(
                    final_locator["parent_slot"], "readout parent_slot")):
            raise ValueError(f"{node_id}: final readout locator parent disagrees with ancestry")

        # Exact source-step trace for every observation consumed by Scientist.
        for step in range(51):
            row = by_step[step]
            loc = _locator_fields(row)
            ancestor_slot = _int(row["ancestor_slot"], "ancestor_slot")
            if loc["step"] != step or loc["slot"] != ancestor_slot:
                raise ValueError(f"{node_id} step {step}: locator does not resolve to the recorded ancestor slot")
            full_rows.append({
                "donor_id": donor_id, "rank": _int(candidate["rank"], "rank"),
                "terminal_node_id": node_id, "batch": batch, "final_slot": final_slot,
                "step": step, "time": float(row["time"]),
                "ancestor_slot": ancestor_slot, "root_slot": _int(row["root_slot"], "root_slot"),
                "selected_parent_slot": _int(row["selected_parent_slot"], "selected_parent_slot"),
                "state_locator_key": f"b{batch:03d}:t{step:03d}:s{ancestor_slot:03d}",
                "source_path": loc["path"], "source_sha256": loc["sha256"],
                "source_step": loc["step"], "source_slot": loc["slot"],
            })

        readout_loc = _locator_fields(by_step[100])
        for interval in range(1, 11):
            lo, hi = 5 * (interval - 1), 5 * interval
            mid = lo + (hi - lo) // 2
            end_row = by_step[hi]
            pool = oracle_pools.get(hi)
            if not isinstance(pool, dict):
                raise ValueError(f"missing oracle pool for interval end step {hi}")
            targets = [t for t in pool.get("targets", []) if str(t.get("node_id")) == node_id]
            if len(targets) != 1:
                raise ValueError(f"{node_id} must have one oracle target mapping at step {hi}; found {len(targets)}")
            target = targets[0]
            if not isinstance(target.get("key"), str) or not target["key"]:
                raise ValueError(f"{node_id} has no oracle group key at step {hi}")
            if (_int(target.get("batch"), "oracle target batch") != batch or
                    _int(target.get("slot"), "oracle target slot") != final_slot or
                    _int(target.get("ancestor_slot"), "oracle target ancestor") != _int(end_row["ancestor_slot"], "end ancestor")):
                raise ValueError(f"{node_id} oracle target ancestry does not match exact step {hi}")

            out = {"rank": _int(candidate["rank"], "rank"), "donor_id": donor_id,
                   "terminal_node_id": node_id, "batch": batch, "final_slot": final_slot,
                   "interval": interval, "start_step": lo, "mid_step": mid, "end_step": hi,
                   "end_oracle_group_key": str(target["key"]),
                   "oracle_pool_step": hi,
                   "final_readout_path": readout_loc["path"],
                   "final_readout_sha256": readout_loc["sha256"],
                   "final_readout_parent_slot": readout_loc["parent_slot"]}
            for label, step in (("start", lo), ("mid", mid), ("end", hi)):
                row = by_step[step]
                loc = _locator_fields(row)
                anc = _int(row["ancestor_slot"], "ancestor_slot")
                if loc["step"] != step or loc["slot"] != anc:
                    raise ValueError(f"{node_id} step {step}: locator does not match ancestry")
                out.update({
                    f"{label}_ancestor_slot": anc,
                    f"{label}_root_slot": _int(row["root_slot"], "root_slot"),
                    f"{label}_selected_parent_slot": _int(row["selected_parent_slot"], "selected_parent_slot"),
                    f"{label}_state_locator_key": f"b{batch:03d}:t{step:03d}:s{anc:03d}",
                    f"{label}_source_path": loc["path"],
                    f"{label}_source_sha256": loc["sha256"],
                    f"{label}_source_step": loc["step"],
                    f"{label}_source_slot": loc["slot"],
                })
            interval_rows.append(out)

    if seen_node_ids != set(donors_by_node):
        raise ValueError("teacher map and TOP30 terminal node sets differ")
    if len(interval_rows) != 300 or len(full_rows) != 1530:
        raise ValueError(f"unexpected export row counts: interval={len(interval_rows)}, full={len(full_rows)}")
    if len({(r["donor_id"], r["interval"]) for r in interval_rows}) != 300:
        raise ValueError("interval export does not contain 30 unique donors x 10 windows")
    if len({(r["donor_id"], r["step"]) for r in full_rows}) != 1530:
        raise ValueError("full trace does not contain 30 unique donors x 51 states")
    return interval_rows, full_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty trace: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def export(study: Path) -> dict:
    study = Path(study).resolve()
    protocol_path = study / "protocol.json"
    top_path = study / "oracle/top30.csv"
    ancestry_path = study / "oracle/ancestry.parquet"
    teacher_path = study / "oracle/teacher_map.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    top30 = pd.read_csv(top_path).to_dict(orient="records")
    ancestry = pd.read_parquet(ancestry_path).to_dict(orient="records")
    teacher_map = json.loads(teacher_path.read_text(encoding="utf-8"))
    oracle_pools = {s: json.loads((study / f"oracle/pools/t{s:03d}.json").read_text(encoding="utf-8"))
                    for s in range(5, 51, 5)}

    inputs = [protocol_path, top_path, ancestry_path, teacher_path]
    inputs.extend(study / f"oracle/pools/t{s:03d}.json" for s in range(5, 51, 5))
    frozen_manifest = protocol.get("artifact_manifest", {})
    source_hashes = {}
    for path in inputs:
        rel = path.relative_to(study).as_posix()
        digest = _sha256(path)
        if rel != "protocol.json":
            frozen = frozen_manifest.get(rel)
            if frozen is None or frozen.get("sha256") != digest:
                raise ValueError(f"frozen artifact hash mismatch or missing manifest entry: {rel}")
        source_hashes[rel] = {"sha256": digest, "bytes": path.stat().st_size}

    interval_rows, full_rows = build_trace_tables(top30, teacher_map, ancestry, oracle_pools)
    interval_path = study / "oracle/top30_interval_trace.csv"
    full_path = study / "oracle/top30_full_step_trace.csv"
    readme_path = study / "oracle/TOP30_TRACE_README.md"
    _write_csv(interval_path, interval_rows)
    _write_csv(full_path, full_rows)
    readme_path.write_text(
        "# TOP30 interval lineage trace (private oracle artifact)\n\n"
        "These files are for human audit and must not be sent to Scientist or Selector workers. "
        "`top30_interval_trace.csv` has one row per terminal candidate and 5-step interval, "
        "with exact start/mid/end ancestry slots, roots, selected-parent slots, source locators and hashes, "
        "plus the end-step oracle group key. `top30_full_step_trace.csv` expands each candidate over "
        "steps 0–50 (30 × 51 = 1,530 rows).\n\n"
        "The frozen ancestry table provides terminal `node_id` values and intermediate `ancestor_slot` "
        "values; it does not define separate intermediate node IDs. Therefore `state_locator_key` is the "
        "explicit tuple `batch:step:ancestor_slot`, and `source_path`/`source_sha256` resolve it to the "
        "original trajectory. `final_readout_*` fields link each terminal to its separate readout file.\n\n"
        "Join the interval file to `top30.csv` by `terminal_node_id` for final candidate metadata, or to "
        "`teacher_map.json` by `donor_id` to verify the private donor mapping. Join full-step trace to "
        "interval rows by `(donor_id, step)` where `step` equals an anchor.\n",
        encoding="utf-8",
    )
    outputs = [interval_path, full_path, readme_path]
    manifest = {
        "format": "evomolsteer.ck2_top30_interval_trace.v1",
        "private_oracle_only": True,
        "study": str(study),
        "counts": {"terminals": 30, "interval_rows": len(interval_rows), "full_step_rows": len(full_rows),
                   "intervals": 10, "full_trace_steps": [0, 50]},
        "source_hashes": source_hashes,
        "output_hashes": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size} for p in outputs},
    }
    manifest_path = study / "oracle/top30_interval_trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    manifest = export(args.study)
    print(json.dumps({"study": manifest["study"], "counts": manifest["counts"],
                      "outputs": list(manifest["output_hashes"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
