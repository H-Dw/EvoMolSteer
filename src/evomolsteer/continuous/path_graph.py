"""Sparse, clock-explicit lineage graph inside a dynamic coordinate window.

Native transport and resampling copies are separate edges. Coordinates remain
in the immutable input trajectory; integer locators make this graph rebuildable.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import digest, read_json, write_json
from ..trajectory_source import open_trajectory, trajectory_paths

FORMAT = "coordinate-path-graph-1.0"
CURRENT, PROPOSAL = 0, 1
NATIVE, COPY = 0, 1


def validate_lineage(a):
    selected = np.asarray(a["selected_indices"])
    roots = np.asarray(a["root_slot"])
    clock = np.asarray(a["score_time"], float)
    clock = clock[:, 0] if clock.ndim == 2 else clock
    state = np.asarray(a["state_time"], float)
    if selected.ndim != 2 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Integer T,N selected-parent indices required")
    steps, n = selected.shape
    if roots.shape != selected.shape or clock.shape != (steps,) or state.shape != (steps,):
        raise ValueError("Genealogy and clock dimensions differ")
    if n < 1 or not np.isfinite(clock).all() or not np.isfinite(state).all():
        raise ValueError("Finite clocks and nonempty population required")
    if np.any(np.diff(clock) <= 0) or np.any(state <= clock) or np.any((selected < 0) | (selected >= n)):
        raise ValueError("Invalid forward clocks or parent indices")
    if not np.allclose(state[:-1], clock[1:], rtol=0, atol=2e-6):
        raise ValueError("Proposal clock must match next current-state clock")
    if not np.array_equal(roots[1:], np.take_along_axis(roots[:-1], selected[:-1], axis=1)):
        raise ValueError("Root propagation disagrees with selected-parent indices")
    offspring = np.asarray(a["offspring_count"])
    expected = np.stack([np.bincount(v, minlength=n) for v in selected])
    if not np.array_equal(offspring, expected):
        raise ValueError("Offspring counts disagree with copy edges")
    resampled = np.asarray(a["resampled"], bool)
    if resampled.shape != (steps,) or not np.array_equal(selected[~resampled],
                                                         np.broadcast_to(np.arange(n), selected[~resampled].shape)):
        raise ValueError("Nonselection rows must retain native identity")
    return clock, state, selected, roots, offspring, resampled


def graph_batch(a, window, source_index=0, node_offset=0, verify_copies=True):
    start, end = map(float, window)
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= 1:
        raise ValueError("Dynamic nonempty inference window required")
    clock, state, selected, roots, offspring, resampled = validate_lineage(a)
    current = np.flatnonzero((clock >= start-2e-6) & (clock <= end+2e-6))
    learning = np.flatnonzero((clock >= start-2e-6) & (state <= end+2e-6) & resampled)
    if not len(learning) or not len(current) or np.any(np.diff(learning) != 1):
        raise ValueError("Contiguous actual selection support required")
    if learning[-1]+1 not in current:
        raise ValueError("Boundary current state must be observable in source")
    n = selected.shape[1]
    if verify_copies:
        x, y = np.asarray(a["current_coords"]), np.asarray(a["proposal_coords"])
        copies = y[learning[:, None], selected[learning]]
        if not np.array_equal(x[learning+1], copies, equal_nan=True):
            raise ValueError("Coordinate copy edges do not reproduce next current states")
    # Local node IDs encode the original step/slot and representation; they do
    # not depend on which candidates survived or on a floating-point match.
    node_id = lambda step, slot, kind: node_offset+2*(step*n+slot)+kind
    nodes, edges = [], []
    on, off = np.asarray(a["pic50_on"], float), np.asarray(a["pic50_off"], float)
    probability = np.asarray(a["selection_probability"], float)
    for step in current:
        for slot in range(n):
            nodes.append((node_id(step, slot, CURRENT), source_index, int(step), slot, CURRENT,
                          float(clock[step]), float(clock[step]), int(roots[step, slot]),
                          float(on[step, slot]), float(off[step, slot]),
                          float(probability[step, slot]), step in learning))
    for step in learning:
        for slot in range(n):
            parent = int(selected[step, slot])
            nodes.append((node_id(step, slot, PROPOSAL), source_index, int(step), slot, PROPOSAL,
                          float(clock[step]), float(state[step]), int(roots[step, slot]),
                          float(on[step, slot]), float(off[step, slot]),
                          float(probability[step, slot]), True))
            edges.append((node_id(step, slot, CURRENT), node_id(step, slot, PROPOSAL), NATIVE,
                          float(state[step]-clock[step]), 1))
            edges.append((node_id(step, parent, PROPOSAL), node_id(step+1, slot, CURRENT), COPY,
                          0., int(offspring[step, parent])))
    columns = ["node_id", "source_index", "step", "slot", "representation", "score_time", "state_time",
               "root_slot", "online_pic50_on", "online_pic50_off", "selection_probability", "learning_node"]
    nodes = pd.DataFrame(nodes, columns=columns)
    edges = pd.DataFrame(edges, columns=["parent_id", "child_id", "kind", "dt", "parent_offspring"])
    if len(nodes) and nodes.node_id.max() > np.iinfo(np.int32).max:
        raise ValueError("Graph integer budget exceeded")
    for field in ["node_id", "source_index", "step", "slot", "root_slot"]:
        nodes[field] = nodes[field].astype(np.int32)
    nodes["representation"] = nodes.representation.astype(np.int8)
    for field in ["parent_id", "child_id", "parent_offspring"]:
        edges[field] = edges[field].astype(np.int32)
    edges["kind"] = edges.kind.astype(np.int8)
    if len(np.unique(nodes.node_id)) != len(nodes) or not set(edges.parent_id).union(edges.child_id).issubset(set(nodes.node_id)):
        raise ValueError("Graph identity/connectivity audit failed")
    return nodes, edges, {"learning_score_times": np.round(clock[learning], 6).tolist(),
                          "boundary_current_time": float(clock[learning[-1]+1]),
                          "actual_selection_score_times": np.round(clock[resampled], 6).tolist(),
                          "coordinate_copies_verified": verify_copies,
                          "next_offset": node_offset+2*n*len(clock)}


def build_path_graph(dataset, campaign, output, window, arm="single", batches=None):
    root, out = Path(dataset).resolve(), Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh graph output")
    folder = root/"results"/campaign
    read_json(folder/"config.json")
    paths = [p for p in trajectory_paths(folder) if p.parent.parent.name == arm]
    if batches is not None:
        paths = [p for p in paths if int(p.parent.name.split("_")[-1]) in batches]
    if not paths:
        raise ValueError("No requested lineage sources")
    ns, es, sources, offset = [], [], [], 0
    for source_index, path in enumerate(paths):
        before = digest(path)
        with open_trajectory(path) as arrays:
            nodes, edges, info = graph_batch(arrays, window, source_index, offset)
        if digest(path) != before:
            raise ValueError("Original trajectory changed during graph extraction")
        ns.append(nodes); es.append(edges); offset = info.pop("next_offset")
        sources.append({"source_index": source_index, "path": path.relative_to(root).as_posix(),
                        "sha256": before, "batch": int(path.parent.name.split("_")[-1]), "arm": arm, **info})
    if any(s["learning_score_times"] != sources[0]["learning_score_times"] for s in sources):
        raise ValueError("Source learning clocks differ")
    nodes, edges = pd.concat(ns, ignore_index=True), pd.concat(es, ignore_index=True)
    out.mkdir(parents=True)
    nodes.to_parquet(out/"nodes.parquet", compression=None, index=False)
    edges.to_parquet(out/"edges.parquet", compression=None, index=False)
    manifest = {"schema_version": FORMAT, "window": list(map(float, window)), "campaign": campaign,
                "sources": sources, "nodes": len(nodes), "edges": len(edges),
                "source_code_sha256": digest(__file__), "representations": {"0": "current", "1": "proposal"},
                "edge_kinds": {"0": "native_transport", "1": "resampling_copy"},
                "online_score_semantics": "Joint forward forecast at score_time, not decoded terminal rescore",
                "storage": "Uncompressed Parquet, integer structure locators; no repeated coordinates",
                "tables": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in out.glob("*.parquet")}}
    write_json(out/"manifest.json", manifest)
    return manifest
