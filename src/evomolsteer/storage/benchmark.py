"""Warm local-cache, paired queries; accuracy is checked separately from timing."""
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter_ns

import h5py
import numpy as np
import pyarrow
import pyarrow.parquet as pq

from ..io import write_json
from .arrays import byte_equal
from .features import FeaturePackage, table_digest
from .trajectory import TrajectoryPackage


def _measure(function, queries, repeats):
    for q in queries[:3]:
        function(q)
    samples = []
    for _ in range(repeats):
        for q in queries:
            start = perf_counter_ns()
            value = function(q)
            samples.append((perf_counter_ns() - start) / 1e6)
            del value
    return {"queries": len(samples), "median_ms": statistics.median(samples),
            "p95_ms": float(np.quantile(samples, .95)), "minimum_ms": min(samples)}


def benchmark(raw_root, trajectory_trials, feature_source, feature_packages, output, repeats=3):
    raw_root, trajectory_trials = Path(raw_root), Path(trajectory_trials)
    feature_packages = Path(feature_packages)
    rng = np.random.default_rng(20261004)
    queries = [(int(t), int(s)) for t, s in zip(rng.integers(0, 100, 24), rng.integers(0, 50, 24))]
    fields = ["current_coords", "proposal_coords", "predicted_coords", "current_bonds", "predicted_bonds",
              "predicted_atomics_probs_f16", "pic50_on", "pic50_off", "offspring_count"]
    raw_results = []
    for arm in ["joint", "single", "unguided"]:
        source = raw_root / arm / "batch_000/trajectory.npz"
        package_path = trajectory_trials / (arm + "_normalized.h5")
        dense_path = trajectory_trials / (arm + "_dense.h5")
        def source_query(q):
            with np.load(source, allow_pickle=False) as z:
                return {key: z[key][q] for key in fields}
        def package_query(q, path=package_path):
            with TrajectoryPackage(path) as package:
                return {key: package.read(key, *q) for key in fields}
        def dense_query(q):
            return package_query(q, dense_path)
        for q in queries:
            a, b, c = source_query(q), package_query(q), dense_query(q)
            if any(not byte_equal(a[k], b[k]) or not byte_equal(a[k], c[k]) for k in fields):
                raise AssertionError("Raw node query differs")
        with np.load(source, allow_pickle=False) as z:
            memory = {k: z[k] for k in z.files}
        def memory_query(q):
            return {k: memory[k][q] for k in fields}
        def source_full(_):
            with np.load(source, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        def package_full(_):
            with TrajectoryPackage(package_path) as package:
                return {k: package.read(k) for k in package.keys}
        raw_results.append({"arm": arm, "npz_node": _measure(source_query, queries, repeats),
                            "dense_hdf5_node": _measure(dense_query, queries, repeats),
                            "normalized_hdf5_node": _measure(package_query, queries, repeats),
                            "preloaded_npz_node": _measure(memory_query, queries, repeats),
                            "npz_full_batch": _measure(source_full, [None], 9),
                            "normalized_hdf5_full_batch": _measure(package_full, [None], 9),
                            "preloaded_array_bytes": sum(a.nbytes for a in memory.values())})
    manifest = json.loads((feature_packages / "manifest.json").read_text())
    reader = pq.ParquetFile(feature_source)
    feature_results = []
    for arm in ["joint", "single", "unguided"]:
        item = next(r for r in manifest["packages"] if r["arm"] == arm and r["batch"] == 0)
        path = feature_packages / item["package"]
        columns = ["node_id", "pic50_on", "pic50_off", "selected", "offspring_count", "geometry_time"] + [c for c in reader.schema_arrow.names if "::" in c]
        group_index = item["source_row_groups"][1]
        rep = "predicted_endpoint"
        def source_query(q):
            return reader.read_row_group(group_index, columns=columns).slice(q[0]*50+q[1], 1)
        def package_query(q):
            with FeaturePackage(path) as package:
                return package.node(rep, *q, columns=columns)
        for q in queries:
            if table_digest(source_query(q)) != table_digest(package_query(q)):
                raise AssertionError("Feature node query differs")
        loaded = reader.read_row_group(group_index, columns=columns)
        def memory_query(q):
            return loaded.slice(q[0]*50+q[1], 1)
        def source_full(_):
            return reader.read_row_group(group_index)
        def package_full(_):
            with FeaturePackage(path) as package:
                return package.table(rep)
        geometry_columns = [c for c in columns if "::" in c]
        def arrow_geometry(table):
            return (np.column_stack([table[c].fill_null(0).to_numpy() for c in geometry_columns]),
                    np.column_stack([table[c].is_valid().to_numpy(zero_copy_only=False) for c in geometry_columns]))
        def source_geometry(q):
            table = reader.read_row_group(group_index, columns=geometry_columns)
            if q is not None:
                table = table.slice(q[0]*50+q[1],1)
            return arrow_geometry(table)
        def package_geometry(q):
            with FeaturePackage(path) as package:
                values, valid = package.geometry(rep, None if q is None else q[0]*50+q[1])
                return np.atleast_2d(values), np.atleast_2d(valid)
        for q in [queries[0], None]:
            a, b = source_geometry(q), package_geometry(q)
            if not all(byte_equal(x,y) for x,y in zip(a,b)):
                raise AssertionError("NumPy geometry arrays or masks differ")
        feature_results.append({"arm": arm, "columns_per_node": len(columns),
                                "parquet_node": _measure(source_query, queries, repeats),
                                "normalized_hdf5_node": _measure(package_query, queries, repeats),
                                "preloaded_parquet_node": _measure(memory_query, queries, repeats),
                                "parquet_one_representation": _measure(source_full, [None], 9),
                                "normalized_hdf5_one_representation": _measure(package_full, [None], 9),
                                "parquet_numpy_geometry_node": _measure(source_geometry, queries, repeats),
                                "normalized_hdf5_numpy_geometry_node": _measure(package_geometry, queries, repeats),
                                "parquet_numpy_geometry_batch": _measure(source_geometry, [None], 9),
                                "normalized_hdf5_numpy_geometry_batch": _measure(package_geometry, [None], 9)})
    result = {"seed": 20261004, "platform": platform.platform(), "python": platform.python_version(),
              "numpy": np.__version__, "h5py": h5py.__version__, "pyarrow": pyarrow.__version__,
              "cache_protocol": "Warm OS cache; no cache flushing. Three warm-ups; fixed 24 queries x repeats. Single process, no concurrent benchmark workers.",
              "open_protocol": "NPZ and HDF5 reopen per node; the ParquetFile metadata handle is kept open. Preloaded baselines exclude initial I/O and decode.",
              "timing_scope": "Local end-to-end read/reconstruction/API time, not storage-device latency; no remote I/O or cold-cache claims.",
              "raw_fields": fields, "raw": raw_results, "features": feature_results}
    write_json(output, result)
    return result
