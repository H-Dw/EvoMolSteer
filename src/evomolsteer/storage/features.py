"""Normalized feature packages: one node table, one exact geometry state pool.

Partition boundaries are campaign/arm/batch; each representation supplies node
to-state IDs. All source rows, order, columns, Arrow types, nulls and schema
metadata can be restored, while selected columns/nodes remain directly readable.
"""
import base64
import hashlib
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..io import digest, write_json
from .arrays import assert_no_filters, dataset, read_array, unique_rows, write_array

FORMAT = "evomolsteer.features.v1"


def column_digest(column):
    a = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    h = hashlib.sha256(str(a.type).encode())
    valid = a.is_valid().to_numpy(zero_copy_only=False)
    h.update(valid.tobytes())
    if pa.types.is_string(a.type) or pa.types.is_large_string(a.type):
        for value in a.drop_null().to_pylist():
            encoded = value.encode("utf-8")
            h.update(len(encoded).to_bytes(8, "little"))
            h.update(encoded)
    else:
        h.update(a.drop_null().to_numpy(zero_copy_only=False).tobytes())
    return h.hexdigest()


def table_digest(table):
    h = hashlib.sha256(table.schema.serialize().to_pybytes())
    for column in table.columns:
        h.update(bytes.fromhex(column_digest(column)))
    return h.hexdigest()


def _write_column(parent, key, column):
    a = column.combine_chunks()
    g = parent.create_group(key)
    g.attrs["sha256"] = column_digest(a)
    if pa.types.is_string(a.type) or pa.types.is_large_string(a.type):
        g.attrs["kind"] = "utf8_dictionary"
        values, ids, mapping = [], [], {}
        for value in a.to_pylist():
            if value not in mapping:
                mapping[value] = len(values)
                values.append(value)
            ids.append(mapping[value])
        payloads = [(v or "").encode("utf-8") for v in values]
        write_array(g, "ids", np.asarray(ids, dtype=np.int64))
        write_array(g, "offsets", np.r_[0, np.cumsum([len(v) for v in payloads])].astype(np.int64))
        dataset(g, "utf8", np.frombuffer(b"".join(payloads), dtype=np.uint8))
        write_array(g, "dictionary_valid", np.asarray([v is not None for v in values]))
    elif pa.types.is_integer(a.type) or pa.types.is_floating(a.type) or pa.types.is_boolean(a.type):
        g.attrs["kind"] = "numeric"
        valid = a.is_valid().to_numpy(zero_copy_only=False)
        values = a.fill_null(False if pa.types.is_boolean(a.type) else 0).to_numpy(zero_copy_only=False)
        write_array(g, "values", values)
        write_array(g, "valid", valid)
    else:
        raise TypeError(f"Unsupported source Arrow type: {a.type}")


def _read_column(g, arrow_type, row=None):
    prefix = () if row is None else (row,)
    if g.attrs["kind"] == "numeric":
        values = np.atleast_1d(read_array(g["values"], prefix))
        valid = np.atleast_1d(read_array(g["valid"], prefix))
        return pa.array(values, type=arrow_type, mask=~valid, from_pandas=False)
    ids = np.atleast_1d(read_array(g["ids"], prefix))
    if row is not None:
        i = int(ids[0])
        valid = bool(read_array(g["dictionary_valid"], (i,)))
        if not valid:
            return pa.array([None], type=arrow_type)
        lo = int(read_array(g["offsets"], (i,)))
        hi = int(read_array(g["offsets"], (i + 1,)))
        return pa.array([g["utf8"][lo:hi].tobytes().decode("utf-8")], type=arrow_type)
    offsets = read_array(g["offsets"])
    valid = read_array(g["dictionary_valid"])
    data = g["utf8"][()].tobytes()
    values = [data[int(offsets[i]):int(offsets[i+1])].decode("utf-8") if valid[i] else None for i in range(len(valid))]
    return pa.array([values[int(i)] for i in ids], type=arrow_type)


def _homogeneous(table, column):
    values = table[column].unique().to_pylist()
    if len(values) != 1 or values[0] is None:
        raise ValueError(f"Expected one {column} per source row group")
    return values[0]


def _pack_partition(tables, row_groups, destination, schema, feature_names):
    start = perf_counter()
    reps = [_homogeneous(t, "representation") for t in tables]
    if len(set(reps)) != len(reps):
        raise ValueError("Repeated representation within partition")
    # A node always means the same candidate slot, regardless of its representation.
    for t in tables[1:]:
        if column_digest(t["node_id"]) != column_digest(tables[0]["node_id"]):
            raise ValueError("Representation node IDs or order differ")
    hashes = [table_digest(t) for t in tables]
    matrix = np.concatenate([np.column_stack([t[c].fill_null(0).to_numpy() for c in feature_names]) for t in tables])
    valid = np.concatenate([np.column_stack([t[c].is_valid().to_numpy(zero_copy_only=False) for c in feature_names]) for t in tables])
    if any(schema.field(c).type != pa.float64() for c in feature_names):
        raise TypeError("Feature geometry pool requires source float64; no implicit casts")
    # Include validity in row equality: a missing value must not alias a true zero.
    record = np.concatenate([matrix.view(np.uint8).reshape(len(matrix), -1), valid.astype(np.uint8)], axis=1)
    unique, ids = unique_rows(record)
    float_width = len(feature_names) * 8
    geometry = unique[:, :float_width].copy().view(np.float64).reshape(-1, len(feature_names))
    geometry_valid = unique[:, float_width:].astype(bool)
    metadata = [c for c in schema.names if c not in feature_names]
    common = [c for c in metadata if all(column_digest(t[c]) == column_digest(tables[0][c]) for t in tables[1:])]
    partial = destination.with_suffix(".h5.partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    with h5py.File(partial, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["schema"] = base64.b64encode(schema.serialize().to_pybytes()).decode()
        f.attrs["feature_names"] = json.dumps(feature_names)
        f.attrs["representations"] = json.dumps(reps)
        f.attrs["common_columns"] = json.dumps(common)
        f.attrs["source_row_groups"] = json.dumps(row_groups)
        f.attrs["source_table_sha256"] = json.dumps(hashes)
        dataset(f, "geometry", geometry)
        write_array(f, "geometry_valid", geometry_valid)
        nodes = f.create_group("nodes")
        for c in common:
            _write_column(nodes, c, tables[0][c])
        offset = 0
        for rep, t in zip(reps, tables):
            g = f.create_group("representations/" + rep)
            dataset(g, "state_ids", ids[offset:offset+t.num_rows])
            for c in metadata:
                if c not in common:
                    _write_column(g, c, t[c])
            offset += t.num_rows
        dataset_count = assert_no_filters(f)
    partial.replace(destination)
    with FeaturePackage(destination) as package:
        for rep, expected in zip(reps, hashes):
            if table_digest(package.table(rep)) != expected:
                raise AssertionError(f"Lossless feature restoration failed: {destination}/{rep}")
    return {"package": destination.name, "rows": len(matrix), "nodes": tables[0].num_rows,
            "unique_geometry_states": len(geometry), "common_metadata_columns": len(common),
            "source_row_groups": row_groups, "package_bytes": destination.stat().st_size,
            "package_sha256": digest(destination), "verified_table_sha256": hashes,
            "unfiltered_datasets": dataset_count, "build_verify_seconds": perf_counter()-start}


def pack_features(source, destination, limit_partitions=None):
    """Stream a homogeneous-row-group feature table into independent batch files."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    reader = pq.ParquetFile(source)
    names = reader.schema_arrow.names
    feature_names = [c for c in names if "::" in c]
    if not feature_names:
        raise ValueError("No geometry features")
    groups = {}
    for i in range(reader.num_row_groups):
        meta = reader.read_row_group(i, columns=["campaign", "arm", "batch"])
        key = tuple(_homogeneous(meta, c) for c in ["campaign", "arm", "batch"])
        groups.setdefault(key, []).append(i)
    results = []
    for key, indices in list(groups.items())[:limit_partitions]:
        campaign, arm, batch = key
        for token in [campaign, arm]:
            if not token or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in token):
                raise ValueError("Unsafe partition label")
        path = destination / f"{campaign}__{arm}__batch_{batch:03d}.h5"
        tables = [reader.read_row_group(i) for i in indices]
        result = _pack_partition(tables, indices, path, reader.schema_arrow, feature_names)
        result.update(campaign=campaign, arm=arm, batch=batch)
        results.append(result)
        print(f"Feature partition {len(results)}/{min(len(groups),limit_partitions or len(groups))}: {arm}/{batch:03d}; {result['unique_geometry_states']}/{result['rows']} unique states", flush=True)
    manifest = {"format": FORMAT, "source": str(source.resolve()), "source_sha256": digest(source),
                "source_bytes": source.stat().st_size, "source_rows": reader.metadata.num_rows,
                "complete": len(results) == len(groups), "feature_columns": len(feature_names),
                "rows": sum(r["rows"] for r in results), "packages": results,
                "package_bytes": sum(r["package_bytes"] for r in results),
                "compression": "none; all datasets contiguous and unfiltered",
                "losslessness": "row order, schema metadata, Arrow types, null masks and all non-null numeric bits/string bytes"}
    write_json(destination / "manifest.json", manifest)
    return manifest


class FeaturePackage:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        if self.file.attrs.get("format") != FORMAT:
            self.file.close()
            raise ValueError("Unknown feature package")
        self.schema = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(self.file.attrs["schema"])))
        self.features = json.loads(self.file.attrs["feature_names"])
        self.common = set(json.loads(self.file.attrs["common_columns"]))
        self.representations = json.loads(self.file.attrs["representations"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def geometry(self, representation, row=None):
        """Direct NumPy API for analysis: exact float64 values plus validity mask.

        Null cells have a zero placeholder and a false validity bit. Consumers
        must use that mask, not treat placeholders as observations. This avoids
        rebuilding 255 Arrow columns when only geometry is needed.
        """
        g = self.file["representations"][representation]
        if row is None:
            ids = g["state_ids"][()]
            return self.file["geometry"][()][ids], read_array(self.file["geometry_valid"])[ids]
        if not 0 <= row < len(g["state_ids"]):
            raise IndexError(row)
        state = int(g["state_ids"][row])
        return self.file["geometry"][state], read_array(self.file["geometry_valid"], (state,))

    def table(self, representation, columns=None, row=None):
        columns = self.schema.names if columns is None else list(columns)
        g = self.file["representations"][representation]
        ids = g["state_ids"][()] if row is None else np.array([g["state_ids"][row]])
        arrays = []
        want_features = [c for c in columns if c in self.features]
        if want_features:
            # A node slice touches one contiguous geometry record. Full scans read
            # the state pool once rather than issuing one HDF5 lookup per feature.
            data, valid = self.geometry(representation, row)
            data = np.ascontiguousarray(np.atleast_2d(data).T)
            valid = np.ascontiguousarray(np.atleast_2d(valid).T)
        for c in columns:
            field = self.schema.field(c)
            if c in self.features:
                j = self.features.index(c)
                arrays.append(pa.array(data[j], type=field.type, mask=~valid[j], from_pandas=False))
            else:
                arrays.append(_read_column(self.file["nodes"][c] if c in self.common else g[c], field.type, row))
        schema = pa.schema([self.schema.field(c) for c in columns], metadata=self.schema.metadata)
        return pa.Table.from_arrays(arrays, schema=schema)

    def node(self, representation, step, slot, columns=None):
        steps = read_array(self.file["nodes"]["step"]["values"])
        slots = read_array(self.file["nodes"]["slot"]["values"])
        rows = np.flatnonzero((steps == step) & (slots == slot))
        if len(rows) != 1:
            raise KeyError((step, slot))
        return self.table(representation, columns, int(rows[0]))


def restore_features(directory, destination):
    """Rebuild an uncompressed Parquet with original row groups and Arrow schema."""
    directory, destination = Path(directory), Path(destination)
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Cannot restore a full source from a partial trial")
    if destination.exists() or destination.with_suffix(destination.suffix+".partial").exists():
        raise FileExistsError(destination)
    entries = []
    for item in manifest["packages"]:
        path = directory / item["package"]
        if digest(path) != item["package_sha256"]:
            raise AssertionError("Package file hash mismatch: " + str(path))
        with FeaturePackage(path) as package:
            for i, rep, expected in zip(item["source_row_groups"], package.representations, item["verified_table_sha256"]):
                entries.append((i, path, rep, expected))
    if [i for i, *_ in sorted(entries)] != list(range(len(entries))):
        raise ValueError("Source row groups are incomplete")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+".partial")
    writer = None
    try:
        for _, path, rep, expected in sorted(entries):
            with FeaturePackage(path) as package:
                table = package.table(rep)
                if table_digest(table) != expected:
                    raise AssertionError("Restored table digest mismatch")
                if writer is None:
                    writer = pq.ParquetWriter(partial, table.schema, compression=None, use_dictionary=False)
                writer.write_table(table, row_group_size=table.num_rows)
    finally:
        if writer is not None:
            writer.close()
    partial.replace(destination)
    return destination


def iter_feature_tables(directory, representations=None, batches=None, arms=None, columns=None):
    """Stream selected partitions in original row-group order, without expansion on disk."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    entries = []
    for item in manifest["packages"]:
        if batches is not None and item["batch"] not in batches:
            continue
        if arms is not None and item["arm"] not in arms:
            continue
        with FeaturePackage(directory / item["package"]) as package:
            for index, rep in zip(item["source_row_groups"], package.representations):
                if representations is None or rep in representations:
                    entries.append((index, directory / item["package"], rep))
    for _, path, rep in sorted(entries):
        with FeaturePackage(path) as package:
            yield package.table(rep, columns=columns)


def verify_features(directory, source=None):
    """Verify every package hash, every source row group and the absence of filters."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    original = pq.ParquetFile(source) if source is not None else None
    if source is not None and digest(source) != manifest["source_sha256"]:
        raise AssertionError("Source Parquet hash changed")
    rows = groups = datasets = 0
    for item in manifest["packages"]:
        path = directory / item["package"]
        if digest(path) != item["package_sha256"]:
            raise AssertionError("HDF5 package hash mismatch: " + str(path))
        with FeaturePackage(path) as package:
            datasets += assert_no_filters(package.file)
            for index, rep, expected in zip(item["source_row_groups"], package.representations, item["verified_table_sha256"]):
                table = package.table(rep)
                if table_digest(table) != expected:
                    raise AssertionError("Feature content hash mismatch")
                if original is not None and table_digest(original.read_row_group(index)) != expected:
                    raise AssertionError("Source row-group mismatch")
                groups += 1
                rows += table.num_rows
    return {"verified_rows": rows, "verified_row_groups": groups, "unfiltered_datasets": datasets,
            "package_count": len(manifest["packages"]), "compared_with_original": source is not None}
