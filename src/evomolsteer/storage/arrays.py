"""Byte-exact array normalization, with an explicit optional lossless codec.

Small integer indices, constants, identical-row dictionaries, exact zero sparse
rows and symmetric matrix storage are representations, not entropy codecs.
Every optimized representation is chosen by its measured payload size.
Default v1 files have no filters. A v2 trajectory may opt into gzip + byte
shuffle; no floating point quantization or scale-offset is allowed.
"""
import hashlib
import json

import h5py
import numpy as np


def byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array_digest(a):
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(json.dumps([a.dtype.str, list(a.shape)], separators=(",", ":")).encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def index_dtype(maximum, minimum=0):
    kinds = (np.uint8, np.uint16, np.uint32, np.uint64) if minimum >= 0 else (np.int8, np.int16, np.int32, np.int64)
    for kind in kinds:
        limits = np.iinfo(kind)
        if minimum >= limits.min and maximum <= limits.max:
            return np.dtype(kind)
    raise OverflowError("Index exceeds supported integer range")


def dataset(group, name, data):
    a = np.asarray(data)
    if group.file.attrs.get('storage_codec') == 'gzip_shuffle' and a.ndim and a.size and a.nbytes >= 2048:
        return group.create_dataset(name, data=a, chunks=True, compression='gzip',
                                    compression_opts=6, shuffle=True, fletcher32=False)
    # Explicitly contiguous, with no codec, shuffle, scale-offset or checksum filter.
    return group.create_dataset(name, data=np.asarray(data), chunks=None,
                                compression=None, shuffle=False, fletcher32=False)


def unique_rows(a):
    """Stable first-occurrence IDs; equality includes NaN payloads and signed zero."""
    a = np.ascontiguousarray(a)
    records = a.reshape(len(a), -1).view(np.dtype((np.void, a[0].nbytes))).ravel()
    _, first, inverse = np.unique(records, return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=index_dtype(max(0, len(order) - 1)))
    remap[order] = np.arange(len(order), dtype=remap.dtype)
    return a[first[order]], remap[inverse]


def _best_payload(a):
    """Dense versus exact positive-zero sparse rows (direct row offsets)."""
    a = np.ascontiguousarray(a)
    result = {"kind": "dense", "values": a, "bytes": a.nbytes}
    if a.ndim != 2 or not a.size:
        return result
    bits = a.view(np.dtype((np.void, a.dtype.itemsize))).reshape(a.shape)
    nz = bits != np.zeros((), dtype=a.dtype).view(bits.dtype)
    row, col = np.nonzero(nz)
    col = col.astype(index_dtype(max(0, a.shape[1] - 1)))
    ptr = np.r_[0, np.cumsum(nz.sum(axis=1))].astype(index_dtype(len(col)))
    values = a[row, col.astype(np.intp)]
    size = col.nbytes + ptr.nbytes + values.nbytes + 1024  # extra HDF5 objects
    if size < result["bytes"]:
        result = {"kind": "sparse_zero", "values": values, "columns": col,
                  "offsets": ptr, "bytes": size, "width": a.shape[1]}
    return result


def write_array(parent, name, source, row_dims=(1, 2, 3)):
    """Store a numeric array; record original dtype/shape and exact content hash."""
    a = np.asarray(source)
    if a.dtype.kind not in "biuf" or a.dtype.hasobject:
        raise TypeError(f"Unsupported numeric dtype: {a.dtype}")
    group = parent.create_group(name)
    group.attrs["shape"] = json.dumps(list(a.shape))
    group.attrs["dtype"] = a.dtype.str
    group.attrs["sha256"] = array_digest(a)
    if a.size and byte_equal(a, np.broadcast_to(a.flat[0], a.shape).astype(a.dtype)):
        group.attrs["kind"] = "constant"
        dataset(group, "values", np.array(a.flat[0], dtype=a.dtype))
        return group
    work = a
    if a.dtype.kind in "iu" and a.size:
        dtype = index_dtype(int(a.max()), int(a.min()))
        if dtype.itemsize < a.dtype.itemsize:
            work = a.astype(dtype)
    best = {"kind": "direct", "payload": _best_payload(work), "bytes": work.nbytes}
    best["bytes"] = best["payload"]["bytes"]
    for n in row_dims:
        if not 0 < n < work.ndim or not work.size:
            continue
        matrix = work.reshape(int(np.prod(work.shape[:n])), -1)
        pool, ids = unique_rows(matrix)
        payload = _best_payload(pool)
        cost = payload["bytes"] + ids.nbytes + 1024
        if cost < best["bytes"]:
            best = {"kind": "dictionary", "payload": payload, "bytes": cost,
                    "ids": ids.reshape(work.shape[:n]), "tail": list(work.shape[n:])}
    group.attrs["kind"] = best["kind"]
    group.attrs["work_shape"] = json.dumps(list(work.shape))
    payload = best["payload"]
    group.attrs["payload_kind"] = payload["kind"]
    dataset(group, "values", payload["values"])
    if payload["kind"] == "sparse_zero":
        group.attrs["width"] = payload["width"]
        dataset(group, "columns", payload["columns"])
        dataset(group, "offsets", payload["offsets"])
    if best["kind"] == "dictionary":
        group.attrs["tail"] = json.dumps(best["tail"])
        dataset(group, "ids", best["ids"])
    return group


def _payload(group, rows=None):
    values = group["values"]
    if group.attrs["payload_kind"] == "dense":
        if rows is None:
            return values[()]
        ids = np.asarray(rows)
        if ids.ndim == 0:
            return values[int(ids)]
        unique, inverse = np.unique(ids, return_inverse=True)
        if unique.size > max(16, len(values) // 4):
            # HDF5 point selection is costly for thousands of dictionary IDs.
            # A contiguous pool read and NumPy gather is the bulk access path.
            return values[()][ids]
        return values[unique.astype(np.intp)][inverse].reshape(*ids.shape, *values.shape[1:])
    offsets = group["offsets"]
    shape = (len(offsets) - 1, int(group.attrs["width"]))
    if rows is None:
        ptr = offsets[()]
        result = np.zeros(shape, dtype=values.dtype)
        row = np.repeat(np.arange(shape[0]), np.diff(ptr.astype(np.int64)))
        result[row, group["columns"][()]] = values[()]
        return result
    ids = np.asarray(rows)
    if ids.size > 16:
        return _payload(group)[ids]
    out = np.zeros((ids.size, shape[1]), dtype=values.dtype)
    for i, row in enumerate(ids.ravel()):
        lo, hi = map(int, offsets[int(row):int(row) + 2])
        out[i, group["columns"][lo:hi]] = values[lo:hi]
    return out.reshape(*ids.shape, shape[1])


def read_array(group, prefix=()):
    """Read whole array or integer-indexed prefix without decoding other rows."""
    prefix = tuple(prefix)
    shape = tuple(json.loads(group.attrs["shape"]))
    dtype = np.dtype(group.attrs["dtype"])
    if len(prefix) > len(shape):
        raise IndexError("Too many prefix indices")
    for i, v in enumerate(prefix):
        if not 0 <= v < shape[i]:
            raise IndexError(prefix)
    kind = group.attrs["kind"]
    if kind == "constant":
        return np.full(shape[len(prefix):], group["values"][()], dtype=dtype)
    if kind == "dictionary":
        ids_ds = group["ids"]
        used = min(len(prefix), ids_ds.ndim)
        ids = ids_ds[prefix[:used]] if used else ids_ds[()]
        tail = tuple(json.loads(group.attrs["tail"]))
        out = _payload(group, ids).reshape(*np.shape(ids), *tail).astype(dtype, copy=False)
        return out[prefix[used:]] if len(prefix) > used else out
    if group.attrs["payload_kind"] == "dense":
        out = group["values"][prefix] if prefix else group["values"][()]
        return np.asarray(out, dtype=dtype)
    if not prefix:
        return _payload(group).reshape(shape).astype(dtype, copy=False)
    out = _payload(group, prefix[0]).astype(dtype, copy=False)
    return out[prefix[1:]]


def assert_no_filters(file):
    count = 0
    def inspect(_, obj):
        nonlocal count
        if isinstance(obj, h5py.Dataset):
            if obj.id.get_create_plist().get_nfilters() or obj.chunks is not None:
                raise AssertionError(f"Non-contiguous or filtered dataset: {obj.name}")
            count += 1
    file.visititems(inspect)
    return count


def assert_lossless_filters(file):
    """Accept only our explicit gzip + byte shuffle profile; never scale-offset."""
    if file.attrs.get('storage_codec') != 'gzip_shuffle':
        raise ValueError('Unknown lossless storage codec')
    counts = {'contiguous_datasets':0,'gzip_shuffle_datasets':0}
    def inspect(_, obj):
        if not isinstance(obj,h5py.Dataset):
            return
        plist = obj.id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        if filters:
            if filters != [h5py.h5z.FILTER_SHUFFLE,h5py.h5z.FILTER_DEFLATE] or obj.compression_opts != 6:
                raise AssertionError('Unexpected storage filter: '+obj.name)
            counts['gzip_shuffle_datasets'] += 1
        else:
            if obj.chunks is not None:
                raise AssertionError('Unexpected unfiltered chunking: '+obj.name)
            counts['contiguous_datasets'] += 1
    file.visititems(inspect)
    return counts
