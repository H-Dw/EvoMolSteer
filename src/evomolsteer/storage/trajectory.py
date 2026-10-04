"""Batch packages with logical node addressing and exact ancestor references."""
import json
import os
import gzip
import io
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from ..io import digest
from .arrays import array_digest, assert_no_filters, assert_lossless_filters, byte_equal, dataset, read_array, write_array

FORMAT = "evomolsteer.trajectory.v1"
FILTERED_FORMAT = "evomolsteer.trajectory.v2"


def _write_field(parent, key, a):
    if key.endswith("_bonds") and a.ndim == 4 and byte_equal(a, a.swapaxes(-1, -2)):
        group = parent.create_group(key)
        group.attrs["kind"] = "symmetric"
        group.attrs["shape"] = json.dumps(list(a.shape))
        group.attrs["dtype"] = a.dtype.str
        group.attrs["sha256"] = array_digest(a)
        x, y = np.triu_indices(a.shape[-1], 1)
        write_array(group, "upper", a[..., x, y], row_dims=(2,))
        write_array(group, "diagonal", np.diagonal(a, axis1=-2, axis2=-1), row_dims=(2,))
    else:
        write_array(parent, key, a)


def _read_field(group, prefix=()):
    if group.attrs["kind"] != "symmetric":
        return read_array(group, prefix)
    shape = tuple(json.loads(group.attrs["shape"]))
    out = np.zeros(shape[len(prefix):], dtype=group.attrs["dtype"])
    n = shape[-1]
    x, y = np.triu_indices(n, 1)
    upper = read_array(group["upper"], prefix)
    out[..., x, y] = upper
    out[..., y, x] = upper
    out[..., np.arange(n), np.arange(n)] = read_array(group["diagonal"], prefix)
    return out


def pack_trajectory(source, destination, normalize=True, codec='none'):
    source = Path(source)
    source_sha = digest(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    result = pack_arrays(arrays,destination,normalize=normalize,codec=codec,source_sha256=source_sha)
    if digest(source)!=source_sha:
        raise ValueError('Source changed during conversion; source must be retained')
    result.update(source=str(source.resolve()),source_bytes=source.stat().st_size,source_sha256=source_sha)
    return result


def pack_arrays(arrays, destination, normalize=True, codec='gzip_shuffle', source_sha256=None):
    """Publish a package only after reopening and bit-checking every input array.

    The input has a leading time axis, including one-step inputs. No temporary
    NPZ is required for online generation. This function never deletes sources.
    """
    destination = Path(destination)
    if not arrays:
        raise ValueError('Empty trajectory')
    if any(not isinstance(k,str) or not k or '/' in k for k in arrays):
        raise ValueError('Invalid array field name')
    if codec not in ('none','gzip_shuffle'):
        raise ValueError('Unknown trajectory codec')
    if destination.exists() or destination.with_suffix(destination.suffix + ".partial").exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    partial = destination.with_suffix(destination.suffix + ".partial")
    aliases = []
    with h5py.File(partial, "x") as f:
        f.attrs["format"] = FORMAT if codec=='none' else FILTERED_FORMAT
        if codec!='none':
            f.attrs['storage_codec'] = codec
        if source_sha256 is not None:
            f.attrs["source_sha256"] = source_sha256
        f.attrs["keys"] = json.dumps(list(arrays))
        f.attrs["normalization"] = bool(normalize)
        group = f.create_group("fields")
        for key, a in arrays.items():
            proposal = key.replace("current_", "proposal_", 1)
            can_alias = (normalize and key.startswith("current_") and proposal in arrays and 'selected_indices' in arrays)
            if can_alias:
                indices = arrays["selected_indices"][:-1]
                selected = arrays[proposal][np.arange(len(a) - 1)[:, None], indices]
                can_alias = byte_equal(a[1:], selected)
            if can_alias:
                g = group.create_group(key)
                g.attrs["kind"] = "previous_proposal"
                g.attrs["source_field"] = proposal
                g.attrs["shape"] = json.dumps(list(a.shape))
                g.attrs["dtype"] = a.dtype.str
                g.attrs["sha256"] = array_digest(a)
                write_array(g, "initial", a[0])
                aliases.append(key)
            elif normalize:
                _write_field(group, key, a)
            else:
                g = group.create_group(key)
                g.attrs["kind"] = "direct"
                g.attrs["payload_kind"] = "dense"
                g.attrs["shape"] = json.dumps(list(a.shape))
                g.attrs["dtype"] = a.dtype.str
                g.attrs["sha256"] = array_digest(a)
                dataset(g, "values", a)
        filters = ({'contiguous_datasets':assert_no_filters(f),'gzip_shuffle_datasets':0}
                   if codec=='none' else assert_lossless_filters(f))
    with TrajectoryPackage(partial) as package:
        verified = package.verify(arrays)
    with partial.open('r+b') as handle:
        os.fsync(handle.fileno())
    if destination.exists():
        raise FileExistsError(destination)
    partial.replace(destination)
    return {"package": str(destination.resolve()),
            "expanded_array_bytes": sum(a.nbytes for a in arrays.values()),
            "package_bytes": destination.stat().st_size,
            "package_sha256": digest(destination), "verified_arrays": verified,
            "lineage_aliases": aliases, "unfiltered_datasets": filters['contiguous_datasets'],
            'codec':codec,'filter_audit':filters,
            "build_verify_seconds": perf_counter() - start}


class TrajectoryPackage:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as handle:wrapped=handle.read(2)==b'\x1f\x8b'
        self._buffer = None
        if wrapped:
            decoded=gzip.decompress(self.path.read_bytes())
            if hasattr(h5py.File,'in_memory'):
                self.file=h5py.File.in_memory(file_image=decoded)
            else:
                self._buffer=io.BytesIO(decoded)
                self.file=h5py.File(self._buffer,'r')
        else:self.file=h5py.File(path,'r')
        if self.file.attrs.get("format") not in (FORMAT,FILTERED_FORMAT):
            self.file.close()
            raise ValueError("Unknown trajectory package")
        self.keys = json.loads(self.file.attrs["keys"])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
        if self._buffer is not None:self._buffer.close()

    def read(self, key, step=None, slot=None):
        g = self.file["fields"][key]
        if slot is not None and step is None:
            raise ValueError("slot requires step")
        prefix = () if step is None else ((step,) if slot is None else (step, slot))
        shape = tuple(json.loads(g.attrs["shape"]))
        if len(prefix) > len(shape) or any(not 0 <= v < shape[i] for i, v in enumerate(prefix)):
            raise IndexError(prefix)
        if g.attrs["kind"] != "previous_proposal":
            return _read_field(g, prefix)
        source = str(g.attrs["source_field"])
        if step == 0:
            return read_array(g["initial"], () if slot is None else (slot,))
        if step is None:
            proposed = self.read(source)
            indices = self.read("selected_indices")[:-1]
            return np.concatenate([read_array(g["initial"])[None], proposed[np.arange(len(proposed)-1)[:, None], indices]])
        indices = self.read("selected_indices", step - 1)
        if slot is None:
            return self.read(source, step - 1)[indices]
        return self.read(source, step - 1, int(indices[slot]))

    def node(self, step, slot):
        """All scored fields of a candidate, including rejected candidates."""
        fields = {}
        for key in self.keys:
            shape = json.loads(self.file["fields"][key].attrs["shape"])
            # score_time's second axis is time channels, not particle slots.
            particle_axis = len(shape) >= 2 and key != "score_time"
            fields[key] = self.read(key, step, slot if particle_axis else None)
        fields["children_next_slots"] = np.flatnonzero(self.read("selected_indices", step) == slot)
        return fields

    def verify(self, reference=None):
        if reference is not None and self.keys!=list(reference):
            raise AssertionError('Array names/order differ from the source')
        if self.file.attrs['format']==FORMAT:
            assert_no_filters(self.file)
        else:
            assert_lossless_filters(self.file)
        for key in self.keys:
            a = self.read(key)
            if array_digest(a) != self.file["fields"][key].attrs["sha256"]:
                raise AssertionError("Array content mismatch: " + key)
            if reference is not None and not byte_equal(a, reference[key]):
                raise AssertionError("Source dtype/shape/bytes mismatch: " + key)
        return len(self.keys)

    def restore(self, destination):
        """Reconstruct an uncompressed NPZ, preserving every array bit, not ZIP bytes."""
        path = Path(destination)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez(handle, **{key: self.read(key) for key in self.keys})
