"""Measure one final slot's actual ancestral path and a whole learning archive.

The one-path view cannot replace failed-branch evidence for enrichment analysis.
No molecular coordinates/probabilities are quantized. Source data are untouched.
"""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np

from evomolsteer.io import digest, write_json
from evomolsteer.storage.trajectory import pack_arrays, TrajectoryPackage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float, default=0.5)
    p.add_argument('--output', required=True)
    p.add_argument('--window-package', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    source = Path(a.source).resolve()
    if out == source.parent or out.is_relative_to(source.parent):
        raise ValueError('Keep measurement outside the immutable source')
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    steps_count, n = arrays['selected_indices'].shape
    if not 0 <= a.slot < n:
        raise ValueError('Invalid final slot')
    slots = np.zeros(steps_count+1, dtype=np.int64)
    slots[-1] = a.slot
    for step in range(steps_count-1, -1, -1):
        slots[step] = arrays['selected_indices'][step, slots[step+1]]
    times = np.round(arrays['score_time'][:, 0].astype(float), 6)
    steps = np.flatnonzero((times >= a.start) & (times <= a.end) & arrays['resampled'].astype(bool))
    if not len(steps) or np.any(np.diff(steps) != 1):
        raise ValueError('Require a nonempty contiguous selection window')
    view = {}
    for key, value in arrays.items():
        particle_axis = value.ndim > 1 and key != 'score_time'
        view[key] = value[steps, slots[steps]][:, None] if particle_axis else value[steps]
    # Original population slot IDs are explicit provenance. In the extracted
    # one-particle chain the local parent/child pointer is necessarily zero.
    view['source_candidate_slot'] = slots[steps, None]
    view['source_child_slot'] = slots[steps+1, None]
    view['selected_indices'] = np.zeros((len(steps), 1), dtype=arrays['selected_indices'].dtype)
    out.mkdir(parents=True)
    npz = out / 'ancestral_path.npz'
    np.savez_compressed(npz, **view)
    h5 = out / 'ancestral_path.h5'
    packed = pack_arrays(view, h5, source_sha256=digest(source))
    wrapped = out / 'ancestral_path.h5.gz'
    wrapped.write_bytes(gzip.compress(h5.read_bytes(), compresslevel=6, mtime=0))
    if gzip.decompress(wrapped.read_bytes()) != h5.read_bytes():
        raise AssertionError('Envelope byte roundtrip differs')
    with TrajectoryPackage(wrapped) as package:
        package.verify(view)
    # A lossless transport/archive wrapper, separate from working HDF5 storage.
    root = Path(a.window_package).resolve()
    members = [p for p in sorted(root.rglob('*')) if p.is_file() and 'npz_baseline' not in p.relative_to(root).parts]
    archive = out / 'selection_learning.tar.gz'
    expected = {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in members}
    with archive.open('xb') as handle:
        with gzip.GzipFile(fileobj=handle, mode='wb', compresslevel=6, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for src in members:
                    info = tarfile.TarInfo(src.relative_to(root).as_posix())
                    info.size = src.stat().st_size
                    info.mtime = 0
                    with src.open('rb') as inp:
                        tar.addfile(info, inp)
    checked = 0
    with tarfile.open(archive, 'r:gz') as tar:
        names = [m.name for m in tar.getmembers()]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise AssertionError('Archive member list differs')
        for m in tar.getmembers():
            h = hashlib.sha256()
            with tar.extractfile(m) as handle:
                for chunk in iter(lambda: handle.read(1048576), b''):
                    h.update(chunk)
            if h.hexdigest() != expected[m.name]['sha256'] or m.size != expected[m.name]['bytes']:
                raise AssertionError('Archive payload bytes differ')
            checked += 1
    write_json(a.report, {'source': str(source), 'source_sha256': digest(source),
                          'final_slot': a.slot, 'selection_steps': steps.tolist(),
                          'original_candidate_slots': slots[steps].tolist(),
                          'coordinates_atoms': int(arrays['mask'][0, 0].sum()),
                          'expanded_one_path_bytes': sum(x.nbytes for x in view.values()),
                          'one_path_npz_bytes': npz.stat().st_size, 'one_path_h5_bytes': h5.stat().st_size,
                          'one_path_wrapped_h5_bytes': wrapped.stat().st_size,
                          'one_path_current_xyz_bytes': view['current_coords'].nbytes,
                          'one_path_scope': 'one final survivor ancestry; excludes failed siblings; pointers remapped, original slot IDs retained',
                          'working_window_package_bytes': sum(x['bytes'] for x in expected.values()),
                          'window_archive_bytes': archive.stat().st_size, 'window_archive_sha256': digest(archive),
                          'archive_files_verified': checked, 'retained_files': expected,
                          'source_code_sha256': digest(__file__)})
    print(json.dumps({'one_path_npz_bytes': npz.stat().st_size, 'one_path_h5_bytes': h5.stat().st_size,
                      'one_path_wrapped_h5_bytes': wrapped.stat().st_size,
                      'window_archive_bytes': archive.stat().st_size, 'archive_files_verified': checked}, indent=2))


if __name__ == '__main__':
    main()
