"""Read-only source audit and measured selection-window storage trial.

Uses the existing byte-exact trajectory codec, never deletes source data.
The trial is a learning view, not a restart archive or a drop-in input bundle.
Run from the repository with its Python environment; arguments are datasets.
"""
import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np

from evomolsteer.io import digest, write_json
from evomolsteer.storage.arrays import byte_equal
from evomolsteer.storage.trajectory import pack_arrays, TrajectoryPackage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--normalized', required=True)
    p.add_argument('--remote-inventory', required=True)
    p.add_argument('--remote-hashes', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--report', required=True)
    p.add_argument('--campaign', default='main1000_w050')
    p.add_argument('--arm', default='single')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--end', type=float, default=0.5)
    a = p.parse_args()
    root, normalized, out = map(lambda x: Path(x).resolve(), (a.input, a.normalized, a.output))
    if out.exists() or out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('Use a new output outside the immutable input')
    if not np.isfinite([a.start, a.end]).all() or a.start > a.end:
        raise ValueError('Invalid score-time bounds')
    inventory = json.loads(Path(a.remote_inventory).read_text())
    hashes = {r['path']: r for r in json.loads(Path(a.remote_hashes).read_text())}
    prefix = f'results/{a.campaign}/{a.arm}/'
    remote_files = [r for r in inventory['files'] if r['path'].startswith(prefix)]
    paths = sorted((root / prefix).glob('batch_*/trajectory.npz'))
    if {p.relative_to(root).as_posix() for p in paths} != set(hashes):
        raise ValueError('Local source batch set differs from the remote audit')
    # Verify every byte, including ancillary snapshots, against the original
    # SHA256 inventory downloaded earlier, then freshly check all trajectories.
    original_checks = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        h, name = line.split(None, 1)
        original_checks[name.strip().removeprefix('./')] = h
    verified_ancillary = 0
    for r in remote_files:
        local = root / r['path']
        if local.stat().st_size != r['bytes']:
            raise ValueError('Remote/local file size differs: ' + r['path'])
        if r['path'] in original_checks and digest(local) != original_checks[r['path']]:
            raise ValueError('Original ancillary checksum differs: ' + r['path'])
        verified_ancillary += 1
    out.mkdir(parents=True)
    records, copied = [], []
    final_names = {'final_records.json', 'molecules_all_built.sdf', 'molecules_raw_decodable.sdf',
                   'quality_records.json', 'posebusters_checks.csv', 'QUALITY_COMPLETE.json',
                   'events.json', 'COMPLETE.json'}
    lineage_fields = ['selected_indices', 'offspring_count', 'root_slot', 'parent_slot',
                      'resampled', 'score_time', 'state_time', 'step_size']
    for source in paths:
        begin = perf_counter()
        relative = source.relative_to(root).as_posix()
        if source.stat().st_size != hashes[relative]['bytes'] or digest(source) != hashes[relative]['sha256']:
            raise ValueError('Fresh remote/local trajectory checksum differs')
        with np.load(source, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        with TrajectoryPackage((normalized / relative).with_suffix('.h5')) as package:
            package.verify(arrays)
            normalized_bytes = package.path.stat().st_size
        times = np.round(arrays['score_time'][:, 0].astype(float), 6)
        steps = np.flatnonzero((times >= a.start) & (times <= a.end) & arrays['resampled'].astype(bool))
        if not len(steps) or np.any(np.diff(steps) != 1):
            raise ValueError('This measured view requires contiguous actual selection events')
        window = {k: v[steps] for k, v in arrays.items()}
        batch = out / source.parent.relative_to(root)
        window_result = pack_arrays(window, batch / 'selection_window.h5', source_sha256=hashes[relative]['sha256'])
        # Exact NPZ baseline on the same retained window, not half of a file.
        baseline = out / 'npz_baseline' / source.parent.name / 'trajectory.npz'
        baseline.parent.mkdir(parents=True)
        np.savez_compressed(baseline, **window)
        with np.load(baseline, allow_pickle=False) as z:
            if z.files != list(window) or any(not byte_equal(z[k], window[k]) for k in z.files):
                raise AssertionError('Window NPZ roundtrip differs')
        terminal = pack_arrays({k: v[-1:] for k, v in arrays.items()}, batch / 'terminal_row.h5',
                               source_sha256=hashes[relative]['sha256'])
        genealogy = pack_arrays({k: arrays[k] for k in lineage_fields}, batch / 'full_lineage.h5',
                                source_sha256=hashes[relative]['sha256'])
        # Cross-check boundary children and terminal ancestors. Failed candidates
        # remain in the window: no survivor-only filtering has been applied.
        with TrajectoryPackage(batch / 'selection_window.h5') as view:
            after = arrays['proposal_coords'][steps[-1], arrays['selected_indices'][steps[-1]]]
            restored_after = view.read('proposal_coords', len(steps)-1)[view.read('selected_indices', len(steps)-1)]
            if not byte_equal(after, restored_after):
                raise AssertionError('Post-selection boundary coordinates differ')
        ancestors = np.arange(arrays['selected_indices'].shape[1])
        for step in range(len(times)-1, steps[-1], -1):
            ancestors = arrays['selected_indices'][step, ancestors]
        with TrajectoryPackage(batch / 'full_lineage.h5') as view:
            restored = np.arange(len(ancestors))
            for step in range(len(times)-1, steps[-1], -1):
                restored = view.read('selected_indices', step)[restored]
            if not np.array_equal(ancestors, restored):
                raise AssertionError('Terminal lineage mapping differs')
        for name in sorted(final_names):
            src = source.parent / name
            if src.is_file():
                dst = batch / name
                shutil.copyfile(src, dst)
                if digest(src) != digest(dst):
                    raise AssertionError('Ancillary copy differs')
                copied.append({'path': dst.relative_to(out).as_posix(), 'bytes': dst.stat().st_size,
                               'sha256': digest(dst)})
        record = {'batch': source.parent.name, 'source_sha256': hashes[relative]['sha256'],
                  'source_npz_bytes': source.stat().st_size, 'full_h5_bytes': normalized_bytes,
                  'window_npz_bytes': baseline.stat().st_size, 'window_h5': window_result,
                  'terminal_h5': terminal, 'genealogy_h5': genealogy,
                  'selection_steps': steps.tolist(), 'first_score_time': float(times[steps[0]]),
                  'last_score_time': float(times[steps[-1]]),
                  'last_selected_proposal_time': float(arrays['state_time'][steps[-1]]),
                  'particles': arrays['selected_indices'].shape[1],
                  'atom_counts': np.unique(arrays['mask'].sum(-1)).tolist(),
                  'seconds': perf_counter()-begin}
        records.append(record)
        print(json.dumps({k: record[k] for k in ['batch', 'source_npz_bytes', 'full_h5_bytes', 'window_npz_bytes', 'seconds']}
                         | {'window_h5_bytes': window_result['package_bytes']}), flush=True)
    campaign = root / 'results' / a.campaign
    shared = list((root / 'inputs').glob('*')) + [campaign / 'config.json']
    shared += [campaign / f'frame_{p.parent.name}.json' for p in paths]
    for src in sorted(set(shared)):
        if src.is_file():
            dst = out / src.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if digest(src) != digest(dst):
                raise AssertionError('Shared geometry/provenance copy differs')
            copied.append({'path': dst.relative_to(out).as_posix(), 'bytes': dst.stat().st_size, 'sha256': digest(dst)})
    manifest = {'format': 'evomolsteer.storage_trial.selection_window.v1', 'complete': True,
                'scope': 'all actual resampling candidates; score_time bounds inclusive',
                'input_root': str(root), 'window': [a.start, a.end], 'records': records, 'copied_files': copied,
                'purpose': 'Selection learning plus exact terminal row and all-step genealogy; not a restart archive',
                'not_retained': ['out-of-window intermediate geometry and predictions, except terminal row',
                                 'PyTorch restart/endpoint snapshots and generator RNG state'],
                'verified': 'Every retained array shape/dtype/bit; original full HDF5 versus NPZ; copied files SHA256'}
    write_json(out / 'manifest.json', manifest)
    totals = {
        'remote_original_directory_bytes': inventory['logical_bytes'],
        'remote_single_directory_bytes': sum(r['bytes'] for r in remote_files),
        'remote_single_allocated_bytes': sum(r['allocated_bytes'] for r in remote_files),
        'full_npz_trajectory_bytes': sum(r['source_npz_bytes'] for r in records),
        'full_h5_trajectory_bytes': sum(r['full_h5_bytes'] for r in records),
        'window_npz_trajectory_bytes': sum(r['window_npz_bytes'] for r in records),
        'window_h5_trajectory_bytes': sum(r['window_h5']['package_bytes'] for r in records),
        'terminal_h5_bytes': sum(r['terminal_h5']['package_bytes'] for r in records),
        'full_genealogy_h5_bytes': sum(r['genealogy_h5']['package_bytes'] for r in records),
        'copied_terminal_provenance_geometry_bytes': sum(r['bytes'] for r in copied),
        'window_package_manifest_bytes': (out/'manifest.json').stat().st_size,
        'batches': len(records), 'particle_slots': sum(r['particles'] for r in records),
        'candidate_events': sum(len(r['selection_steps'])*r['particles'] for r in records),
        'verified_full_trajectory_arrays': sum(r['window_h5']['verified_arrays'] for r in records),
        'remote_local_ancillary_files_checked': verified_ancillary,
    }
    totals['full_directory_h5_replacement_bytes'] = (totals['remote_single_directory_bytes']
        - totals['full_npz_trajectory_bytes'] + totals['full_h5_trajectory_bytes'])
    totals['window_learning_package_bytes'] = sum(totals[k] for k in [
        'window_h5_trajectory_bytes', 'terminal_h5_bytes', 'full_genealogy_h5_bytes',
        'copied_terminal_provenance_geometry_bytes', 'window_package_manifest_bytes'])
    for name, before, after in [
        ('full_trajectory_reduction', 'full_npz_trajectory_bytes', 'full_h5_trajectory_bytes'),
        ('full_directory_reduction', 'remote_single_directory_bytes', 'full_directory_h5_replacement_bytes'),
        ('same_window_codec_reduction', 'window_npz_trajectory_bytes', 'window_h5_trajectory_bytes'),
        ('learning_package_reduction', 'remote_single_directory_bytes', 'window_learning_package_bytes')]:
        totals[name] = 1 - totals[after]/totals[before]
    write_json(a.report, {'totals': totals, 'manifest': manifest, 'remote_inventory': a.remote_inventory,
                          'remote_hashes': a.remote_hashes, 'source_code_sha256': digest(__file__)})
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
