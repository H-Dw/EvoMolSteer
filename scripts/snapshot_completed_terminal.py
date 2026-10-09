"""Freeze only finished common batches from a running generation campaign.

Produces an explicitly partial terminal transport without writing to the source.
The derived COMPLETE marker certifies the snapshot, never the whole campaign.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from archive_generation import archive
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.storage.evaluation_view import execution_arrays, state_signature


def snapshot(dataset, campaign, output):
    root, output = Path(dataset).resolve(), Path(output).resolve()
    run = (root / 'results' / campaign).resolve()
    if run.parent != root / 'results' or output.is_relative_to(root) or output.exists():
        raise ValueError('One campaign and fresh external snapshot destination required')
    config = read_json(run / 'config.json')
    options = config['experiment']
    arms, declared = options['arms'].split(','), options['batch_indices']
    if set(arms) - {'gradient', 'unguided'}:
        raise ValueError('This terminal snapshot accepts no particle-selection arm')
    included = [batch for batch in declared
        if all((run / arm / f'batch_{batch:03d}/COMPLETE.json').is_file() for arm in arms)]
    if not included:
        raise ValueError('No common completed batch')
    batch_size, steps = options['batch'], options['steps']
    source_complete = (run / 'COMPLETE.json').is_file()
    source_hashes = {}
    def copy(source, destination):
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError('Aliased source payload')
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = digest(source)
        shutil.copyfile(source, destination)
        if before != digest(source) or before != digest(destination):
            raise ValueError('Source changed while freezing terminal payload')
        source_hashes[source.relative_to(root).as_posix()] = before
    with tempfile.TemporaryDirectory(prefix='evomolsteer-completed-snapshot-') as temporary:
        stage = Path(temporary) / 'dataset'
        target = stage / 'results' / campaign
        target.mkdir(parents=True)
        for path in (root / 'inputs').rglob('*'):
            if path.is_file():
                copy(path, stage / path.relative_to(root))
        for name in ['config.json', 'reward_program.json', 'coordinate_gradient_preflight.json']:
            copy(run / name, target / ('SOURCE_CONFIG.json' if name == 'config.json' else name))
        for path in (run / 'provenance').rglob('*'):
            if path.is_file():
                copy(path, target / path.relative_to(run))
        rows, entries = [], []
        for batch in included:
            frame = run / f'frame_batch_{batch:03d}.json'
            copy(frame, target / frame.name)
            for arm in arms:
                folder = run / arm / f'batch_{batch:03d}'
                destination = target / arm / folder.name
                complete = read_json(folder / 'COMPLETE.json')
                records = read_json(folder / 'final_records.json')
                if complete['n'] != batch_size or len(records) != batch_size or {r['slot'] for r in records} != set(range(batch_size)):
                    raise ValueError('Incomplete slot coverage in closed batch')
                for name in ['COMPLETE.json', 'final_records.json', 'molecules_raw_decodable.sdf',
                             'molecules_all_built.sdf', 'guidance_trace.jsonl', 'events.json', 'window_state.npz']:
                    path = folder / name
                    if path.exists():
                        copy(path, destination / name)
                trajectory = folder / 'trajectory.h5'
                arrays = execution_arrays(trajectory)
                if arrays['resampled'].any() or not np.array_equal(arrays['selected_indices'], np.tile(np.arange(batch_size), (steps, 1))):
                    raise ValueError('Incomplete continuation or selection detected')
                state = destination / 'execution_state.npz'
                np.savez_compressed(state, **arrays)
                with np.load(state, allow_pickle=False) as saved:
                    if any(saved[k].dtype != v.dtype or saved[k].tobytes() != v.tobytes() for k, v in arrays.items()):
                        raise ValueError('Snapshot state changed')
                trajectory_sha = digest(trajectory)
                source_hashes[trajectory.relative_to(root).as_posix()] = trajectory_sha
                entries.append({'batch_path': destination.relative_to(target).as_posix(),
                    'source_trajectory_sha256': trajectory_sha, 'source_bytes': trajectory.stat().st_size,
                    'snapshot_sha256': digest(state), 'initial_state_signature': state_signature(arrays), 'steps': steps})
                rows.extend(records)
        scope = {'schema_version': 'completed-batch-snapshot-1.0', 'snapshot_complete': True,
            'source_campaign_complete_at_capture': source_complete, 'original_expected_n_per_arm': options['n'],
            'snapshot_n_per_arm': len(included) * batch_size, 'included_batches': included,
            'excluded_declared_batches': [b for b in declared if b not in included],
            'source_dataset': str(root), 'source_campaign': campaign, 'captured_unix': time.time(),
            'source_config_sha256': source_hashes['results/' + campaign + '/config.json'],
            'source_files_sha256': source_hashes,
            'semantics': 'Immutable completed common batches; derived COMPLETE certifies this snapshot only; no source writes or generation calls'}
        config['experiment']['n'] = scope['snapshot_n_per_arm']
        config['experiment']['batch_indices'] = included
        config['completed_batch_snapshot'] = {key: value for key, value in scope.items() if key != 'source_files_sha256'}
        write_json(target / 'config.json', config)
        write_json(target / 'SNAPSHOT.json', scope)
        write_json(target / 'COMPLETE.json', {'status': 'complete', 'records': len(rows),
            'completion_scope': 'Completed-batch snapshot only', 'source_campaign_complete': source_complete})
        write_json(target / 'final_records.json', rows)
        write_json(target / 'EVALUATION_VIEW.json', {'schema_version': 'terminal-execution-view-1.0',
            'complete': True, 'campaign': campaign, 'batches': entries, 'source_deleted': False,
            'scope': scope, 'converter_source_sha256': digest(__file__),
            'usage': 'Terminal evaluation of fixed completed subset; does not certify full cohort completion'})
        result = archive(stage, campaign, output)
    result.update(transport='completed-batch-terminal-snapshot-1.0', snapshot_scope=scope)
    write_json(Path(str(output) + '.json'), result)
    return {'snapshot_n_per_arm': scope['snapshot_n_per_arm'], 'included_batches': included,
            'source_campaign_complete': source_complete, 'archive_sha256': result['archive_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.dataset, args.campaign, args.output)))
