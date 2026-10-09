import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
loader = importlib.util.spec_from_file_location('completed_snapshot', ROOT / 'scripts/snapshot_completed_terminal.py')
snapshot = importlib.util.module_from_spec(loader)
loader.loader.exec_module(snapshot)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'running'
    run = root / 'results/control'
    (root / 'inputs').mkdir(parents=True)
    (root / 'inputs/protein.pdb').write_text('unchanged input')
    run.mkdir(parents=True)
    config = {'experiment': {'n': 4, 'batch': 2, 'steps': 100, 'seed': 42,
        'arms': 'unguided,gradient', 'batch_indices': [55,56]}}
    (run / 'config.json').write_text(json.dumps(config))
    (run / 'reward_program.json').write_text('{}')
    (run / 'coordinate_gradient_preflight.json').write_text('{"passed":true}')
    (run / 'frame_batch_055.json').write_text('{}')
    for arm, batch in [('unguided',55),('gradient',55),('unguided',56)]:
        folder = run / arm / f'batch_{batch:03d}'
        folder.mkdir(parents=True)
        (folder/'COMPLETE.json').write_text('{"n":2}')
        (folder/'final_records.json').write_text(json.dumps([{'slot':0},{'slot':1}]))
        (folder/'molecules_raw_decodable.sdf').write_text('molecule\n$$$$\n')
        (folder/'trajectory.h5').write_bytes(b'closed synthetic trajectory')
    arrays = {'current_coords':np.zeros((1,2,3,3)), 'current_atomics':np.zeros((1,2,3),dtype=np.uint8),
        'current_bonds':np.zeros((1,2,3,3),dtype=np.uint8), 'selected_indices':np.tile(np.arange(2),(100,1)),
        'resampled':np.zeros(100,dtype=np.uint8)}
    monkeypatch.setattr(snapshot, 'execution_arrays', lambda path: arrays)
    return root, arrays


def test_freezes_common_closed_batches_and_keeps_running_source_unchanged(source, tmp_path):
    root, _ = source
    run = root/'results/control'
    before = (run/'config.json').read_bytes()
    output = tmp_path/'snapshot.tar.gz'
    result = snapshot.snapshot(root, 'control', output)
    assert result['snapshot_n_per_arm'] == 2 and result['included_batches'] == [55]
    assert not result['source_campaign_complete']
    assert not (run/'COMPLETE.json').exists() and (run/'config.json').read_bytes() == before
    with tarfile.open(output) as archive:
        cfg = json.load(archive.extractfile('results/control/config.json'))
        scope = json.load(archive.extractfile('results/control/SNAPSHOT.json'))
        marker = json.load(archive.extractfile('results/control/COMPLETE.json'))
        assert cfg['experiment']['n'] == 2 and cfg['experiment']['batch_indices'] == [55]
        assert scope['original_expected_n_per_arm'] == 4 and scope['excluded_declared_batches'] == [56]
        assert not marker['source_campaign_complete'] and 'snapshot' in marker['completion_scope']
        assert not any('batch_056' in name for name in archive.getnames())
    with pytest.raises(ValueError):
        snapshot.snapshot(root, 'control', output)


@pytest.mark.parametrize('corruption', ['selection', 'resampled', 'slots'])
def test_incomplete_or_selected_payload_is_rejected(source, tmp_path, corruption):
    root, arrays = source
    if corruption == 'selection':
        arrays['selected_indices'][20,0] = 1
    elif corruption == 'resampled':
        arrays['resampled'][20] = 1
    else:
        (root/'results/control/gradient/batch_055/final_records.json').write_text('[{"slot":0}]')
    with pytest.raises(ValueError):
        snapshot.snapshot(root, 'control', tmp_path/'rejected.tar.gz')
    assert not (tmp_path/'rejected.tar.gz').exists()
    assert not (root/'results/control/COMPLETE.json').exists()


def test_no_common_finished_batch_or_inside_destination_is_rejected(source):
    root, _ = source
    with pytest.raises(ValueError):
        snapshot.snapshot(root, 'control', root/'snapshot.tar.gz')
    (root/'results/control/gradient/batch_055/COMPLETE.json').unlink()
    with pytest.raises(ValueError, match='No common'):
        snapshot.snapshot(root, 'control', root.parent/'outside.tar.gz')
