import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('plip_probe', Path(__file__).parents[1] / 'scripts/benchmark_plip_batches.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_summary_does_not_hide_an_oom_behind_successful_repeats():
    rows = [{'teacher': 'top2', 'batch_size': 100, 'status': 'ok', 'seconds': 40,
             'peak_allocated_GiB': 10, 'peak_reserved_GiB': 12, 'peak_device_GiB': 13},
            {'teacher': 'top2', 'batch_size': 100, 'status': 'oom'}]
    summary = probe.median_cases(rows)[0]
    assert summary['passed'] is False
    assert summary['seconds_median'] == 40
    assert summary['repeats'] == 2


def test_batch_join_preserves_each_original_slot_and_rejects_changed_metadata():
    torch = pytest.importorskip('torch')
    first = {'coords': torch.arange(60).reshape(20, 3), 'meta': ['a'] * 20, 'static': 'same'}
    second = {'coords': first['coords'] + 60, 'meta': ['b'] * 20, 'static': 'same'}
    combined = probe.join_blocks([first, second], torch)
    assert torch.equal(combined['coords'][:20], first['coords'])
    assert torch.equal(combined['coords'][20:], second['coords'])
    assert combined['meta'] == ['a'] * 20 + ['b'] * 20
    second['static'] = 'different'
    with pytest.raises(ValueError, match='metadata'):
        probe.join_blocks([first, second], torch)
