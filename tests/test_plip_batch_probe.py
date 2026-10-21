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


def test_tensordict_prior_is_joined_as_a_mapping():
    torch = pytest.importorskip('torch')
    tensordict = pytest.importorskip('tensordict')
    a = tensordict.TensorDict({'coords': torch.zeros(20, 10, 3)}, batch_size=[20])
    b = tensordict.TensorDict({'coords': torch.ones(20, 10, 3)}, batch_size=[20])
    result = probe.join_blocks([a, b], torch)
    assert result['coords'].shape == (40, 10, 3)
    assert result['coords'][20:].sum() == 600


def test_mapping_may_iterate_rows_instead_of_keys_without_torch():
    from types import SimpleNamespace
    class RowMapping:
        def keys(self): return ['label']
        def items(self): return [('label', 'same')]
        def __getitem__(self, key):
            assert key == 'label'
            return 'same'
        def __iter__(self):
            raise AssertionError('TensorDict iteration yields rows, not keys')
    assert probe.join_blocks([RowMapping(), RowMapping()], SimpleNamespace(is_tensor=lambda v: False)) == {'label': 'same'}
