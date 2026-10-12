"""Clock accounting must not double-count nested calls or partial log records."""
import pytest
from evomolsteer.ablation.timing import union_seconds, json_events, describe


def test_overlapping_calls_are_wall_time_not_sum():
    assert union_seconds([(2, 8), (0, 5), (3, 4), (10, 12)]) == 10


def test_zero_length_or_empty_intervals_do_not_add_time():
    assert union_seconds([]) == 0
    assert union_seconds([(3, 3), (4, 5), (5, 5)]) == 1


def test_reversed_clock_rejected():
    with pytest.raises(ValueError, match='Negative'):
        union_seconds([(4, 3)])


def test_noisy_and_incomplete_live_log_lines():
    assert json_events('warning\n{"seconds":3}\n{"seconds":\n[1,2]\n') == [{'seconds':3}]
    assert describe([]) is None
