"""Contract checks for the read-only, clock-explicit inference audit."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'analyst_suffix_audit', Path(__file__).parents[1]/'scripts/audit_analyst_credit_and_native_suffix.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(score, state, controlled=False):
    return {'score_time': score, 'state_time': state, 'injection_l2_A': [.1 if controlled else 0.],
            'particle_resampled': False, 'reward_evaluated': controlled, 'active': controlled,
            'production_target_forward_calls': 1}


def test_stop_at_state_boundary_then_native():
    result = audit.audit_suffix_rows([row(.49, .5, True), row(.5, .51)], [0., .5], True)
    assert result['controlled_steps'] == 1
    assert result['last_controlled_clocks'] == (.49, .5)
    assert result['first_postwindow_native_clocks'] == (.5, .51)
    assert result['suffix_injection_is_exact_zero']


def test_dynamic_nonzero_start_window_and_native_control():
    rows = [row(.19, .2), row(.2, .21, True), row(.39, .4, True), row(.4, .41)]
    assert audit.audit_suffix_rows(rows, [.2, .4], True)['controlled_steps'] == 2
    native = [row(.2, .21), row(.4, .41)]
    assert audit.audit_suffix_rows(native, [.2, .4], False)['controlled_steps'] == 0


@pytest.mark.parametrize('field,value,match', [
    ('injection_l2_A', [.001], 'contains an injection'),
    ('active', True, 'contains an injection'),
    ('reward_evaluated', True, 'support differs'),
    ('particle_resampled', True, 'unexpectedly resampled'),
    ('production_target_forward_calls', 2, 'forward count'),
    ('injection_l2_A', [float('nan')], 'Nonfinite'),
])
def test_reject_suffix_policy_changes(field, value, match):
    altered = row(.5, .51)
    altered[field] = value
    with pytest.raises(ValueError, match=match):
        audit.audit_suffix_rows([altered], [0., .5], True)
