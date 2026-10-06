import copy
import pytest
from evomolsteer.generation.motif_verification import effective_program, verify_frozen


def programs():
    base = {'round': 9, 'program_id': 'R9', 'derivation': {'plan': 'discovery'},
            'window': [0., .5], 'native_rms_ratio': .05, 'reference_sha256': 'strict',
            'constraints': {'max_atom_step_A': .025}, 'unknown_future_control': 1}
    return {n: {**copy.deepcopy(base), 'round': n, 'program_id': f'R{n}',
                'derivation': {'plan': f'R{n}'}} for n in (9, 13, 14, 15)}


def test_round_metadata_does_not_redefine_frozen_reward():
    rows = programs()
    assert effective_program(rows[9]) == effective_program(rows[15])
    assert len(verify_frozen(rows, 9)) == 64


@pytest.mark.parametrize('field,value', [('native_rms_ratio', .1), ('window', [0., 1.]),
                                      ('reference_sha256', 'outside'), ('unknown_future_control', 2)])
def test_all_actual_parameters_including_future_controls_are_frozen(field, value):
    rows = programs(); rows[15][field] = value
    with pytest.raises(ValueError, match='reward changed'):
        verify_frozen(rows, 9)


def test_legacy_winner_cannot_be_used_even_if_all_replays_match():
    rows = programs(); rows[2] = rows[9]
    with pytest.raises(ValueError, match='strict-scope'):
        verify_frozen(rows, 2)
