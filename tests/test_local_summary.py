import pytest

from evomolsteer.path_replay.local_summary import merge_condition, source_identity


def test_source_identity_ignores_copy_annotation_but_keeps_device_and_source():
    audit = {'device_name': 'GPU_A', 'implementation': 'frozen', 'condition': {'id': 'zero'}}
    reused = dict(audit, reused_current_from='receipt', condition={'id': 'zero', 'reuse_current_from': 'receipt'})
    assert source_identity(audit) == source_identity(reused)
    assert source_identity(audit) != source_identity(dict(reused, device_name='GPU_B'))
    assert source_identity(audit) != source_identity(dict(reused, implementation='changed'))


def test_identical_reused_controls_are_counted_once():
    data, signatures = {}, {}
    rows = [{'id': 'original_path', 'pic50': 8.2}]
    merge_condition(data, signatures, 'zero', rows, 'same-source-and-artifacts')
    merge_condition(data, signatures, 'zero', list(rows), 'same-source-and-artifacts')
    assert len(data['zero']) == 1


@pytest.mark.parametrize('rows,signature', [
    ([{'id': 'original_path', 'pic50': 8.3}], 'same-source-and-artifacts'),
    ([{'id': 'another_path', 'pic50': 8.2}], 'same-source-and-artifacts'),
    ([{'id': 'original_path', 'pic50': 8.2}], 'different-device-or-artifacts'),
])
def test_same_condition_name_cannot_hide_changed_controls(rows, signature):
    data, signatures = {}, {}
    merge_condition(data, signatures, 'zero', [{'id': 'original_path', 'pic50': 8.2}], 'same-source-and-artifacts')
    with pytest.raises(ValueError, match='different results or execution evidence'):
        merge_condition(data, signatures, 'zero', rows, signature)
