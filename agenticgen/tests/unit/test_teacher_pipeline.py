import threading

import pytest

from agenticgen.adaptive_schedule.teacher_pipeline import run_teacher_pipeline
from agenticgen.adaptive_schedule.io import read


def test_other_teacher_uses_gpu_during_analysis_and_recovery_skips_completed(tmp_path):
    a_analyzing, b_on_gpu = threading.Event(), threading.Event()
    calls, committed = [], set()
    main_thread = threading.get_ident()
    callers = []
    def infer(teacher, round_no, job):
        assert threading.get_ident() == main_thread
        if round_no > 1:
            assert (teacher, round_no-1) in committed
        if teacher == 'B' and round_no == 1:
            assert a_analyzing.wait(5)
            b_on_gpu.set()
        calls.append((teacher, round_no))
        return {'manifest': f'{teacher}/{round_no}.json'}
    def analyze(teacher, round_no, artifacts, caller):
        assert threading.get_ident() != main_thread
        callers.append(caller)
        if teacher == 'A' and round_no == 1:
            a_analyzing.set()
            assert b_on_gpu.wait(5)
        return {'teacher_id': teacher, 'observed_round': round_no, 'jobs': []}
    def commit(teacher, round_no, artifacts, selection):
        committed.add((teacher, round_no))
        return {'quality_complete': True, 'retention_complete': True}
    kwargs = dict(rounds=2, infer=infer, analyze=analyze, commit=commit)
    state = run_teacher_pipeline({'A': {}, 'B': {}}, tmp_path, **kwargs)
    assert sorted(calls) == [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
    assert len(set(id(c) for c in callers)) == 1
    assert all(t['phase'] == 'complete' for t in state['teachers'].values())
    run_teacher_pipeline({'A': {}, 'B': {}}, tmp_path, **kwargs)
    assert len(calls) == 4


def test_wrong_teacher_does_not_commit_or_restart_and_error_is_preserved(tmp_path):
    commits = []
    def analyze(teacher, round_no, artifacts, caller):
        return {'teacher_id': 'B', 'observed_round': round_no}
    def commit(teacher, *args):
        commits.append(teacher)
        return {'quality_complete': True, 'retention_complete': True}
    kwargs = dict(rounds=2, infer=lambda *args: {}, analyze=analyze, commit=commit)
    with pytest.raises(RuntimeError, match='failed'):
        run_teacher_pipeline({'A': {}, 'B': {}}, tmp_path, **kwargs)
    assert 'A' not in commits
    state = read(tmp_path/'pipeline.json')
    assert state['teachers']['A']['phase'] == 'failed'
    assert state['teachers']['B']['phase'] == 'complete'
    with pytest.raises(RuntimeError, match='explicit recovery'):
        run_teacher_pipeline({'A': {}, 'B': {}}, tmp_path, **kwargs)


def test_quality_or_retention_incomplete_cannot_start_next_round(tmp_path):
    calls = []
    def infer(teacher, round_no, job):
        calls.append((teacher, round_no))
        return {}
    with pytest.raises(RuntimeError, match='failed'):
        run_teacher_pipeline({'A': {}, 'B': {}}, tmp_path, rounds=2, infer=infer,
            analyze=lambda t, r, a, c: {'teacher_id': t, 'observed_round': r},
            commit=lambda *a: {'quality_complete': False, 'retention_complete': True})
    assert calls == [('A', 1), ('B', 1)]
