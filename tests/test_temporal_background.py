"""Failure isolation tests; no SSH, API or molecular generation."""
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('temporal_background', ROOT / 'scripts/run_temporal_guidance_study.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def coordinator(tmp_path):
    study = module.TemporalStudy.__new__(module.TemporalStudy)
    study.study = tmp_path
    study.state = {'jobs': {}, 'failures': []}
    study.save = lambda **values: study.state.update(values)
    return study


def test_failed_case_does_not_stop_next_case(tmp_path):
    study = coordinator(tmp_path)
    def fail():
        raise ValueError('synthetic invalid response')
    assert study.attempt('first', fail) is None
    assert study.attempt('second', lambda: {'passed': True}) == {'passed': True}
    assert study.state['jobs']['first']['status'] == 'skipped_error'
    assert study.state['jobs']['second']['status'] == 'complete'
    assert study.state['failures'][0]['fallback_not_counted_as_success'] is True
    assert (tmp_path / 'failures/first.json').is_file()


def test_resume_never_relaunches_completed_or_skipped_cases(tmp_path):
    study = coordinator(tmp_path)
    study.state['jobs'] = {'done': {'status': 'complete', 'result': {'cached': True}},
                           'bad': {'status': 'skipped_error'}}
    def should_not_run():
        raise AssertionError('already attempted')
    assert study.attempt('done', should_not_run) == {'cached': True}
    assert study.attempt('bad', should_not_run) is None
