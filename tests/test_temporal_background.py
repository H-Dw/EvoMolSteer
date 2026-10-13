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


def test_confirmation_loads_only_missing_cohort(tmp_path, monkeypatch):
    import pandas as pd
    from evomolsteer.io import write_json
    study = coordinator(tmp_path / 'study')
    study.repo = tmp_path
    study.plan = {'control_manifest': 'bank.json', 'screen_batches': [0, 1, 2, 3],
                  'confirmation_batches': list(range(8))}
    write_json(tmp_path / 'bank.json', {'effective_signature': 'M2'})
    calls = []
    def infer(program, signature, *, batches, lane, reference_path):
        calls.append(list(batches))
        target = tmp_path / ('cohort_' + str(batches[0]) + '.csv')
        pd.DataFrame({'batch': batches, 'slot': [0] * len(batches)}).to_csv(target, index=False)
        return {'outcomes': {'gradient': {'metrics': target.name}}, 'initial_state_signatures': {}}
    study.infer = infer
    monkeypatch.setattr(module, 'compare_result', lambda *args: {'eligible_for_400': True})
    study.measure(tmp_path / 'program.json', tmp_path / 'reference.json', 'candidate', [0, 1, 2, 3], 'screen')
    result = study.measure(tmp_path / 'program.json', tmp_path / 'reference.json', 'candidate', list(range(8)), 'confirmation')
    assert calls == [[0, 1, 2, 3], [4, 5, 6, 7]]
    merged = pd.read_csv(tmp_path / result['result']['outcomes']['gradient']['metrics'])
    assert merged.batch.tolist() == list(range(8))


def test_cross_version_reuse_excludes_failed_roles_and_checks_metric_hash(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    from evomolsteer.io import write_json, digest
    study = coordinator(tmp_path / 'new')
    study.repo = tmp_path
    study.state['program_results'] = {}
    study.plan = {'source_sha256': {'input': 'frozen'}, 'control_manifest': 'bank.json',
                  'batch_size': 1, 'recipes': {'x': {}}}
    study.config = {'reuse_audit': 'audit.json'}
    write_json(tmp_path / 'bank.json', {'generator_sources': {}, 'result': {}})
    source = tmp_path / 'old'
    metric = source / 'metric.csv'
    source.mkdir()
    pd.DataFrame({'batch': [0], 'slot': [0], 'pic50_on_rescore': [7.5],
                  'valid_connected': [True], 'pb_fast_pass': [True],
                  'energy_status': ['converged'], 'mmff_relief_per_heavy': [.5]}).to_csv(metric, index=False)
    write_json(tmp_path / 'audit.json', {'source_sha256': {'old/metric.csv': digest(metric)}})
    study.config['reuse_audit_sha256'] = digest(tmp_path / 'audit.json')
    result = {'name': 'cohort', 'effective_signature': 'signature',
              'outcomes': {'gradient': {'metrics': 'old/metric.csv'}}}
    write_json(source / 'binding.json', {'source_sha256': {'input': 'frozen'},
               'control_manifest_sha256': digest(tmp_path / 'bank.json')})
    write_json(source / 'transport_records/cohort.execution_audit.json', {'passed': True})
    write_json(source / 'run_status.json', {'status': 'complete_with_skipped_errors',
        'program_results': {'key': result}, 'jobs': {
            'control_x_screen': {'status': 'complete', 'result': {'result': result}},
            'roles_old': {'status': 'skipped_error'}}})
    monkeypatch.setattr(module, 'verify_initial_states', lambda *a: None)
    monkeypatch.setattr(module, 'compile_recipe', lambda *a: ('signature', 'reference'))
    monkeypatch.setattr(module, 'program_signature', lambda p: p)
    study.import_completed_results(source)
    assert list(study.state['jobs']) == ['control_x_screen']
    assert study.state['program_results']['key'] == result
    metric.write_text('batch,slot\n0,1\n')
    with pytest.raises(ValueError, match='archived numerical evidence'):
        study.import_completed_results(source)


def test_reuse_only_prohibits_new_generation_before_remote_access(tmp_path):
    import pytest
    from evomolsteer.io import write_json
    study = coordinator(tmp_path / 'study')
    study.repo = tmp_path
    study.config = {'reuse_only': True, 'prior_seed_mode': 'indexed_batch'}
    study.state['program_results'] = {}
    write_json(tmp_path / 'ref.json', {})
    def no_network(*args, **kwargs):
        raise AssertionError('Must not reach remote connection')
    study.ssh = no_network
    with pytest.raises(ValueError, match='prohibits new FLOWR'):
        study.infer(tmp_path / 'program.json', 'signature', lane='temporal',
                    reference_path=tmp_path / 'ref.json', batches=[0, 1])
