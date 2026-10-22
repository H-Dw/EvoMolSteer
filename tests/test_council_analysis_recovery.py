"""Recovery changes analysis control only; native samples and evidence stay intact."""
import contextlib
import importlib.util
import sys
from pathlib import Path

import pytest

from evomolsteer.adaptive_schedule.io import read, write, sha, value_sha
from evomolsteer.adaptive_schedule.teacher_pipeline import run_teacher_pipeline

spec = importlib.util.spec_from_file_location('analysis_recovery', Path(__file__).parents[1] / 'scripts/recover_plip_council_analysis.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
sys.modules['recover_plip_council_analysis'] = recovery
spec2 = importlib.util.spec_from_file_location('analysis_resume', Path(__file__).parents[1] / 'scripts/resume_plip_council_analysis.py')
resume = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(resume)


@pytest.fixture
def failed_analysis(tmp_path, monkeypatch):
    root, repo = tmp_path / 'results', tmp_path / 'repo'
    root.mkdir(); repo.mkdir()
    monkeypatch.setattr(recovery, 'idle', lambda root: contextlib.nullcontext())
    monkeypatch.setattr(recovery, 'process_alive', lambda pid: False)
    monkeypatch.setattr(recovery, 'repository', lambda: repo)
    monkeypatch.setattr(recovery, 'commit', lambda: 'recovery-tool')
    monkeypatch.setattr(recovery, 'source_fingerprint', lambda: {'source': 'unchanged'})
    write(root / 'config.json', {'versions': ['v1', 'v2', 'v3'], 'max_recoveries': 3})
    write(root / 'teachers.json', ['A', 'B'])
    frozen = {'config_sha256': sha(root / 'config.json'), 'source_fingerprint': {'source': 'unchanged'}}
    for name in ('PREPARED.json', 'ACCEPTANCE_COMPLETE.json'):
        write(root / name, frozen)
    base = root / 'versions/v3'
    skills = {}
    for role, name in [('analyst', 'plip-group-analyst'), ('synthesizer', 'plip-group-synthesizer')]:
        p = repo / 'skills' / name / 'SKILL.md'; p.parent.mkdir(parents=True)
        p.write_text('Frozen synthetic skill')
        q = base / f'{role}_SKILL.md'; q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes()); skills[role] = sha(p)
    write(base / 'binding.json', dict(frozen, skill_hashes=skills))
    write(base / 'deployment.json', {'commit': 'frozen-runtime'})
    write(root / 'workers.json', {'pid': 123, 'version': 'v3'})
    rnd = base / 'A/round_01'; wave = rnd / 'wave'; council = rnd / 'council'
    for name in ('SAMPLING_COMPLETE.json', 'EVALUATION_COMPLETE.json', 'COMPLETE.json'):
        write(wave / name, {'attempts': 400})
    write(wave / 'quality_records.json', [{'pic50_on_rescore': 4.0}] * 400)
    write(wave / 'final_records.json', [{'pic50_on_rescore': 4.0}] * 400)
    write(wave / 'QUALITY_COMPLETE.json', {'attempts': 400, 'records_sha256': sha(wave / 'quality_records.json')})
    artifacts = {'wave': str(wave), 'parent': 'shared-r0', 'schedules': [[5] * 4] * 400}
    write(rnd / 'artifacts.json', artifacts)
    write(council / 'analyst_failures.json', {'G1': {'error_type': 'LLMInterfaceError', 'message': 'Illegal relative tick'}})
    write(council / 'analyst_timing.json', {'G2': {'start': 1, 'end': 2}})
    for gid in ['G1', 'G2', 'G3', 'G4']:
        folder = council / 'analysts' / gid; folder.mkdir(parents=True)
        if gid == 'G1':
            (folder / 'turn_01.attempt_1.invalid.txt').write_text('Preserve rejected model response')
        else:
            write(folder / 'hypotheses.json', {'cached': True})
    initial = {'A': {}, 'B': {}}
    state = {'binding': {'initial_jobs_sha256': value_sha(initial), 'rounds': 1, 'teachers': ['A', 'B'],
                         'gpu_owners': 1, 'global_llm_concurrency': 4},
             'teachers': {'A': {'phase': 'failed', 'round': 1, 'artifacts': artifacts,
                 'error': {'type': 'RuntimeError', 'message': 'An isolated analyst failed; synthesis was not run; completed requests are retained'}},
                 'B': {'phase': 'complete', 'round': 1}}, 'events': [], 'last_gpu_teacher': 'B'}
    write(base / 'pipeline.json', state)
    return root, base, council, initial


def test_retry_reuses_wave_and_preserves_other_teacher_and_requests(failed_analysis):
    root, base, council, initial = failed_analysis
    before = recovery.protected_inventory(root, base)
    old = read(base / 'pipeline.json')
    assert recovery.recover(root, 'v3')['eligible']
    result = recovery.recover(root, 'v3', apply=True)
    after = read(base / 'pipeline.json')
    assert after['teachers']['A']['phase'] == 'analyzing'
    assert after['teachers']['B'] == old['teachers']['B']
    assert all(sha(root / p) == h for p, h in before.items())
    assert (council / 'saved_attempts' / Path(result['operation']).name / 'analyst_failures.json').is_file()
    assert read(Path(result['operation']) / 'journal.json')['runtime_commit'] == 'frozen-runtime'
    generated = []
    resumed = run_teacher_pipeline(initial, base, rounds=1,
        infer=lambda *args: generated.append(args),
        analyze=lambda t, r, a, c: {'teacher_id': t, 'observed_round': r},
        commit=lambda *args: {'quality_complete': True, 'retention_complete': True})
    assert generated == []
    assert all(t['phase'] == 'complete' for t in resumed['teachers'].values())


def test_live_worker_blocks_recovery(failed_analysis, monkeypatch):
    root, _, _, _ = failed_analysis
    monkeypatch.setattr(recovery, 'process_alive', lambda pid: True)
    with pytest.raises(ValueError, match='still alive'):
        recovery.recover(root, 'v3', apply=True)


def test_other_teacher_must_finish(failed_analysis):
    root, base, _, _ = failed_analysis
    state = read(base / 'pipeline.json'); state['teachers']['B']['phase'] = 'analyzing'
    write(base / 'pipeline.json', state)
    with pytest.raises(ValueError, match='terminal'):
        recovery.recover(root, 'v3', apply=True)


def test_corrupt_quality_blocks_recovery(failed_analysis):
    root, base, _, _ = failed_analysis
    write(base / 'A/round_01/wave/quality_records.json', [])
    with pytest.raises(ValueError, match='evaluated'):
        recovery.recover(root, 'v3', apply=True)


def test_recovery_limit_and_incomplete_journal_block(failed_analysis):
    root, _, _, _ = failed_analysis
    for i in range(3):
        write(root / f'operations/council_analysis_retry_{i}/journal.json',
              {'version': 'v3', 'teachers': ['A'], 'phase': 'complete'})
    with pytest.raises(ValueError, match='budget exhausted'):
        recovery.recover(root, 'v3', apply=True)
    write(root / 'operations/council_analysis_retry_2/journal.json', {'version': 'v3', 'phase': 'prepared'})
    with pytest.raises(ValueError, match='Incomplete recovery'):
        recovery.recover(root, 'v3', apply=True)


def test_frozen_source_and_skill_cannot_change(failed_analysis, monkeypatch):
    root, base, _, _ = failed_analysis
    monkeypatch.setattr(recovery, 'source_fingerprint', lambda: {'source': 'changed'})
    with pytest.raises(ValueError, match='source/config'):
        recovery.recover(root, 'v3')
    monkeypatch.setattr(recovery, 'source_fingerprint', lambda: {'source': 'unchanged'})
    (base / 'analyst_SKILL.md').write_text('Changed instructions')
    with pytest.raises(ValueError, match='skill changed'):
        recovery.recover(root, 'v3')


def test_completed_version_cannot_be_reopened(failed_analysis):
    root, _, _, _ = failed_analysis
    write(root / 'FINALIZED_v3.json', {})
    with pytest.raises(ValueError, match='unfinished'):
        recovery.recover(root, 'v3', apply=True)


def test_non_interface_failure_is_not_silently_retried(failed_analysis):
    root, _, council, _ = failed_analysis
    write(council / 'analyst_failures.json', {'G1': {'error_type': 'ValueError', 'message': 'Input checksum changed'}})
    with pytest.raises(ValueError, match='interface-validation'):
        recovery.recover(root, 'v3', apply=True)


@pytest.fixture
def resume_setup(failed_analysis, monkeypatch):
    root, _, _, _ = failed_analysis
    runtime = root.parent / 'runtime'; runtime.mkdir()
    for name in ('run_plip_group.sh', 'plip_group_pipeline.py', 'launch_plip_adaptive.py'):
        p = runtime / 'scripts' / name; p.parent.mkdir(exist_ok=True)
        p.write_bytes(b'original')
    monkeypatch.setattr(resume, 'recovery_lock', lambda root: contextlib.nullcontext())
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda args, **kw:
                        'frozen-runtime' if args[-1] == 'HEAD' else b'original')
    monkeypatch.setattr(resume, 'process_alive', lambda pid: pid == 999)
    calls = []
    def launch(args, **kwargs):
        calls.append(args)
        write(root / 'launch.json', {'commit': 'frozen-runtime', 'version': 'v3', 'pid': 999})
    monkeypatch.setattr(resume.subprocess, 'run', launch)
    return root, runtime, calls


def test_one_shot_resumes_original_runtime_only_after_recovery(resume_setup):
    root, runtime, calls = resume_setup
    out = resume.resume(root, 'v3', runtime, 0)
    assert out['phase'] == 'resumed' and out['new_worker_pid'] == 999
    assert len(calls) == 1 and str(runtime / 'scripts/run_plip_group.sh') in calls[0]
    assert read(root / 'versions/v3/pipeline.json')['teachers']['A']['phase'] == 'analyzing'


def test_bounded_wait_never_interrupts_live_worker(resume_setup, monkeypatch):
    root, runtime, calls = resume_setup
    monkeypatch.setattr(resume, 'process_alive', lambda pid: True)
    before = sha(root / 'versions/v3/pipeline.json')
    with pytest.raises(RuntimeError, match='remains alive'):
        resume.resume(root, 'v3', runtime, 0)
    assert calls == [] and sha(root / 'versions/v3/pipeline.json') == before


def test_resume_cannot_switch_version_commit(resume_setup, monkeypatch):
    root, runtime, calls = resume_setup
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *args, **kw: 'other-commit')
    with pytest.raises(ValueError, match='original fixed deployment'):
        resume.resume(root, 'v3', runtime, 0)
    assert calls == []


def test_failed_recovery_gate_cannot_launch(resume_setup, monkeypatch):
    root, runtime, calls = resume_setup
    def fail(*args, **kwargs):
        raise ValueError('Synthetic source mismatch')
    monkeypatch.setattr(resume, 'recover', fail)
    with pytest.raises(ValueError, match='source mismatch'):
        resume.resume(root, 'v3', runtime, 0)
    assert calls == []
