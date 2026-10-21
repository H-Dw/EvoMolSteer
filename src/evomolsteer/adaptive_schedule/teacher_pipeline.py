"""Readiness-based teacher scheduling with a single GPU owner.

The caller supplies validated native inference, council analysis and CPU commit
adapters. This scheduler never substitutes the legacy twenty-slot runner.
Adapters must be idempotent: journal recovery can repeat an interrupted callback.
"""
import concurrent.futures
from pathlib import Path
import time

from .council import BoundedCaller
from .io import read, write, value_sha


def run_teacher_pipeline(initial_jobs, output, *, rounds, infer, analyze, commit,
                         caller=None):
    """infer runs on this calling thread; network/CPU analysis runs in the pool.

    initial_jobs: teacher -> first group-selection manifest. infer returns frozen
    artifact references; analyze returns a council selection; commit verifies and
    publishes TOP1/quality/cleanup before its result can re-enter the GPU queue.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if len(initial_jobs) != 2 or type(rounds) is not int or not 1 <= rounds <= 10:
        raise ValueError('Two teachers and 1..10 rounds are required')
    binding = {'initial_jobs_sha256': value_sha(initial_jobs), 'rounds': rounds,
               'teachers': list(initial_jobs), 'gpu_owners': 1, 'global_llm_concurrency': 4}
    path = output/'pipeline.json'
    if path.exists():
        state = read(path)
        if state['binding'] != binding:
            raise ValueError('Frozen pipeline jobs or round count changed')
        if any(t['phase'] == 'failed' for t in state['teachers'].values()):
            raise RuntimeError('A recorded failed teacher requires explicit recovery; completed work remains preserved')
    else:
        state = {'binding': binding, 'teachers': {t: {'round': 1, 'phase': 'ready', 'job': job}
                 for t, job in initial_jobs.items()}, 'events': [], 'last_gpu_teacher': None}
        write(path, state)
    caller = caller if isinstance(caller, BoundedCaller) else BoundedCaller(caller) if caller else BoundedCaller()
    def event(kind, teacher, round_no, **extra):
        state['events'].append({'kind': kind, 'teacher': teacher, 'round': round_no, 'time': time.time(), **extra})
        write(path, state)
    def postprocess(teacher, round_no, artifacts):
        selection = analyze(teacher, round_no, artifacts, caller)
        if selection.get('teacher_id') != teacher or selection.get('observed_round') != round_no:
            raise ValueError('Council result belongs to a different teacher or round')
        # commit is CPU-only; must finish quality and retention publication.
        receipt = commit(teacher, round_no, artifacts, selection)
        if not receipt.get('quality_complete') or not receipt.get('retention_complete'):
            raise ValueError('Next inference requires completed quality and verified TOP1 publication/cleanup')
        return selection, receipt
    failures, pending = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Recovery resumes analysis from saved references, never inventing new samples.
        for teacher, t in state['teachers'].items():
            if t['phase'] == 'generating':
                t['phase'] = 'ready'
            elif t['phase'] == 'analyzing':
                pending[pool.submit(postprocess, teacher, t['round'], t['artifacts'])] = teacher
        while True:
            for future in list(pending):
                if not future.done():
                    continue
                teacher = pending.pop(future)
                t = state['teachers'][teacher]
                try:
                    selection, receipt = future.result()
                    if selection.get('teacher_id') != teacher or selection.get('observed_round') != t['round']:
                        raise ValueError('Council result belongs to a different teacher or round')
                    t['last_commit'] = receipt
                    t['last_selection'] = selection
                    completed_round = t['round']
                    if completed_round == rounds:
                        t['phase'] = 'complete'
                    else:
                        t.update(round=completed_round+1, phase='ready', job=selection)
                    event('round_committed', teacher, completed_round)
                except Exception as exc:
                    t['phase'] = 'failed'
                    t['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    failures.append(teacher)
                    event('analysis_or_commit_failed', teacher, t['round'])
            ready = [teacher for teacher, t in state['teachers'].items() if t['phase'] == 'ready']
            if ready:
                # Alternate when both are ready; never idle the GPU for a teacher
                # whose Agent is still running while another teacher is ready.
                teacher = next((t for t in ready if t != state['last_gpu_teacher']), ready[0])
                t = state['teachers'][teacher]
                t['phase'] = 'generating'
                state['last_gpu_teacher'] = teacher
                event('gpu_started', teacher, t['round'])
                try:
                    artifacts = infer(teacher, t['round'], t['job'])
                    t.update(phase='analyzing', artifacts=artifacts)
                    event('gpu_finished_analysis_queued', teacher, t['round'])
                    pending[pool.submit(postprocess, teacher, t['round'], artifacts)] = teacher
                except Exception as exc:
                    t['phase'] = 'failed'
                    t['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    failures.append(teacher)
                    event('inference_failed', teacher, t['round'])
                continue
            if pending:
                concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                continue
            break
    if failures:
        raise RuntimeError('One or more teachers failed; the journal preserves completed rounds and errors')
    write(output/'PIPELINE_COMPLETE.json', {'binding_sha256': value_sha(binding),
          'teacher_rounds': {t: rounds for t in initial_jobs}, 'master_review_required': True})
    return state
