"""Bounded, audited retry of failed council analysis after the worker is idle.

This operation changes only the pipeline phase. It does not call an LLM, sample,
edit responses, relax validation, or change a frozen runtime. Relaunch the same
version from its original deployment after this script reports success.
"""
import argparse
import contextlib
import copy
import os
from pathlib import Path
import shutil
import time

from agenticgen.adaptive_schedule.io import read, write, sha, value_sha, source_fingerprint, repository, commit
from agenticgen.adaptive_schedule.protocol_state import load_config, config_path, prepared_path, acceptance_path


@contextlib.contextmanager
def idle(root):
    import fcntl
    with contextlib.ExitStack() as stack:
        for name in ('supervisor.lock', 'worker.lock'):
            f = stack.enter_context((root / name).open('a'))
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    return True


def protected_inventory(root, base):
    paths = [root / n for n in ('config.json', 'teachers.json', 'PREPARED.json', 'ACCEPTANCE_COMPLETE.json')]
    paths += [p for p in (root/'protocols').rglob('*') if p.is_file()]
    if (root/'ACTIVE_PROTOCOL.json').exists(): paths.append(root/'ACTIVE_PROTOCOL.json')
    paths += list(root.glob('FINALIZED_*.json'))
    for pattern in ('*/COMPLETE.json', '*/best.json', '*/quality_records.json'):
        paths += list((root / 'shared_r0').glob(pattern))
    # Include all existing response, source, quality and score evidence, including
    # failed requests. Large tensor payloads remain bound by the existing manifests.
    paths += [p for p in base.rglob('*') if p.is_file() and p.name != 'pipeline.json'
              and p.suffix in {'.json', '.txt', '.md', '.xml'}]
    return {p.relative_to(root).as_posix(): sha(p) for p in paths}


def validate_wave_quality(config, wave):
    attempts=config.get('population_size',100)*config.get('hypotheses',4)
    quality=read(wave/'QUALITY_COMPLETE.json')
    if quality['attempts']!=attempts or sha(wave/'quality_records.json')!=quality['records_sha256']:
        raise ValueError('All configured hypothesis attempts must have completed quality evaluation')
    if len(read(wave/'final_records.json'))!=attempts or len(read(wave/'quality_records.json'))!=attempts:
        raise ValueError('Candidate count changed')
    return attempts


def check(root, version):
    config = load_config(root)
    if version not in config['versions'] or (root / f'FINALIZED_{version}.json').exists():
        raise ValueError('Only an unfinished authorized version can be recovered')
    if process_alive(read(root / 'workers.json')['pid']):
        raise ValueError('Worker is still alive; let the other teacher finish')
    base = root / 'versions' / version
    binding = read(base / 'binding.json')
    for frozen in (read(prepared_path(root)), read(acceptance_path(root)), binding):
        if frozen['source_fingerprint'] != source_fingerprint() or frozen['config_sha256'] != sha(config_path(root)):
            raise ValueError('Frozen source/config differs; this tool cannot adopt code changes')
    from agenticgen.adaptive_schedule.search_policy import load_version_policy
    if config.get('search_policy') and binding.get('search_policy_sha256') != value_sha(load_version_policy(root,version)):
        raise ValueError('Version search policy changed')
    for role, name in [('analyst', 'plip-group-analyst'), ('synthesizer', 'plip-group-synthesizer')]:
        if not (sha(base / f'{role}_SKILL.md') == binding['skill_hashes'][role]
                == sha(repository() / 'agenticgen/src/agenticgen/resources/skills' / name / 'SKILL.md')):
            raise ValueError('Frozen skill changed')
    state = read(base / 'pipeline.json')
    if any(t['phase'] not in {'failed', 'complete'} for t in state['teachers'].values()):
        raise ValueError('Every other teacher must be terminal before recovery')
    failed = [name for name, t in state['teachers'].items() if t['phase'] == 'failed']
    if not failed:
        raise ValueError('No failed council analysis to recover')
    prior = []
    for p in (root / 'operations').glob('council_analysis_retry_*/journal.json'):
        j = read(p)
        if j.get('version') == version:
            if j.get('phase') != 'complete':
                raise ValueError('Incomplete recovery transaction requires its journal to be checked')
            prior.append(j)
    limit = config.get('max_recoveries', 0)
    if type(limit) is not int or limit < 1:
        raise ValueError('An explicit positive recovery limit is required')
    for teacher in failed:
        if sum(teacher in j['teachers'] for j in prior) >= limit:
            raise ValueError('Council recovery budget exhausted')
        if Path(teacher).name != teacher or teacher in {'.', '..'}:
            raise ValueError('Unsafe teacher identity')
        t = state['teachers'][teacher]
        rnd = base / teacher / f"round_{t['round']:02d}"
        council, wave = rnd / 'council', rnd / 'wave'
        if t.get('error', {}).get('message') != 'An isolated analyst failed; synthesis was not run; completed requests are retained':
            raise ValueError('Only isolated analyst failures are supported')
        if (council / 'COMPLETE.json').exists() or (wave / 'CLEANUP_COMPLETE.json').exists():
            raise ValueError('Already committed or cleaned data must not be reset')
        if read(rnd / 'artifacts.json') != t['artifacts'] or Path(t['artifacts']['wave']).resolve() != wave.resolve():
            raise ValueError('Analysis artifacts do not match the saved wave')
        for name in ('SAMPLING_COMPLETE.json', 'EVALUATION_COMPLETE.json', 'COMPLETE.json'):
            if not (wave / name).is_file():
                raise ValueError('Sampling/evaluation must already be complete')
        validate_wave_quality(config,wave)
        failures = read(council / 'analyst_failures.json')
        if not failures or not set(failures) <= {'G1', 'G2', 'G3', 'G4'}:
            raise ValueError('Missing isolated failure evidence')
        for gid, failure in failures.items():
            if failure['error_type'] != 'LLMInterfaceError':
                raise ValueError('Only recorded interface-validation failures are supported')
            folder = council / 'analysts' / gid
            if (folder / 'hypotheses.json').exists() or not list(folder.glob('*.invalid.txt')):
                raise ValueError('Failed analyst does not match its preserved responses')
    return base, state, failed, prior


def recover(root, version, *, apply=False):
    root = Path(root).resolve()
    with idle(root):
        base, state, failed, prior = check(root, version)
        if not apply:
            return {'eligible': True, 'version': version, 'teachers': failed, 'prior_recoveries': len(prior)}
        protected = protected_inventory(root, base)
        op = root / 'operations' / f'council_analysis_retry_{version}_{time.time_ns()}'
        after = copy.deepcopy(state)
        for teacher in failed:
            t = after['teachers'][teacher]
            t['phase'] = 'analyzing'
            t.pop('error', None)
        after['events'].append({'kind': 'operator_analysis_retry', 'teachers': failed, 'time': time.time(),
                                'operation': str(op), 'inference_repeated': False})
        journal = {'phase': 'prepared', 'version': version, 'teachers': failed,
                   'recovery_tool_commit': commit(), 'runtime_commit': read(base / 'deployment.json')['commit'],
                   'pipeline_before': state, 'pipeline_after_sha256': value_sha(after),
                   'protected_files_sha256': protected, 'inference_repeated': False,
                   'policy': 'Same input, skill, validation and API budgets; successful responses cached; new failed attempts append.'}
        write(op / 'journal.json', journal)
        for teacher in failed:
            council = base / teacher / f"round_{state['teachers'][teacher]['round']:02d}" / 'council'
            archive = council / 'saved_attempts' / op.name
            archive.mkdir(parents=True)
            # These control files may be rewritten by analysis; keep their exact
            # pre-retry bytes in the eventual master review's source inventory.
            for name in ('analyst_failures.json', 'analyst_timing.json'):
                if (council / name).exists():
                    shutil.copyfile(council / name, archive / name)
        if any(sha(root / name) != h for name, h in protected.items()):
            raise ValueError('Protected data changed during recovery')
        write(base / 'pipeline.json', after)
        journal.update(phase='complete', protected_hashes_equal=True,
                       pipeline_after_file_sha256=sha(base / 'pipeline.json'), completed=time.time())
        write(op / 'journal.json', journal)
        return {'operation': str(op), 'version': version, 'teachers': failed, 'inference_repeated': False,
                'runtime_commit': journal['runtime_commit'], 'ready_for_same_runtime_relaunch': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--version', required=True)
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    print(recover(a.root, a.version, apply=a.apply))
