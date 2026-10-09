"""Launch a detached, frozen FLOWR sample-size comparison; no monitor is created.

Uses the existing numerical adapter unchanged. The worker runs cohorts serially,
verifies terminal transport, then retires only its owned full test trajectories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_spec(spec):
    if spec.get('schema_version') != 'guidance-sample-size-1.0':
        raise ValueError('Registered sample-size specification required')
    n, batch, ids = spec['n_per_arm'], spec['batch'], spec['batch_indices']
    if type(n) is not int or type(batch) is not int or min(n, batch) < 1 or n % batch:
        raise ValueError('Complete positive batches required')
    if not isinstance(ids, list) or len(ids) != n // batch or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Exact distinct global batches required')
    if spec['seed'] != 42 or spec['steps'] != 100:
        raise ValueError('Frozen master seed and integration grid required')
    if len(spec['cohorts']) != 2 or [c['name'] for c in spec['cohorts']] != ['R11', 'R26_native']:
        raise ValueError('Serial frozen candidate and paired controls required')
    if [c['arms'] for c in spec['cohorts']] != ['gradient', 'unguided,gradient']:
        raise ValueError('No resampling cohort or hidden extra arm permitted')
    return spec


def generation_command(repo, flowr, work, spec, cohort, checkpoint):
    return [sys.executable, '-u', str(repo / 'scripts/generate_flowcompat_v2_flowr.py'),
        '--flowr-root', str(flowr), '--input-dataset', str(flowr / spec['historical_dataset']),
        '--root', str(work / 'generated'), '--checkpoint', str(checkpoint),
        '--program', str(repo / cohort['program']), '--reference', str(repo / cohort['reference']),
        '--campaign', cohort['name'], '--n', str(spec['n_per_arm']), '--batch', str(spec['batch']),
        '--steps', str(spec['steps']), '--seed', str(spec['seed']), '--arms', cohort['arms'],
        '--batch-indices', ','.join(map(str, spec['batch_indices'])), '--export-terminal']


def validate_inputs(repo, flowr, work, spec_path):
    spec = validate_spec(read(spec_path))
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Committed clean numerical source required')
    if not work.is_relative_to(flowr / 'experiments') or work == flowr / 'experiments':
        raise ValueError('Explicit new FLOWR experiment workspace required')
    if work.name != spec['experiment_name']:
        raise ValueError('Workspace and experiment identity differ')
    checkpoint = flowr / spec['checkpoint']
    if sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Frozen FLOWR checkpoint mismatch')
    old = read(flowr / spec['historical_dataset'] / 'results/main1000_w050/config.json')['experiment']
    if (old['n'], old['batch'], old['seed']) != (spec['n_per_arm'], spec['batch'], spec['seed']):
        raise ValueError('Requested sample count must match actual historical Steer')
    windows = []
    for cohort in spec['cohorts']:
        for key in ['program', 'reference']:
            path = (repo / cohort[key]).resolve()
            if not path.is_relative_to(repo / 'configs') or sha(path) != cohort[key + '_sha256']:
                raise ValueError('Frozen program/reference mismatch')
        program = read(repo / cohort['program'])
        if program['reference_sha256'] != cohort['reference_sha256'] or program['seed'] != spec['seed']:
            raise ValueError('Reward/reference contract mismatch')
        if program['affinity_head_gradient'] or program['additional_per_step_affinity_calls'] != 0:
            raise ValueError('Extra affinity optimization is forbidden')
        windows.append(program['window'])
    if windows[0] != windows[1]:
        raise ValueError('Paired rewards require the same learned window')
    return spec, checkpoint


def validate_execution_view(view, spec, arms):
    expected = {(arm, batch) for arm in arms.split(',') for batch in spec['batch_indices']}
    observed, signatures = set(), {}
    for item in view['batches']:
        arm, directory = item['batch_path'].split('/')
        batch = int(directory.removeprefix('batch_'))
        if item['steps'] != spec['steps']:
            raise ValueError('Incomplete native inference')
        observed.add((arm, batch))
        signatures[f'{arm}/{batch}'] = item['initial_state_signature']
    if observed != expected or len(view['batches']) != len(expected):
        raise ValueError('Complete exact batch coverage required')
    return signatures


def retain_terminal(run, work, cohort):
    destination = work / 'seed_structures' / cohort['name']
    destination.mkdir(parents=True)
    result = {}
    for arm in cohort['arms'].split(','):
        target = destination / (arm + '.sdf')
        sources = sorted((run / arm).glob('batch_*/molecules_all_built.sdf'))
        with target.open('wb') as output:
            for source in sources:
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, output)
        result[arm] = {'path': str(target), 'sha256': sha(target), 'source_batches': len(sources)}
    shutil.copyfile(run / 'final_records.json', destination / 'final_records.json')
    write(destination / 'manifest.json', result)
    return result


def validate_selection_state(state, spec):
    import numpy as np
    expected = np.tile(np.arange(spec['batch']), (spec['steps'], 1))
    if state['resampled'].shape != (spec['steps'],) or state['resampled'].any() or not np.array_equal(state['selected_indices'], expected):
        raise ValueError('Particle selection detected or incomplete selection records')


def validate_pairing(left, right, batches):
    for batch in batches:
        signatures = [left['initial_state_signatures'][f'gradient/{batch}'],
            right['initial_state_signatures'][f'gradient/{batch}'], right['initial_state_signatures'][f'unguided/{batch}']]
        if len(set(signatures)) != 1:
            raise ValueError('Three-arm initial states are not paired')


def worker(repo, flowr, work, spec_path):
    spec, checkpoint = validate_inputs(repo, flowr, work, spec_path)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    status = read(work / 'launch.json')
    if status['code_commit'] != commit or status['spec_sha256'] != sha(spec_path):
        raise ValueError('Detached source changed after launch')
    status.update(status='running', started_unix=time.time(), completed_cohorts=[])
    write(work / 'status.json', status)
    from archive_generation import archive
    from retire_experiment_outputs import retire
    try:
        for cohort in spec['cohorts']:
            if shutil.disk_usage(work).free < spec['minimum_free_bytes']:
                raise ValueError('Insufficient space; no new cohort launched')
            # Upstream identity is independently captured before and after each run.
            before = work / (cohort['name'] + '.upstream.before.json')
            after = work / (cohort['name'] + '.upstream.after.json')
            source_args = ['--input-dataset', str(flowr)]
            for name in spec['upstream_sources']:
                source_args += ['--relative-source', name]
            subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args, '--output-record', str(before)], check=True)
            command = generation_command(repo, flowr, work, spec, cohort, checkpoint)
            write(work / (cohort['name'] + '.command.json'), {'command': command, 'code_commit': commit})
            with (work / (cohort['name'] + '.log')).open('wb') as log:
                subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args, '--reference-record', str(before), '--output-record', str(after)], check=True)
            subprocess.run([sys.executable, str(repo / 'scripts/bind_flowr_source_attestation.py'), '--dataset', str(work / 'generated'), '--campaign', cohort['name'], '--before-record', str(before), '--after-record', str(after)], check=True)
            run = work / 'generated/results' / cohort['name']
            complete = read(run / 'COMPLETE.json')
            if complete['records'] != spec['n_per_arm'] * len(cohort['arms'].split(',')):
                raise ValueError('Actual generation count mismatch')
            seeds = retain_terminal(run, work, cohort)
            archive_path = work / 'archives' / (cohort['name'] + '.tar.gz')
            receipt = archive(work / 'generated', cohort['name'], archive_path, evaluation_only=True)
            with tarfile.open(archive_path, 'r:gz') as bundle:
                view = json.load(bundle.extractfile('results/' + cohort['name'] + '/EVALUATION_VIEW.json'))
                signatures = validate_execution_view(view, spec, cohort['arms'])
                import numpy as np
                for item in view['batches']:
                    name = 'results/' + cohort['name'] + '/' + item['batch_path'] + '/execution_state.npz'
                    import io
                    with np.load(io.BytesIO(bundle.extractfile(name).read()), allow_pickle=False) as state:
                        validate_selection_state(state, spec)
            summary = {'cohort': cohort['name'], 'complete': complete, 'terminal_seeds': seeds,
                'transport': receipt, 'initial_state_signatures': signatures, 'scientific_evaluation': 'deferred_local'}
            summary_path = work / (cohort['name'] + '.complete.json')
            write(summary_path, summary)
            cleanup = {'allowed_bases': [str(work / 'generated/results')], 'targets': [str(run)],
                'result_report': str(summary_path), 'protected': [str(repo), str(flowr / 'checkpoints'), str(flowr / spec['historical_dataset']), str(work / 'archives'), str(work / 'seed_structures')]}
            plan = work / (cohort['name'] + '.cleanup.plan.json')
            write(plan, cleanup)
            retire(plan, work / (cohort['name'] + '.cleanup.json'), True)
            status['completed_cohorts'].append(cohort['name'])
            write(work / 'status.json', status)
        left, right = read(work / 'R11.complete.json'), read(work / 'R26_native.complete.json')
        validate_pairing(left, right, spec['batch_indices'])
        status.update(status='complete', completed_unix=time.time(), paired_initial_states=True,
            actual_attempts_per_arm=spec['n_per_arm'], particle_resampling=False,
            affinity_head_gradient=False, terminal_structures_retained=True)
    except Exception as error:
        status.update(status='failed', failure_type=type(error).__name__, error=str(error), stopped_unix=time.time())
        write(work / 'status.json', status)
        raise
    write(work / 'status.json', status)
    print(json.dumps(status))


def launch(repo, flowr, work, spec_path):
    spec, _ = validate_inputs(repo, flowr, work, spec_path)
    if (work / 'launch.json').exists():
        raise FileExistsError('One-shot task already submitted; inspect its saved status')
    work.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(repo / 'src'), str(flowr),
        str(flowr / 'experiments/evomolsteer_online_20261004/runtime_deps'), os.environ.get('PYTHONPATH', '')]),
        'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}
    command = [sys.executable, '-u', str(repo / 'scripts/run_guidance_scale_campaign.py'), '--mode', 'run',
        '--repo', str(repo), '--flowr-root', str(flowr), '--work', str(work), '--spec', str(spec_path)]
    # Child briefly waits for the atomic launch receipt before entering worker.
    with (work / 'worker.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    result = {'status': 'submitted', 'pid': child.pid, 'code_commit': commit, 'spec_sha256': sha(spec_path),
        'work': str(work), 'n_per_arm': spec['n_per_arm'], 'arms': ['R11', 'R26', 'unguided'],
        'batch_indices': spec['batch_indices'], 'master_seed': spec['seed'], 'submitted_unix': time.time(),
        'monitor': 'none', 'source_checkout': str(repo)}
    write(work / 'status.json', result)
    # Release the worker only after its status record is fully persisted.
    write(work / 'launch.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['launch', 'run'], required=True)
    for name in ['repo', 'flowr-root', 'work', 'spec']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    repo, flowr, work, spec = map(lambda value: Path(value).resolve(), [a.repo, a.flowr_root, a.work, a.spec])
    if a.mode == 'launch':
        print(json.dumps(launch(repo, flowr, work, spec)))
    else:
        for _ in range(100):
            if (work / 'launch.json').exists():
                break
            time.sleep(.1)
        try:
            worker(repo, flowr, work, spec)
        except Exception as error:
            saved = read(work / 'status.json') if (work / 'status.json').exists() else {}
            saved.update(status='failed', failure_type=type(error).__name__, error=str(error), stopped_unix=time.time())
            write(work / 'status.json', saved)
            raise
