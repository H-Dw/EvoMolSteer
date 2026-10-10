"""Serial inference-only worker for a frozen, explicitly owned ablation cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tarfile
import io
import numpy as np
from evomolsteer.io import read_json, write_json, digest
from run_guidance_scale_campaign import validate_selection_state, retain_terminal
from archive_generation import archive
from retire_experiment_outputs import retire
from evomolsteer.generation.cohort_contract import validate_artifacts

def run(repo, flowr, work, spec_path):
    spec = read_json(spec_path)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if spec['seed'] != 42 or spec['steps'] != 100 or spec['n'] != spec['batch'] * len(spec['batches']):
        raise ValueError('Paired seed and complete batch protocol required')
    if not work.resolve().is_relative_to((flowr / 'experiments').resolve()) or not work.name.startswith('evomolsteer_dependency_'):
        raise ValueError('Explicit owned workspace required')
    if not set(spec['arms'].split(',')) <= {'unguided', 'gradient', 'gradient_zero'}:
        raise ValueError('Steer/SMC arms prohibited')
    checkpoint = flowr / 'checkpoints/flowr_root_v2.ckpt'
    if digest(checkpoint) != 'f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0':
        raise ValueError('Wrong FLOWR checkpoint')
    # Also compare the reference embedded in the reward program. A job can
    # otherwise have correct file hashes but route to the wrong reference.
    program, reference = validate_artifacts(repo, spec)
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / (spec['name'] + '.status.json')
    if status_path.exists(): raise FileExistsError(status_path)
    status = {'status': 'running', 'name': spec['name'], 'code_commit': commit, 'spec_sha256': digest(spec_path), 'started_unix': time.time()}
    write_json(status_path, status)
    try:
        command = [sys.executable, '-u', str(repo / 'scripts/generate_dependency_flowr.py'),
            '--flowr-root', str(flowr), '--input-dataset', str(flowr / 'experiments/ck2_clk3_lineage_20261003'),
            '--root', str(work / 'generated'), '--checkpoint', str(checkpoint),
            '--program', str(program), '--reference', str(reference), '--campaign', spec['name'],
            '--n', str(spec['n']), '--batch', str(spec['batch']), '--steps', '100', '--seed', '42',
            '--arms', spec['arms'], '--batch-indices', ','.join(map(str, spec['batches'])), '--export-terminal']
        write_json(work / (spec['name'] + '.command.json'), {'command': command, 'code_commit': commit})
        sources = ['flowr/models/fm_pocket.py', 'flowr/models/utils.py', 'flowr/data/data_modules.py', 'flowr/util/smc.py']
        # Use the established upstream source list, rather than guessing layout.
        sources = read_json(repo / 'configs/experiments/flowcompat_scale1000_v1/spec.json')['upstream_sources']
        source_args = ['--input-dataset', str(flowr)]
        for name in sources: source_args += ['--relative-source', name]
        before, after = work / (spec['name'] + '.before.json'), work / (spec['name'] + '.after.json')
        subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args, '--output-record', str(before)], check=True)
        with (work / (spec['name'] + '.log')).open('wb') as log:
            subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args, '--reference-record', str(before), '--output-record', str(after)], check=True)
        subprocess.run([sys.executable, str(repo / 'scripts/bind_flowr_source_attestation.py'), '--dataset', str(work / 'generated'), '--campaign', spec['name'], '--before-record', str(before), '--after-record', str(after)], check=True)
        root = work / 'generated/results' / spec['name']
        seeds = retain_terminal(root, work, {'name': spec['name'], 'arms': spec['arms']})
        destination = work / 'archives' / (spec['name'] + '.tar.gz')
        transport = archive(work / 'generated', spec['name'], destination, evaluation_only=True)
        with tarfile.open(destination) as bundle:
            view = json.load(bundle.extractfile('results/' + spec['name'] + '/EVALUATION_VIEW.json'))
            signatures = {}
            for batch in view['batches']:
                if batch['steps'] != 100: raise ValueError('Incomplete inference')
                signatures[batch['batch_path']] = batch['initial_state_signature']
                name = 'results/' + spec['name'] + '/' + batch['batch_path'] + '/execution_state.npz'
                with np.load(io.BytesIO(bundle.extractfile(name).read()), allow_pickle=False) as state:
                    validate_selection_state(state, {'batch': spec['batch'], 'steps': 100})
        expected = {f'{arm}/batch_{batch:03d}' for arm in spec['arms'].split(',') for batch in spec['batches']}
        if set(signatures) != expected: raise ValueError('Wrong batch coverage')
        for batch in spec['batches']:
            if len({signatures[f'{arm}/batch_{batch:03d}'] for arm in spec['arms'].split(',')}) != 1:
                raise ValueError('Initial state mismatch')
        status.update(status='complete', completed_unix=time.time(), complete=read_json(root / 'COMPLETE.json'),
            transport=transport, seed_structures=seeds, initial_state_signatures=signatures,
            particle_selection=False, scientific_evaluation='deferred_local')
        write_json(status_path, status)
        plan = work / (spec['name'] + '.retire.plan.json')
        write_json(plan, {'allowed_bases': [str(work / 'generated/results')], 'targets': [str(root)],
            'result_report': str(status_path), 'protected': [str(repo), str(flowr / 'checkpoints'), str(flowr / 'experiments/ck2_clk3_lineage_20261003'), str(work / 'archives'), str(work / 'seed_structures')]})
        retire(plan, work / (spec['name'] + '.retire.json'), True)
    except Exception as error:
        status.update(status='failed', error=str(error), stopped_unix=time.time())
        write_json(status_path, status)
        raise
    print(json.dumps(status))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True); parser.add_argument('--flowr-root', required=True)
    parser.add_argument('--work', required=True); parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    run(Path(args.repo), Path(args.flowr_root), Path(args.work), Path(args.spec))
