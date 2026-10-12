"""Remote inference-only temporal cohort, with compact terminal transport."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import numpy as np

from archive_generation import archive
from retire_experiment_outputs import retire
from run_guidance_scale_campaign import validate_selection_state, retain_terminal
from evomolsteer.generation.cohort_contract import validate_artifacts
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.policy import canonical_hash, validate_policy


def run(repo, flowr, work, spec_path):
    spec = read_json(spec_path)
    if (spec['seed'] != 42 or spec['steps'] != 100 or spec['arms'] != 'gradient'
            or spec['prior_seed_mode'] != 'indexed_batch'
            or spec['n'] != spec['batch'] * len(set(spec['batches']))):
        raise ValueError('Frozen complete candidate-only indexed batches required')
    if not work.resolve().is_relative_to((flowr / 'experiments').resolve()) or not work.name.startswith('evomolsteer_dependency_temporal_'):
        raise ValueError('Explicit temporal workspace required')
    program, reference = validate_artifacts(repo, spec)
    p = read_json(program)
    import gzip
    ref = json.loads(gzip.decompress(reference.read_bytes()))
    validate_policy(p['temporal_policy'], ref['window'], ref['times'], spec['steps'])
    if canonical_hash(p['temporal_policy']) != p['temporal_policy_sha256']:
        raise ValueError('Schedule digest mismatch')
    bank = read_json(repo / 'docs/experiments/m2_tools_full_drop_ablation_20261012/shared_control_bank/manifest.json')
    if any(digest(repo / name) != sha for name, sha in bank['generator_sources'].items()):
        raise ValueError('Frozen generator sources changed')
    checkpoint = flowr / 'checkpoints/flowr_root_v2.ckpt'
    if digest(checkpoint) != bank['checkpoint_sha256']:
        raise ValueError('Wrong frozen checkpoint')
    work.mkdir(parents=True, exist_ok=True)
    name = spec['name']
    if '/' in name or '\\' in name or name in {'.', '..'}:
        raise ValueError('One cohort name required')
    status_path = work / (name + '.status.json')
    if status_path.exists():
        raise FileExistsError(status_path)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    status = dict(status='running', name=name, code_commit=commit, spec_sha256=digest(spec_path), started_unix=time.time())
    write_json(status_path, status)
    root = work / 'generated/results' / name
    try:
        command = [sys.executable, '-u', str(repo / 'scripts/generate_scheduled_dependency_flowr.py'),
            '--allow-execution', '--flowr-root', str(flowr),
            '--input-dataset', str(flowr / 'experiments/ck2_clk3_lineage_20261003'),
            '--root', str(work / 'generated'), '--checkpoint', str(checkpoint),
            '--program', str(program), '--reference', str(reference), '--campaign', name,
            '--n', str(spec['n']), '--batch', str(spec['batch']), '--steps', str(spec['steps']),
            '--seed', str(spec['seed']), '--arms', 'gradient',
            '--batch-indices', ','.join(map(str, spec['batches'])), '--prior-seed-mode', 'indexed_batch', '--export-terminal']
        write_json(work / (name + '.command.json'), {'command': command, 'code_commit': commit})
        source_args = ['--input-dataset', str(flowr)]
        for item in read_json(repo / 'configs/experiments/flowcompat_scale1000_v1/spec.json')['upstream_sources']:
            source_args += ['--relative-source', item]
        before, after = work / (name + '.before.json'), work / (name + '.after.json')
        subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args, '--output-record', str(before)], check=True)
        with (work / (name + '.log')).open('wb') as log:
            subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, str(repo / 'scripts/attest_flowr_sources.py'), *source_args,
                        '--reference-record', str(before), '--output-record', str(after)], check=True)
        subprocess.run([sys.executable, str(repo / 'scripts/bind_flowr_source_attestation.py'),
                        '--dataset', str(work / 'generated'), '--campaign', name,
                        '--before-record', str(before), '--after-record', str(after)], check=True)
        seeds = retain_terminal(root, work, {'name': name, 'arms': 'gradient'})
        destination = work / 'archives' / (name + '.tar.gz')
        transport = archive(work / 'generated', name, destination, evaluation_only=True)
        signatures = {}
        with tarfile.open(destination) as bundle:
            view = json.load(bundle.extractfile('results/' + name + '/EVALUATION_VIEW.json'))
            for batch in view['batches']:
                if batch['steps'] != spec['steps']:
                    raise ValueError('Incomplete native suffix')
                key = batch['batch_path']
                signatures[key] = batch['initial_state_signature']
                if signatures[key] != bank['result']['initial_state_signatures'][key]:
                    raise ValueError('Candidate initial input differs from fixed M2')
                member = 'results/' + name + '/' + key + '/execution_state.npz'
                with np.load(io.BytesIO(bundle.extractfile(member).read()), allow_pickle=False) as arrays:
                    validate_selection_state(arrays, {'batch': spec['batch'], 'steps': spec['steps']})
        if set(signatures) != {f'gradient/batch_{b:03d}' for b in spec['batches']}:
            raise ValueError('Wrong attempt coverage')
        status.update(status='complete', completed_unix=time.time(), complete=read_json(root / 'COMPLETE.json'),
                      transport=transport, seed_structures=seeds, initial_state_signatures=signatures,
                      particle_selection=False, scientific_evaluation='deferred_local')
        write_json(status_path, status)
    except Exception as error:
        status.update(status='failed', stopped_unix=time.time(), error_type=type(error).__name__, error=str(error))
        write_json(status_path, status)
        # Keep failure traces/logs; retire only this newly owned cohort's arrays.
        if root.is_dir():
            for path in root.glob('gradient/batch_*/guidance_trace.jsonl'):
                target = work / 'failure_diagnostics' / name / path.parent.name / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        raise
    finally:
        if root.exists():
            plan = work / (name + '.retire.plan.json')
            write_json(plan, {'allowed_bases': [str(work / 'generated/results')], 'targets': [str(root)],
                'result_report': str(status_path), 'protected': [str(repo), str(flowr / 'checkpoints'),
                str(flowr / 'experiments/ck2_clk3_lineage_20261003'), str(work / 'archives'), str(work / 'seed_structures')]})
            retire(plan, work / (name + '.retire.json'), True)
    print(json.dumps(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['repo', 'flowr-root', 'work', 'spec']:
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    run(Path(args.repo), Path(args.flowr_root), Path(args.work), Path(args.spec))
