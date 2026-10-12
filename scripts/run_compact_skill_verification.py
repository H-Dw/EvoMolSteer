"""Run real-API compact Skills, then a committed, paired candidate-only repeat.

Reuse controls; never restart the completed ablation or select a lucky chain.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from evomolsteer.agent_workflow.compact import run_roles
from evomolsteer.ablation.shared_controls import verify_attempts, verify_initial_states
from evomolsteer.generation.terminal_statistics import converged_energy
from evomolsteer.io import digest, read_json, write_json
from analyze_dependency_study import metrics, paired
from run_m2_skill_api_study import Study, shell


class Verification(Study):
    def cleanup(self, name, restored):
        # Preserve one compressed terminal SDF before normal temporary cleanup.
        target = self.repo / 'data/final_skill_verification' / self.study.name
        target.mkdir(parents=True, exist_ok=True)
        with self.ssh() as client, client.open_sftp() as sftp:
            with sftp.open(self.config['remote_work'] + '/seed_structures/' + name + '/gradient.sdf', 'rb') as stream:
                data = stream.read()
        path = target / 'gradient.sdf.gz'
        path.write_bytes(gzip.compress(data, mtime=0))
        write_json(self.study / 'terminal_structure.json', {'path': path.relative_to(self.repo).as_posix(),
            'sha256': digest(path), 'uncompressed_bytes': len(data)})
        super().cleanup(name, restored)


def compare(repo, config, result):
    study = repo / config['study']
    bank = read_json(repo / config['control_manifest'])
    verify_initial_states(result, bank['result'], config['batches'])
    candidate = pd.read_csv(repo / result['outcomes']['gradient']['metrics'])
    controls = {arm: pd.read_csv(repo / bank['result']['outcomes'][arm]['metrics'])
                for arm in ['gradient', 'unguided']}
    for frame in [candidate, *controls.values()]:
        verify_attempts(frame, config['batches'], 50)
    comparisons = {label: paired(candidate, frame) for label, frame in
                   [('M2', controls['gradient']), ('unguided', controls['unguided'])]}
    steer = pd.read_parquet(repo / config['historical_metrics'])
    # Independent historical cohort: bootstrap its batch means separately.
    a = candidate.groupby('batch').pic50_on_rescore.mean().to_numpy()
    b = steer.groupby('batch').pic50_on_rescore.mean().to_numpy()
    rng = np.random.default_rng(42)
    boots = a[rng.integers(len(a), size=(100000, len(a)))].mean(1) - b[rng.integers(len(b), size=(100000, len(b)))].mean(1)
    comparisons['historical_Steer'] = {'delta_affinity_mean': float(candidate.pic50_on_rescore.mean() - steer.pic50_on_rescore.mean()),
        'batch_bootstrap_CI95': np.quantile(boots, [.025, .975]).tolist(), 'paired': False,
        'equal_online_budget': False, 'historical_n': len(steer)}
    def quality(frame):
        energy = converged_energy(frame)
        return {'attempts': len(frame), 'mean_pic50': float(frame.pic50_on_rescore.mean()),
            'p95_pic50': float(frame.pic50_on_rescore.quantile(.95)),
            'valid': int(frame.valid_connected.sum()), 'PB_pass': int(frame.pb_fast_pass.sum()),
            'strain_median': float(energy.median()), 'strain_p90': float(energy.quantile(.9)),
            'energy_coverage': len(energy) / len(frame),
            'elite_valid': int((frame.valid_connected & (frame.pic50_on_rescore >= config['elite_threshold'])).sum())}
    report = {'primary': 'All-attempt FLOWR predicted pIC50; not experimental affinity',
        'quality': {'compact': quality(candidate), 'M2': quality(controls['gradient']),
                    'unguided': quality(controls['unguided']), 'historical_Steer': quality(steer)},
        'comparisons': comparisons, 'control_manifest_sha256': digest(repo / config['control_manifest']),
        'paired_initial_states_verified': True, 'controls_generated_again': False,
        'candidate_repeat_not_new_independent_RNG': True, 'selection': read_json(study / 'role_selection.json')['selected'],
        'delivery': result['delivery']['batches'], 'elite_threshold': config['elite_threshold']}
    write_json(study / 'verification_quality.json', report)
    pd.DataFrame([{'cohort': name, **q} for name, q in report['quality'].items()]).to_csv(study / 'verification_quality.csv', index=False)
    return report


def run(repo, config):
    bank = read_json(repo / config['control_manifest'])
    for path, sha in {**bank['files'], **bank['generator_sources']}.items():
        if digest(repo / path) != sha:
            raise ValueError('Frozen comparison source changed: ' + path)
    selection = run_roles(repo, config)
    s = Verification(repo, config)
    with s.ssh() as client:
        import shlex
        shell(client, 'mkdir -p ' + shlex.quote(config['remote_work']))
    chosen = selection['selected']
    result = s.infer(repo / chosen['program'], chosen['signature'], arms='gradient',
        lane='compact_repeat', reference_path=repo / chosen['reference'], batches=config['batches'])
    compare(repo, config, result)
    s.save(status='complete', phase='final_verification', task=None)
    s.commit('report(skills): compare compact real API guidance repeat with frozen controls')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = read_json(repo / args.config)
    try:
        run(repo, config)
    except Exception as error:
        write_json(repo / config['study'] / 'failure.json', {'error_type': type(error).__name__, 'error': str(error)})
        raise
