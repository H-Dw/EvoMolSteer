"""Detached local API/analysis coordinator; serial remote FLOWR, skip failed cases."""
import argparse
import copy
from pathlib import Path
import shlex
import time
import traceback

import numpy as np
import pandas as pd

from run_m2_skill_api_study import Study, shell
from evomolsteer.ablation.shared_controls import verify_attempts, verify_initial_states
from evomolsteer.ablation.statistics import compare, summarize, quality_envelope
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.delivery import audit_delivery
from evomolsteer.temporal_guidance.policy import program_signature
from evomolsteer.temporal_guidance.workflow import compile_recipe, prepare, run_roles, sources


def compare_result(repo, plan, result, batches):
    bank = read_json(repo / plan['control_manifest'])
    verify_initial_states(result, bank['result'], batches)
    candidate = pd.read_csv(repo / result['outcomes']['gradient']['metrics'])
    verify_attempts(candidate, batches, plan['batch_size'])
    controls = {arm: pd.read_csv(repo / bank['result']['outcomes'][arm]['metrics']) for arm in ['gradient', 'unguided']}
    controls = {k: f.loc[f.batch.isin(batches)].copy() for k, f in controls.items()}
    for frame in controls.values():
        verify_attempts(frame, batches, plan['batch_size'])
    threshold = plan['evaluation']['elite_threshold']
    summaries = {'candidate': summarize([candidate], threshold),
                 'M2': summarize([controls['gradient']], threshold),
                 'unguided': summarize([controls['unguided']], threshold)}
    comparisons = {k: compare([candidate], [f], family=len(plan['recipes']), seed=42)
                   for k, f in [('M2', controls['gradient']), ('unguided', controls['unguided'])]}
    steer = pd.read_parquet(repo / plan['historical_metrics'])
    summaries['Steer'] = summarize([steer], threshold)
    a = candidate.groupby('batch').pic50_on_rescore.mean().to_numpy()
    b = steer.groupby('batch').pic50_on_rescore.mean().to_numpy()
    rng = np.random.default_rng(42)
    boot = a[rng.integers(len(a), size=(20000, len(a)))].mean(1) - b[rng.integers(len(b), size=(20000, len(b)))].mean(1)
    comparisons['Steer'] = {'delta_pic50': float(a.mean() - b.mean()), 'CI95': np.quantile(boot, [.025, .975]).tolist(),
                            'paired': False, 'equal_online_budget': False, 'historical_n': len(steer)}
    envelope = quality_envelope(summaries['candidate'], summaries['M2'])
    eligible = (comparisons['M2']['simultaneous_CI'][0] > -plan['evaluation']['affinity_noninferiority_margin']
                and comparisons['unguided']['simultaneous_CI'][0] > 0 and envelope['passed'])
    return {'batches': batches, 'summary': summaries, 'comparisons': comparisons, 'secondary_envelope': envelope,
            'eligible_for_400': bool(eligible), 'confirmation_is_extended_development_not_independent': True,
            'score_coverage': 1., 'controls_generated_again': False,
            'screening_rule': 'simultaneous affinity noninferiority vs M2 and advantage vs unguided, secondary quality envelope',
            'extreme_tail_note': '200/400 attempts are underpowered for rare paths; no automatic expansion'}


class TemporalStudy(Study):
    def __init__(self, repo, config):
        super().__init__(repo, config)
        self.plan = read_json(repo / config['plan_config'])
        if self.plan['output'] != config['study']:
            raise ValueError('Runner/plan output mismatch')
        self.state.setdefault('jobs', {})
        self.state.setdefault('failures', [])
        if config.get('reuse_study'):
            self.import_completed_results(repo / config['reuse_study'])

    def import_completed_results(self, source):
        """Reuse verified numerical cohorts, never failed roles or obsolete prompts."""
        source = source.resolve()
        if not source.is_relative_to(self.repo) or source == self.study:
            raise ValueError('Reuse source must be a different study inside this repository')
        previous = read_json(source / 'run_status.json')
        if previous['status'] not in {'complete', 'complete_with_skipped_errors'}:
            raise ValueError('Only a finished numerical study is eligible for cross-version reuse')
        binding = read_json(source / 'binding.json')
        if self.config.get('reuse_audit'):
            audit = self.repo / self.config['reuse_audit']
            if digest(audit) != self.config['reuse_audit_sha256']:
                raise ValueError('Changed pinned numerical collection audit')
            for name, sha in read_json(audit)['source_sha256'].items():
                if digest(self.repo / name) != sha:
                    raise ValueError('Changed archived numerical evidence: ' + name)
        if binding['source_sha256'] != self.plan['source_sha256']:
            raise ValueError('Cross-version reuse changed the frozen scientific sources')
        bank = read_json(self.repo / self.plan['control_manifest'])
        if binding['control_manifest_sha256'] != digest(self.repo / self.plan['control_manifest']):
            raise ValueError('Cross-version reuse changed the RNG/control bank')
        drift = {name: {'frozen_sha256': sha, 'current_sha256': digest(self.repo / name)}
                 for name, sha in bank['generator_sources'].items() if digest(self.repo / name) != sha}
        if self.config.get('reuse_only') and not self.config.get('reuse_audit'):
            raise ValueError('Reuse-only mode requires a pinned collection audit')
        for name in drift:
            if not self.config.get('reuse_only'):
                raise ValueError('Changed frozen generator: ' + name)
        source_files = {}
        for result in previous['program_results'].values():
            path = self.repo / result['outcomes']['gradient']['metrics']
            frame = pd.read_csv(path)
            batches = sorted(frame.batch.unique().tolist())
            verify_attempts(frame, batches, self.plan['batch_size'])
            verify_initial_states(result, bank['result'], batches)
            audit_path = source / 'transport_records' / (result['name'] + '.execution_audit.json')
            if not read_json(audit_path)['passed']:
                raise ValueError('Reuse cohort failed the original execution audit')
            source_files[path.relative_to(self.repo).as_posix()] = digest(path)
            source_files[audit_path.relative_to(self.repo).as_posix()] = digest(audit_path)
        for job, data in previous['jobs'].items():
            if data['status'] != 'complete' or not job.startswith(('control_', 'generation_')):
                continue
            if job.startswith('control_'):
                suffix = '_confirmation' if job.endswith('_confirmation') else '_screen'
                recipe = job[len('control_'):-len(suffix)]
                if recipe not in self.plan['recipes']:
                    continue
                program, _ = compile_recipe(self.repo, self.plan, recipe)
                if data['result']['result']['effective_signature'] != program_signature(program):
                    raise ValueError('Reused candidate numerical signature changed: ' + recipe)
            self.state['jobs'].setdefault(job, copy.deepcopy(data))
        for key, value in previous['program_results'].items():
            self.state['program_results'].setdefault(key, copy.deepcopy(value))
        manifest = {'source_study': source.relative_to(self.repo).as_posix(),
                    'source_state_sha256': digest(source / 'run_status.json'),
                    'files': source_files, 'cohorts': len(previous['program_results']),
                    'old_role_failures_reused': False, 'new_FLOWR_runs_for_identical_programs': 0,
                    'reuse_only': self.config.get('reuse_only', False), 'current_generator_drift': drift,
                    'generation_provenance': 'Original pinned cohorts retain their original generator binding; '
                                             'current source drift is permitted only when new inference is prohibited'}
        if self.config.get('reuse_audit'):
            manifest['reuse_audit_sha256'] = self.config['reuse_audit_sha256']
        write_json(self.study / 'reuse_manifest.json', manifest)

    def infer(self, program_path, signature, arms='gradient', lane='screen', *, reference_path=None, batches=None):
        if self.config.get('reuse_only'):
            if reference_path is None or batches is None:
                raise ValueError('Reuse-only temporal inference requires explicit reference and batches')
            key = signature + ':' + arms + ':' + lane
            mode = self.config.get('prior_seed_mode', 'legacy_stream')
            if mode != 'legacy_stream':
                key += ':' + mode
            key += ':' + ','.join(map(str, batches)) + ':' + digest(reference_path)
            if key not in self.state['program_results']:
                raise ValueError('Reuse-only evidence recovery prohibits new FLOWR inference: ' + key)
        return super().infer(program_path, signature, arms, lane, reference_path=reference_path, batches=batches)

    def remote_command(self, spec):
        c = self.config
        env = ('source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n'
               'export LD_LIBRARY_PATH=/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}\n'
               'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1\n'
               'export PYTHONPATH=' + shlex.quote(c['remote_repo'] + '/src:' + c['flowr'] + ':'
                    + c['flowr'] + '/experiments/evomolsteer_online_20261004/runtime_deps') + '\n')
        args = [c['remote_python'], '-u', c['remote_repo'] + '/scripts/run_temporal_guidance_cohort.py',
                '--repo', c['remote_repo'], '--flowr-root', c['flowr'], '--work', c['remote_work'],
                '--spec', c['remote_repo'] + '/' + spec]
        return 'set -euo pipefail\n' + env + 'cd ' + shlex.quote(c['remote_repo']) + '\n' + shlex.join(args)

    def delivery(self, root):
        result = super().delivery(root)
        target = self.study / 'transport_records' / (root.name + '.temporal_delivery')
        audit_delivery(sorted(root.glob('gradient/batch_*/guidance_trace.jsonl')),
                       read_json(root / 'reward_program.json'), target, self.plan['steps'])
        result['temporal_audit'] = target.relative_to(self.repo).as_posix()
        return result

    def attempt(self, job, action):
        old = self.state['jobs'].get(job)
        if old and old['status'] in {'complete', 'skipped_error'}:
            return old.get('result')
        self.state['jobs'][job] = {'status': 'running', 'started_unix': time.time()}
        self.save(status='running', task=job)
        try:
            result = action()
            self.state['jobs'][job].update(status='complete', result=result, completed_unix=time.time())
            self.save(status='running', task=job)
            return result
        except Exception as error:
            failure = {'job': job, 'error_type': type(error).__name__, 'error': str(error),
                       'traceback': traceback.format_exc(),
                       'stopped_unix': time.time(), 'fallback_not_counted_as_success': True}
            self.state['failures'].append(failure)
            self.state['jobs'][job].update(status='skipped_error', failure=failure)
            write_json(self.study / 'failures' / (job + '.json'), failure)
            self.save(status='running', task=job)
            return None

    def measure(self, program, reference, signature, batches, label):
        bank = read_json(self.repo / self.plan['control_manifest'])
        if signature == bank['effective_signature']:
            return {'reuse_M2': True, 'notice': '本次 Agent 输出未改变执行函数', 'new_FLOWR_runs': 0}
        # Load the checkpoint once per four-batch cohort. Confirmation reuses
        # the first cohort and adds only the missing 200 attempts. Grouping
        # does not change indexed-batch priors, noise or the per-batch size.
        pieces = []
        groups = [self.plan['screen_batches'],
                  [b for b in self.plan['confirmation_batches'] if b not in self.plan['screen_batches']]]
        for cohort in groups:
            group = [b for b in cohort if b in batches]
            if not group:
                continue
            key = signature + ':batches:' + ','.join(map(str, group))
            piece = self.attempt('generation_' + signature[:12] + '_b' + '-'.join(map(str, group)),
                lambda group=group: self.infer(program, signature, batches=group,
                                              lane='temporal', reference_path=reference))
            if piece is None:
                raise RuntimeError('Incomplete cohort after a skipped batch: ' + key)
            pieces.append(piece)
        frames = [pd.read_csv(self.repo / p['outcomes']['gradient']['metrics']) for p in pieces]
        target = self.study / 'transport_records' / (signature[:16] + '_' + label + '.metrics.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(target, index=False)
        result = {'name': label, 'effective_signature': signature,
                  'outcomes': {'gradient': {'metrics': target.relative_to(self.repo).as_posix()}},
                  'initial_state_signatures': {k: v for p in pieces for k, v in p['initial_state_signatures'].items()}}
        report = compare_result(self.repo, self.plan, result, batches)
        write_json(self.study / 'comparisons' / (signature[:16] + '_' + label + '.json'), report)
        return {'result': result, 'comparison': report}

    def control(self, recipe, batches, label):
        p, ref = compile_recipe(self.repo, self.plan, recipe)
        path = self.study / 'compiled_controls' / recipe / 'reward_program.json'
        write_json(path, p)
        return self.measure(path, self.repo / ref, program_signature(p), batches, label)

    def role_condition(self, condition):
        selection = run_roles(self.repo, self.plan, condition, execute=True)
        self.commit('experiment(temporal): bind real API choice ' + condition)
        chosen = selection['selected']
        measurement = self.measure(self.repo / chosen['program'], self.repo / chosen['reference'],
                                   chosen['signature'], self.plan['screen_batches'], 'screen')
        return {'selection': selection, 'measurement': measurement}

    def run(self):
        sources(self.repo, self.plan)
        bank = read_json(self.repo / self.plan['control_manifest'])
        checked = dict(bank['files'])
        if not self.config.get('reuse_only'):
            checked.update(bank['generator_sources'])
        for name, sha in checked.items():
            if digest(self.repo / name) != sha:
                raise ValueError('Frozen comparison input changed: ' + name)
        prepare(self.repo, self.plan)
        with self.ssh() as client:
            shell(client, 'mkdir -p ' + shlex.quote(self.config['remote_work']))
            free = int(shell(client, 'df -Pk ' + shlex.quote(self.config['remote_work']) + " | awk 'NR==2 {print $4}'").strip())
            if free < 614400:
                raise RuntimeError('Less than 600 MiB private workspace available')
        self.save(status='running', error_policy='skip_each_failed_condition_and_continue',
                  controls_generated_again=False, backend='real_api', model='deepseek-flash')
        for phase in [1, 2, 3]:
            conditions = [c for c in self.plan['conditions'] if c['phase'] == phase]
            self.save(phase=phase)
            recipes = list(dict.fromkeys(r for c in conditions for r in c['recipes']))
            screens = {}
            for recipe in recipes:
                screens[recipe] = self.attempt('control_' + recipe + '_screen',
                    lambda recipe=recipe: self.control(recipe, self.plan['screen_batches'], 'screen'))
            # Choices remain frozen and do not receive current quality outcomes.
            for condition in conditions:
                self.attempt('roles_' + condition['id'], lambda condition=condition: self.role_condition(condition['id']))
            for recipe, result in screens.items():
                if result and result.get('comparison', {}).get('eligible_for_400'):
                    self.attempt('control_' + recipe + '_confirmation',
                        lambda recipe=recipe: self.control(recipe, self.plan['confirmation_batches'], 'confirmation'))
            self.attempt('publish_phase_' + str(phase), lambda phase=phase:
                self.commit('report(temporal): finish sequential phase ' + str(phase) + ' with isolated failure records'))
        self.save(status='complete_with_skipped_errors' if self.state['failures'] else 'complete',
                  phase='finished', task=None)
        self.attempt('publish_finished', lambda: self.commit('report(temporal): complete temporal exploration queue'))
        self.save(status='complete_with_skipped_errors' if self.state['failures'] else 'complete', task=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    study = TemporalStudy(repo, read_json(repo / args.config))
    try:
        study.run()
    except Exception as error:
        study.save(status='blocked_preflight_or_coordinator', error_type=type(error).__name__, error=str(error))
        raise
