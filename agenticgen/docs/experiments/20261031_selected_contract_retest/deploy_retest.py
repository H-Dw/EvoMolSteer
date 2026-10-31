"""Deploy only the two newly authorized reruns, preserving the finalized suite."""

import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET

OLD = Path('/workspace/agenticgen_guidance_comparison5_20261031')
ROOT = Path('/workspace/agenticgen_guidance_selected_retest_20261031')
STORAGE = Path('/root/private_data/uniguide_selected_retest_20261031')
DOC = 'agenticgen/docs/experiments/20261031_selected_contract_retest'
JOBS = ['03_prefix_joint_6pyy', '06_five_windows_3w54']
TESTS = (
    'strict_final_submission', 'guidance_comparison5', 'comparison8_contracts',
    'prefix_promotion_recovery', 'dynamic_matching_jump', 'linear_guidance',
    'adaptive_calibration', 'geometry_tensor_contract', 'geometry_chemical_state',
    'uniguide_geometry', 'uniguide_geometry_batch', 'uniguide_geometry_dynamics',
    'guidance_skill_bundle', 'guidance_skill_release_v2', 'guidance_analysis_tools',
    'deepseek_tools', 'coordinate_statistics', 'absolute_uncapped_statistics',
    'absolute_coordinate_protocol', 'ug04_task_scheduling',
)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2)
        stream.write('\n')


def command(args, cwd=None, timeout=120):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command {args[0]} failed ({result.returncode}): ' + result.stderr[-1200:])
    return result.stdout


def deploy(commit, bundle, bundle_sha):
    if re.fullmatch(r'[0-9a-f]{40}', commit) is None:
        raise ValueError('An exact pushed Git commit is required')
    audit = STORAGE / 'audits' / ('selected-' + commit[:8])
    repo = STORAGE / 'runs' / ('selected-' + commit[:8])
    audit.mkdir(parents=True, exist_ok=True)

    def status(stage, **fields):
        state = dict(timestamp=now(), stage=stage, commit=commit)
        state.update(fields)
        write(audit/'DEPLOY_STATUS.json', state)

    try:
        if ROOT.exists() or repo.exists():
            raise ValueError('New deployment/root already exists; preserve it and inspect its audit')
        if shutil.disk_usage('/root/private_data').free < (1 << 30):
            raise ValueError('Less than 1 GiB of durable space before deployment')
        old_plan = read(OLD/'PLAN.json')
        old_repo = Path(old_plan['repo'])
        if sha(OLD/'FINALIZED.json') != '9c6fd372b65f1ee05bd9828e41cdb8e48c7b9ff6d8000ff1c2da58c52e83a781':
            raise ValueError('The finalized original suite changed')
        if command(['git', '-C', str(old_repo), 'rev-parse', 'HEAD']).strip() != old_plan['commit']:
            raise ValueError('Original fixed checkout changed')
        if command(['git', '-C', str(old_repo), 'status', '--porcelain', '--untracked-files=no']).strip():
            raise ValueError('Original fixed checkout is dirty')

        status('fetching_fixed_pushed_commit')
        repo.mkdir(parents=True)
        command(['git', 'init', '-q', str(repo)])
        command(['git', 'remote', 'add', 'origin', 'https://github.com/H-Dw/EvoMolSteer.git'], cwd=repo)
        alternate = repo/'.git/objects/info/alternates'
        alternate.parent.mkdir(parents=True, exist_ok=True)
        alternate.write_text(str((old_repo/'.git/objects').resolve()) + '\n')
        transports = []
        for prefix in ['', 'https://ghfast.top/', 'https://gh-proxy.com/', 'https://ghproxy.net/', 'https://gh.llkk.cc/']:
            try:
                result = subprocess.run(['git', 'fetch', prefix+'https://github.com/H-Dw/EvoMolSteer.git', commit],
                    cwd=repo, capture_output=True, text=True, timeout=20)
                transports.append(dict(transport=prefix or 'direct', code=result.returncode))
                if result.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                transports.append(dict(transport=prefix or 'direct', timeout=20))
        else:
            if not bundle or sha(bundle) != bundle_sha:
                raise ValueError('All GitHub transports failed; no valid pushed-commit bundle')
            command(['git', 'bundle', 'verify', str(bundle)], cwd=repo)
            command(['git', 'fetch', str(bundle), 'refs/heads/codex/uniguide-flowr-agent'], cwd=repo)
            transports.append(dict(transport='verified_pushed_only_bundle', sha256=bundle_sha))
        directories = command(['git', '-C', str(old_repo), 'sparse-checkout', 'list']).splitlines()
        directories.append(DOC)
        command(['git', 'sparse-checkout', 'init', '--cone'], cwd=repo)
        command(['git', 'sparse-checkout', 'set', *sorted(set(directories))], cwd=repo)
        command(['git', 'checkout', '-B', 'codex/uniguide-flowr-agent', commit], cwd=repo)
        if command(['git', 'rev-parse', 'HEAD'], cwd=repo).strip() != commit:
            raise ValueError('New checkout differs from the pushed commit')
        if sha(repo/DOC/'deploy_retest.py') != sha(__file__):
            raise ValueError('Deployment helper differs from the pushed fixed source')
        write(audit/'DEPLOYMENT.json', dict(timestamp=now(), commit=commit, repo=str(repo),
            transports=transports, read_only_alternate=str(alternate), old_checkout_unchanged=True,
            deployment_helper_sha256=sha(__file__)))

        python = '/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python'
        env = ('source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1\n'
            f'export PYTHONPATH={repo}/agenticgen/src:{repo}/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root\n'
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n')
        cpu_env = env + 'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\nunset DEEPSEEK_API_KEY\n'
        status('remote_cpu_regression')
        tests = [python, '-m', 'pytest', '-q', '-c', 'agenticgen/pyproject.toml',
            *['agenticgen/tests/unit/test_' + name + '.py' for name in TESTS],
            '--basetemp=' + str(audit/'pytest'), '--junitxml=' + str(audit/'regression.xml')]
        with (audit/'regression.log').open('xb') as stream:
            result = subprocess.run(['bash', '-c', cpu_env+shlex.join(tests)], cwd=repo,
                stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        suites = ET.parse(audit/'regression.xml').getroot()
        element = suites.find('testsuite') if suites.tag != 'testsuite' else suites
        regression_counts = {k: int(element.get(k)) for k in ('tests', 'failures', 'errors', 'skipped')}
        write(audit/'REGRESSION.json', dict(timestamp=now(), commit=commit, returncode=result.returncode,
            cpu_only=True, scientific_attempts=0, llm_calls=0, **regression_counts))
        if result.returncode:
            raise ValueError('CPU regression failed; no API or sampling started')

        source_snapshot = Path(old_plan['persistent_storage'])/'audits/comparison5-072e6f78/INPUT_SNAPSHOT.json'
        snapshot = read(source_snapshot)
        for source, item in snapshot['copies'].items():
            if sha(item['destination']) != item['sha256']:
                raise ValueError('Frozen durable input changed: ' + item['destination'])
        old_jobs = {j['id']: j for j in old_plan['jobs']}
        for job_id in JOBS:
            c = read(old_jobs[job_id]['config'])
            if sha(c['checkpoint']) != c['checkpoint_sha256']:
                raise ValueError('Model checkpoint differs from the original frozen input')
            for name, digest in c['native_hashes'].items():
                if sha(Path(c['flowr'])/name) != digest:
                    raise ValueError('Frozen native FLOWR source differs')
        write(audit/'INPUT_VERIFIED.json', dict(timestamp=now(), original_snapshot=str(source_snapshot),
            original_snapshot_sha256=sha(source_snapshot), copies_verified=len(snapshot['copies']),
            reused_frozen_inputs_only=True, imported_scientific_attempts=0))

        status('real_beta_strict_contract_probe')
        probe = [python, str(repo/DOC/'probe_contract.py'), '--source-plan', str(OLD/'PLAN.json'),
                 '--output', str(audit/'engineering_probe')]
        # Interactive startup may expose environment; this log stays remote only.
        with (audit/'api_probe.launcher.log').open('xb') as stream:
            result = subprocess.run(['bash', '-ic', env +
                'export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\n' +
                shlex.join(probe)], cwd=repo, stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode or not (audit/'engineering_probe/ENGINEERING_PROBE_COMPLETE.json').exists():
            raise ValueError('Real strict contract probe failed; preserve claims and inspect without rerunning')
        probe_complete = read(audit/'engineering_probe/ENGINEERING_PROBE_COMPLETE.json')

        status('preserving_original_and_building_two_job_plan')
        ROOT.mkdir(parents=True, exist_ok=False)
        overlay = read(repo/'agenticgen/configs/uniguide_selected_contract_retest.json')
        jobs = []
        for job_id in JOBS:
            original = old_jobs[job_id]
            c = copy.deepcopy(read(original['config']))
            if c.get('postprocessing_recovery'):
                raise ValueError('Fresh reruns cannot import historical candidate measurements')
            c.update(rounds=5, root=str(ROOT/'jobs'/job_id/'attempt_002'),
                agent_tools=overlay['agent_tools'], checkpoint_learning=overlay['checkpoint_learning'])
            c['llm'] = dict(c['llm'], **overlay['llm'], attempts=3, max_format_repairs=1)
            config = Path(str(repo)+'-configs')/(job_id+'.json')
            write(config, c, exclusive=True)
            jobs.append(dict(original, config=str(config), config_sha256=sha(config), root=c['root'],
                commit=commit, skill_release='2.8.2', rounds=5, attempt='attempt_002',
                previous_failed_root=read(original['config'])['root']))

        protected = read(OLD/'SOURCE_PRESERVATION.json')
        for name, digest in protected.items():
            if sha(name) != digest:
                raise ValueError('Historical preservation mismatch: ' + name)
        for path in (OLD/'jobs').rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and not path.name.endswith('.lock'):
                protected[str(path)] = sha(path)
        for name in ['FINALIZED.json', 'PLAN.json', 'SOURCE_PRESERVATION.json', 'ALL_JOBS_ATTEMPTED.json', 'SUITE_FINISHED.json']:
            protected[str(OLD/name)] = sha(OLD/name)
        for relative in command(['git', '-C', str(old_repo), 'ls-files', '-z']).split('\0'):
            if relative and (old_repo/relative).is_file():
                protected[str(old_repo/relative)] = sha(old_repo/relative)
        write(ROOT/'SOURCE_PRESERVATION.json', protected, exclusive=True)
        totals = {'calibration': 0, 'formal': 0}
        historical = []
        for j in old_plan['jobs']:
            folder = Path(read(j['config'])['root'])
            attempts = read(folder/'attempts.json')
            counts = {stage: sum(r['count'] for r in attempts.values() if r['stage']==stage) for stage in totals}
            historical.append(dict(job=j['id'], root=str(folder), counts=counts,
                attempts_sha256=sha(folder/'attempts.json')))
            for stage in totals:
                totals[stage] += counts[stage]
        if totals != {'calibration':1640, 'formal':2150}:
            raise ValueError('Original finalized accounting changed')
        write(ROOT/'HISTORICAL_ACCOUNTING.json', dict(timestamp=now(), previous_total=3790,
            stages=totals, jobs=historical, old_failure_attempts_not_refunded=600,
            new_calibration_limit=700, new_formal_limit=1500, combined_upper_limit=5990))
        module_paths = [*list((repo/'agenticgen/src/agenticgen/uniguide').rglob('*.py')),
                        *list((repo/'agenticgen/src/agenticgen/providers').rglob('*.py'))]
        plan = dict(timestamp=now(), root=str(ROOT), repo=str(repo), commit=commit,
            generation_context_root=old_plan['generation_context_root'], jobs=jobs,
            plan_document_sha256=sha(repo/DOC/'plan.zh-CN.md'), fresh_calibration=True,
            maximum_calibration_attempts=700, maximum_formal_attempts=1500, rounds=5,
            explanation_max_characters=3000, global_executor_lock=old_plan['global_executor_lock'],
            wait_for_global_lock=True, persistent_storage=str(STORAGE),
            durable_snapshot_root=str(STORAGE/'result_snapshots'), trajectory_storage=str(ROOT),
            trajectory_storage_is_overlay=True, source_preservation_count=len(protected),
            runtime_source_sha256={str(p.relative_to(repo)):sha(p) for p in sorted(module_paths)},
            original_plan=str(OLD/'PLAN.json'), original_FINALIZED_sha256=sha(OLD/'FINALIZED.json'),
            engineering_probe_complete=str(audit/'engineering_probe/ENGINEERING_PROBE_COMPLETE.json'),
            engineering_probe_complete_sha256=sha(audit/'engineering_probe/ENGINEERING_PROBE_COMPLETE.json'),
            scientific_attempts_before_retest=3790, failure_policy='preserve_skip_continue; explicit user-authorized new roots only')
        write(ROOT/'PLAN.json', plan, exclusive=True)
        write(audit/'READY.json', dict(timestamp=now(), commit=commit, repo=str(repo), root=str(ROOT),
            plan=str(ROOT/'PLAN.json'), plan_sha256=sha(ROOT/'PLAN.json'), protected_files=len(protected),
            regression=regression_counts, engineering_probe_http_responses=probe_complete['actual_responses'],
            scientific_attempts=0, no_old_measurement_or_budget_imported=True))

        status('launching_authorized_background_suite')
        launch = ['bash', '-i', str(repo/DOC/'launch.sh'), str(repo), str(ROOT/'PLAN.json')]
        with (ROOT/'launcher.log').open('xb') as stream:
            process = subprocess.Popen(launch, cwd=repo, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        launched = dict(timestamp=now(), pid=process.pid, command=launch, commit=commit,
            rounds=5, jobs=2, plan_sha256=sha(ROOT/'PLAN.json'), launcher_log_kept_remote_only=True)
        write(audit/'LAUNCH.json', launched, exclusive=True)
        status('background_launched', **launched)
    except Exception as error:
        write(audit/'DEPLOY_FAILED.json', dict(timestamp=now(), error_type=type(error).__name__,
            error=str(error), preserved=True, automatic_restart=False))
        status('deployment_failed_preserved', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--bundle')
    parser.add_argument('--bundle-sha256')
    args = parser.parse_args()
    deploy(args.commit, args.bundle, args.bundle_sha256)
