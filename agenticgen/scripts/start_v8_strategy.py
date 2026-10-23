"""Remote V8 transition after fixed installation; no connection or credentials here."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from agenticgen.adaptive_schedule.io import read, write, sha, commit, value_sha, repository
from agenticgen.adaptive_schedule.group_run import SKILLS, assert_prepared
from agenticgen.adaptive_schedule.group_review import committed_file, review_check
from agenticgen.adaptive_schedule.protocol_state import protocol_dir, prepared_path, acceptance_path
from agenticgen.adaptive_schedule.search_policy import load_version_policy
from agenticgen.adaptive_schedule.strategy_review import AUTHORIZATION, MODE
from agenticgen.runtime.extend_budget import assert_committed_runtime
from agenticgen.runtime.launch import idle_locks
from agenticgen.runtime.restart_failed_version import no_worker
from agenticgen.runtime.review_policy import stage, finish, verify_ready

RESULT_ROOT = Path('/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022')
REVIEW = 'agenticgen/docs/experiments/v7_strategy_review/'


def passing_tests(path):
    suites = [x for x in ET.parse(path).getroot().iter('testsuite')]
    if not suites or sum(int(x.get('tests', '0')) for x in suites) == 0 or any(
            int(x.get(k, '0')) for x in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Remote tests must pass without dependency skips before adoption')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = RESULT_ROOT; repo = repository(); revision = commit()
    if os.name != 'posix' or revision != args.commit:
        raise ValueError('Use the exact committed, installed server runtime')
    assert_committed_runtime()
    base = root/'versions/v8'
    with idle_locks(root):
        no_worker(root)
        if base.exists():
            raise ValueError('V8 directory already exists; inspect receipts, never repeat transition/launch')
    dep = Path('/opt/MolSteer/operations')/('agenticgen_deploy_'+revision[:8])
    dep.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(AGENTICGEN_REPOSITORY=str(repo), AGENTICGEN_PYTHON=sys.executable)

    def run(command, name):
        with (dep/name).open('wb') as log:
            status = subprocess.run(command, cwd='/tmp', env=env, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError('Operation failed; inspect '+str(dep/name))

    xml = dep/'unit-tests.xml'
    run([sys.executable, '-m', 'pytest', '-q', str(repo/'agenticgen/tests'), '--junitxml', str(xml)], 'tests.log')
    passing_tests(xml)
    write(dep/'tests_receipt.json', {'commit':revision, 'junit_sha256':sha(xml), 'passed':True})
    prepared = read(prepared_path(root))
    if prepared.get('strategy_review_operation'):
        operation = root/prepared['strategy_review_operation']
        if read(operation/'journal.json')['commit'] != revision:
            raise ValueError('A different adoption is staged; inspect before proceeding')
    else:
        operation = Path(stage(root, 'v7', repo/AUTHORIZATION)['operation'])
    journal = read(operation/'journal.json')
    if journal['phase'] == 'staged':
        run(['bash', str(repo/'agenticgen/scripts/run.sh'), 'validate', '--root', str(root)], 'native_acceptance.log')
        finish(root, operation)
    verify_ready(root, 'v8')
    with idle_locks(root):
        no_worker(root); config = assert_prepared(root)
        if base.exists():
            raise ValueError('V8 appeared during acceptance; do not overwrite')
        for name in ('review.zh-CN.md', 'strategy_review.json'):
            committed_file(REVIEW+name, sha(repo/REVIEW/name), revision)
        policy = load_version_policy(root, 'v8')
        approval = {'mode':MODE, 'reviewer':'GPT master agent', 'review_completed':True,
            'full_review_completed':False, 'coverage':[], 'selected_strategy_contexts':4,
            'previous_version':'v7', 'next_version':'v8', 'review_commit':revision, 'time':time.time(),
            'finalized_sha256':sha(root/'FINALIZED_v7.json'),
            'manifest_sha256':sha(root/'versions/v7/review/manifest.json'),
            'review_document':REVIEW+'review.zh-CN.md', 'review_document_sha256':sha(repo/REVIEW/'review.zh-CN.md'),
            'strategy_review':REVIEW+'strategy_review.json', 'strategy_review_sha256':sha(repo/REVIEW/'strategy_review.json'),
            'authorization_document':AUTHORIZATION, 'authorization_document_sha256':sha(repo/AUTHORIZATION),
            'skill_sha256':{role:sha(repo/path) for role,path in SKILLS.items()},
            'search_policy_sha256':value_sha(policy),
            'condition_comparison_sha256':value_sha(config['condition_comparison']),
            'new_prepared_sha256':sha(prepared_path(root)), 'new_acceptance_sha256':sha(acceptance_path(root)),
            'review_policy_operation':operation.relative_to(root).as_posix(),
            'scientific_settings_unchanged':True, 'offline_llm_replay_performed':False,
            'performance_improvement_claimed':False, 'aborted_full_review_required':False}
        write(base/'search_policy.json', policy); write(base/'review_approval.json', approval)
        gate = review_check(root, 'v8')
        if not gate['approved'] or gate['mode'] != MODE or gate['sessions'] != 4:
            raise ValueError('Unexpected strategy gate result')
        write(dep/'approval_receipt.json', {'commit':revision, 'gate':gate,
                                         'approval_sha256':sha(base/'review_approval.json')})
    run(['bash', str(repo/'agenticgen/scripts/run.sh'), 'launch', '--root', str(root), '--version', 'v8'], 'launch.log')
    launch = read(root/'launch.json')
    if launch['version'] != 'v8' or launch['commit'] != revision:
        raise ValueError('Launch receipt differs; inspect actual worker')
    receipt = {k:launch[k] for k in ('version','commit','pid','time')}
    receipt['environment'] = 'full API/proxy inherited; values omitted'
    write(dep/'launch_receipt.json', receipt)
    print('V8 launch receipt saved; inspect actual pipeline before claiming any completed round.')


if __name__ == '__main__':
    main()
