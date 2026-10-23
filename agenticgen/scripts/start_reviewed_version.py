"""Start one reviewed version under the already accepted best-round protocol.

No protocol migration, sampling recovery, secret handling or remote transport.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from agenticgen.adaptive_schedule.io import read, write, sha, commit, value_sha, repository
from agenticgen.adaptive_schedule.group_run import SKILLS, assert_prepared
from agenticgen.adaptive_schedule.group_review import committed_file, review_check
from agenticgen.adaptive_schedule.protocol_state import prepared_path, acceptance_path
from agenticgen.adaptive_schedule.search_policy import load_version_policy
from agenticgen.adaptive_schedule.strategy_review import AUTHORIZATION, MODE
from agenticgen.adaptive_schedule.version_gate import version_metrics
from agenticgen.adaptive_schedule.selection import rollback_advice
from agenticgen.runtime.extend_budget import assert_committed_runtime
from agenticgen.runtime.launch import idle_locks
from agenticgen.runtime.restart_failed_version import no_worker
from agenticgen.runtime.review_policy import verify_ready as verify_review_policy
from agenticgen.runtime.restart_failed_teacher import verify_ready as verify_teacher_restart

RESULT_ROOT = Path('/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022')


def verify_active_protocol(root, version):
    """Follow the accepted current protocol, including completed teacher repairs.

    A repair preserves the adopted review scope but archives its old operation
    pointer. Its own native/protected-source gate is the active readiness check.
    This dispatch performs no migration, deletion or recovery.
    """
    prepared = read(prepared_path(root))
    policy = prepared.get('master_review_policy', {})
    if policy.get('mode') != MODE or policy.get('full_review_prohibited') is not True:
        raise ValueError('The accepted best-round review scope is required')
    routes = {'strategy_review_operation': verify_review_policy,
              'teacher_restart_operation': verify_teacher_restart}
    active = [name for name in routes if prepared.get(name)]
    if len(active) != 1:
        raise ValueError('Expected one supported active protocol operation')
    name = active[0]
    if routes[name](root, version) is None:
        raise ValueError('Active protocol verifier returned no accepted transaction')
    return {'kind': name, 'operation': prepared[name]}


def previous_version(root, version):
    if not re.fullmatch(r'v(?:9|[12][0-9]|30)', version):
        raise ValueError('Only subsequent reviewed versions V9..V30 are supported')
    if (root/'versions'/version).exists() or (root/f'FINALIZED_{version}.json').exists():
        raise ValueError('Version already exists; inspect receipts instead of repeating approval or launch')
    previous = f'v{int(version[1:])-1}'
    base = root/'versions'/previous
    final = read(root/f'FINALIZED_{previous}.json')
    if (final.get('outcomes') != 1000 or final.get('rounds') != 5
            or final['report_sha256'] != sha(base/'report.json')
            or final['review_manifest_sha256'] != sha(base/'review/manifest.json')):
        raise ValueError('Previous finalized results must be complete and unchanged')
    return previous


def passing_tests(path):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    if not suites or not sum(int(x.get('tests', '0')) for x in suites) or any(
            int(x.get(k, '0')) for x in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Remote tests must pass without skips')


def reviewed_control_state(root, version, review, revision):
    """Bind an authored rollback to immutable history before staging a new version."""
    advice = rollback_advice(version_metrics(root))
    path = repository()/review/'rollback_plan.json'
    if not advice.get('required'):
        if path.exists():
            raise ValueError('Rollback plan present without two completed regressions')
        return load_version_policy(root, version), None
    plan = read(committed_file(review+'rollback_plan.json', sha(path), revision))
    restore = advice['restore_version']; old = root/'versions'/restore
    binding = read(old/'binding.json'); restored_policy = load_version_policy(root, restore)
    if (plan.get('previous_version') != f'v{int(version[1:])-1}'
            or plan.get('next_version') != version or plan.get('restored_version') != restore
            or plan.get('restored_binding_sha256') != sha(old/'binding.json')
            or plan.get('source_binding_sha256') != value_sha(binding['source_fingerprint'])
            or plan.get('restored_policy') != restored_policy):
        raise ValueError('Authored rollback plan differs from historical-best binding/policy')
    for role, skill in SKILLS.items():
        committed_file(skill, binding['skill_hashes'][role], revision)
    document = review+'rollback_compatibility.zh-CN.md'
    digest = sha(repository()/document)
    if (plan.get('code_compatibility_document') != document
            or plan.get('code_compatibility_sha256') != digest):
        raise ValueError('Rollback compatibility document differs')
    committed_file(document, digest, revision)
    return restored_policy, {
        'restored_version':restore, 'restored_binding_sha256':sha(old/'binding.json'),
        'source_binding_sha256':value_sha(binding['source_fingerprint']),
        'code_compatibility_document':document, 'code_compatibility_sha256':digest,
        'plan_document':review+'rollback_plan.json', 'plan_document_sha256':sha(path)}


def validation_tests(changed):
    """Select only changed deployment interfaces; unknown code needs an explicit plan."""
    selected = set()
    mapping = {
        'agenticgen/scripts/start_reviewed_version.py': 'test_reviewed_start.py',
        'agenticgen/scripts/deploy_reviewed_version.py': 'test_reviewed_deploy.py',
        'agenticgen/scripts/fetch_fixed_github_api.py': 'test_github_object_fetch.py',
    }
    for path in changed:
        if not path.startswith('agenticgen/'):
            continue  # Unrelated repository work is not installed as agenticgen.
        if path.startswith('agenticgen/docs/') or path.endswith('/SKILL.md'):
            continue
        if path in mapping:
            selected.add('agenticgen/tests/unit/'+mapping[path])
            if path == 'agenticgen/scripts/start_reviewed_version.py':
                selected.add('agenticgen/tests/unit/test_control_gates.py')
        elif path in {'agenticgen/tests/unit/'+name for name in mapping.values()}:
            selected.add(path)
        else:
            raise ValueError('Executable change needs a scoped remote validation plan: '+path)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = RESULT_ROOT; repo = repository(); revision = commit()
    if os.name != 'posix' or revision != args.commit or not re.fullmatch('[0-9a-f]{40}', args.commit):
        raise ValueError('Use the exact committed, installed server runtime')
    assert_committed_runtime()
    with idle_locks(root):
        no_worker(root)
        previous = previous_version(root, args.version)
        verify_active_protocol(root, args.version)
        assert_prepared(root)  # Skills/scripts may change; accepted scientific Python must match.
        review = f'agenticgen/docs/experiments/{previous}_strategy_review/'
        reviewed_control_state(root, args.version, review, revision)
    review = f'agenticgen/docs/experiments/{previous}_strategy_review/'
    dep = Path('/opt/MolSteer/operations')/('agenticgen_deploy_'+revision[:8])
    dep.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(AGENTICGEN_REPOSITORY=str(repo), AGENTICGEN_PYTHON=sys.executable)

    def run(command, name):
        with (dep/name).open('wb') as log:
            result = subprocess.run(command, cwd='/tmp', env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Operation failed; inspect '+str(dep/name))

    previous_commit = read(root/'launch.json')['commit']
    if not re.fullmatch('[0-9a-f]{40}', previous_commit):
        raise ValueError('Previous scientific source is not a fixed commit')
    changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only',
                                      previous_commit, revision, '--', 'agenticgen/'], text=True).splitlines()
    tests = validation_tests(changed)
    validation = {'commit':revision, 'compared_to':previous_commit, 'changed_files':changed,
                  'test_files':tests, 'local_tests_run':False, 'full_suite_run':False}
    if tests:
        xml = dep/'related-tests.xml'
        run([sys.executable, '-m', 'pytest', '-q', *[str(repo/p) for p in tests],
             '--junitxml', str(xml)], 'related-tests.log')
        passing_tests(xml)
        validation.update(passed=True, junit_sha256=sha(xml))
    else:
        validation.update(passed=None, status='not_run_skills_documents_only',
                          rationale='Strategy effectiveness is evaluated by the next real experiment.')
    write(dep/'tests_receipt.json', validation)
    with idle_locks(root):
        no_worker(root); previous_version(root, args.version)
        config = assert_prepared(root)
        active_protocol = verify_active_protocol(root, args.version)
        for name in ('review.zh-CN.md', 'strategy_review.json'):
            committed_file(review+name, sha(repo/review/name), revision)
        policy, rollback = reviewed_control_state(root, args.version, review, revision)
        approval = {'mode':MODE, 'reviewer':'GPT master agent', 'review_completed':True,
            'full_review_completed':False, 'coverage':[], 'selected_strategy_contexts':4,
            'previous_version':previous, 'next_version':args.version, 'review_commit':revision, 'time':time.time(),
            'finalized_sha256':sha(root/f'FINALIZED_{previous}.json'),
            'manifest_sha256':sha(root/'versions'/previous/'review/manifest.json'),
            'review_document':review+'review.zh-CN.md', 'review_document_sha256':sha(repo/review/'review.zh-CN.md'),
            'strategy_review':review+'strategy_review.json', 'strategy_review_sha256':sha(repo/review/'strategy_review.json'),
            'authorization_document':AUTHORIZATION, 'authorization_document_sha256':sha(repo/AUTHORIZATION),
            'skill_sha256':{role:sha(repo/path) for role,path in SKILLS.items()},
            'search_policy_sha256':value_sha(policy),
            'condition_comparison_sha256':value_sha(config['condition_comparison']),
            'new_prepared_sha256':sha(prepared_path(root)), 'new_acceptance_sha256':sha(acceptance_path(root)),
            'active_protocol_gate':active_protocol,
            'scientific_settings_unchanged':True, 'offline_llm_replay_performed':False,
            'performance_improvement_claimed':False, 'aborted_full_review_required':False}
        if rollback is not None:
            approval['rollback'] = rollback
        base = root/'versions'/args.version
        write(base/'search_policy.json', policy); write(base/'review_approval.json', approval)
        gate = review_check(root, args.version)
        if not gate['approved'] or gate['mode'] != MODE or gate['sessions'] != 4:
            raise ValueError('Unexpected strategy gate result')
        write(dep/'approval_receipt.json', {'commit':revision, 'gate':gate,
                                         'approval_sha256':sha(base/'review_approval.json'),
                                         'search_policy':policy, 'rollback':rollback})
    run(['bash', str(repo/'agenticgen/scripts/run.sh'), 'launch', '--root', str(root), '--version', args.version], 'launch.log')
    launch = read(root/'launch.json')
    if launch['version'] != args.version or launch['commit'] != revision:
        raise ValueError('Launch receipt differs; inspect actual worker')
    receipt = {k:launch[k] for k in ('version','commit','pid','time')}
    receipt['environment'] = 'full API/proxy inherited; values omitted'
    write(dep/'launch_receipt.json', receipt)
    print(f'{args.version} launch receipt saved; sampling completion remains unconfirmed.')


if __name__ == '__main__':
    main()
