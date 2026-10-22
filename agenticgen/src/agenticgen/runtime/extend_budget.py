"""Append the authorized V30 budget at a fully reviewed, idle version boundary.

No old result, binding or acceptance is edited. Staging requires new native
acceptance; finishing records compatibility, never a scientific launch approval.
"""
import subprocess
from pathlib import Path

from agenticgen.adaptive_schedule.io import read, write, sha, commit, repository, source_fingerprint
from agenticgen.adaptive_schedule.protocol_state import protocol_dir, acceptance_path
from agenticgen.runtime.launch import idle_locks
from agenticgen.runtime.restart_failed_version import no_worker, verify_protected, inventory

ALLOWED_SOURCE_CHANGES = {'adaptive_schedule/group_protocol.py', 'adaptive_schedule/group_review.py',
                          'runtime/extend_budget.py', 'cli.py'}


def committed(path):
    from agenticgen.adaptive_schedule.group_review import committed_file
    p = Path(path).resolve(); repo = repository().resolve()
    if not p.is_relative_to(repo):
        raise ValueError('Boundary artifacts must be committed in the deployed repository')
    relative = p.relative_to(repo).as_posix()
    committed_file(relative, sha(p), commit())
    return {'path': relative, 'sha256': sha(p)}


def assert_committed_runtime():
    repo = repository()
    prefix = 'agenticgen/src/agenticgen/'
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only',
                                     commit(), '--', prefix], text=True).splitlines()
    expected = {n[len(prefix):] for n in names if n.endswith('.py')}
    if expected != set(source_fingerprint()):
        raise ValueError('Runtime contains missing or uncommitted Python files')
    if subprocess.call(['git', '-C', str(repo), 'diff', '--quiet', commit(), '--', prefix]):
        raise ValueError('Runtime or skills differ from the fixed commit')


def check_review(root, previous, progress_path):
    ref = committed(progress_path); p = read(progress_path)
    base = root/'versions'/previous; manifest = read(base/'review/manifest.json')
    if (p.get('version') != previous or p.get('reviewed_sessions') != 52 or
            p.get('expected_sessions') != 52 or manifest.get('expected_sessions') != 52 or
            p.get('finalized_sha256') != sha(root/f'FINALIZED_{previous}.json') or
            p.get('manifest_sha256') != sha(base/'review/manifest.json')):
        raise ValueError('Complete current-version review is required for budget extension')
    expected = {}
    for shard in manifest['shards']:
        path = root/shard['path']
        if sha(path) != shard['sha256']:raise ValueError('Review shard changed')
        for s in read(path)['sessions']:
            if s['id'] in expected:raise ValueError('Duplicate source session')
            expected[s['id']] = s
            for name, digest in s['files_sha256'].items():
                if sha(root/name) != digest:raise ValueError('Reviewed source changed')
    actual = {}
    for item in p['coverage']:
        path = repository()/item['path']
        if committed(path)['sha256'] != item['sha256']:raise ValueError('Coverage changed')
        for row in read(path)['sessions']:
            sid = row['id']; source = expected.get(sid)
            if sid in actual or source is None or row.get('evidence_sha256') != source['evidence_sha256']:
                raise ValueError('Review coverage does not bind source sessions')
            if any(len(row.get(k, '').strip()) < 30 for k in
                   ['assessment', 'decision_update', 'counterexample_or_limitation']):
                raise ValueError('Substantive review missing')
            cites = row.get('evidence_files', [])
            if not cites or not set(cites) <= set(source['files_sha256']) or not any('input' in n or 'request' in n for n in cites):
                raise ValueError('Actual prompt evidence must be cited')
            actual[sid] = row
    if set(actual) != set(expected) or len(actual) != 52:raise ValueError('Review coverage incomplete')
    aborted = {q.relative_to(root).as_posix(): sha(q)
               for folder in (root/'operations').glob(f'failed_version_rerun_{previous}_*/failure_audit')
               for q in folder.rglob('*') if q.is_file()}
    if aborted:
        path = Path(progress_path).parent/'aborted_attempt_coverage.json'
        if (p.get('aborted_attempt_substantive_review_complete') is not True or
                committed(path)['sha256'] != p.get('aborted_coverage_sha256')):
            raise ValueError('Aborted attempt review is incomplete')
        a = read(path); covered = dict(a['control_files_sha256'])
        if not a['contexts']:raise ValueError('Aborted contexts missing')
        for ctx in a['contexts']:
            if ctx.get('substantively_reviewed') is not True or any(len(ctx.get('review', {}).get(k, '').strip()) < 30
                    for k in ['observation', 'interpretation', 'decision_and_counterexample', 'revision']):
                raise ValueError('Substantive aborted context review missing')
            covered.update(ctx['files_sha256'])
        if covered != aborted:raise ValueError('Aborted evidence coverage changed or incomplete')
        ref['aborted_coverage'] = committed(path)
    return ref


def stage(root, previous, authorization, progress_path):
    from agenticgen.adaptive_schedule.group_protocol import configuration
    from agenticgen.adaptive_schedule.wave_retention import verify_snapshot
    root = Path(root).resolve()
    with idle_locks(root):
        no_worker(root); assert_committed_runtime()
        auth = committed(authorization); a = read(authorization)
        required = {'authority': 'human_user', 'first_version': 'v6', 'previous_last_version': 'v25',
                    'last_version': 'v30', 'version_count': 25, 'rounds_per_version': 5,
                    'optimization_results_per_version': 1000, 'shared_r0_results': 50,
                    'maximum_formal_results': 25050, 'preserve_existing_protocol_and_acceptance': True,
                    'full_review_required_for_each_next_version': True}
        if any(a.get(k) != v for k, v in required.items()):raise ValueError('Unsupported budget authorization')
        old_dir = protocol_dir(root); old = read(old_dir/'PREPARED.json')
        prior = read(old_dir/'config.json'); new = configuration()
        if prior['versions'] != [f'v{i}' for i in range(6, 26)] or previous not in prior['versions']:
            raise ValueError('Budget extension requires the original V6–V25 protocol')
        if new['versions'] != [f'v{i}' for i in range(6, 31)] or new['total_result_budget'] != 25050:
            raise ValueError('Runtime does not implement the authorized V30 budget')
        changed_config = {k for k in prior.keys() | new.keys() if prior.get(k) != new.get(k)}
        if changed_config != {'versions', 'total_result_budget'} or prior['total_result_budget'] != 20050:
            raise ValueError('Budget adoption may not change scientific or API settings')
        acceptance = read(old_dir/'ACCEPTANCE_COMPLETE.json')
        if (old['config_sha256'] != sha(old_dir/'config.json') or old['teachers_sha256'] != sha(root/'teachers.json') or
                acceptance['config_sha256'] != old['config_sha256'] or acceptance['source_fingerprint'] != old['source_fingerprint']):
            raise ValueError('Original frozen preparation/acceptance changed')
        if old.get('historical_references_sha256') and old['historical_references_sha256'] != sha(root/'historical_references.json'):
            raise ValueError('Frozen historical reference manifest changed')
        current = source_fingerprint()
        changed_source = sorted(k for k in old['source_fingerprint'].keys() | current.keys()
                                if old['source_fingerprint'].get(k) != current.get(k))
        if not changed_source or not set(changed_source) <= ALLOWED_SOURCE_CHANGES:
            raise ValueError('Budget adoption includes unsupported scientific source changes')
        done = prior['versions'][:prior['versions'].index(previous)+1]
        bound = {p.parent.name for p in (root/'versions').glob('*/binding.json')}
        if bound != set(done):raise ValueError('Version boundary has incomplete, missing or future bindings')
        for version in done:
            base = root/'versions'/version; f = read(root/f'FINALIZED_{version}.json')
            m = read(base/'review/manifest.json'); state = read(base/'pipeline.json')
            if (f.get('version') != version or f.get('rounds') != 5 or f.get('outcomes') != 1000 or
                    f.get('review_manifest_sha256') != sha(base/'review/manifest.json') or
                    f.get('report_sha256') != sha(base/'report.json') or m.get('binding_sha256') != sha(base/'binding.json') or
                    len(state['teachers']) != 2 or any(t['phase'] != 'complete' or t['round'] != 5 or t.get('error') for t in state['teachers'].values())):
                raise ValueError('Version finalization or pipeline is incomplete')
            for teacher in state['teachers']:
                for r in range(1, 6):
                    tb = base/teacher; wave = tb/f'round_{r:02d}'/'wave'
                    markers = [wave/'COMPLETE.json', wave/'QUALITY_COMPLETE.json', wave/'CLEANUP_COMPLETE.json',
                               tb/f'round_{r:02d}'/'council/COMPLETE.json', tb/f'RETENTION_COMPLETE_{r:02d}.json']
                    if not all(p.is_file() for p in markers):raise ValueError('Round barriers incomplete')
                    if read(wave/'COMPLETE.json').get('attempts') != 100 or read(wave/'QUALITY_COMPLETE.json').get('attempts') != 100:
                        raise ValueError('Round sampling/quality count changed')
        review = check_review(root, previous, progress_path)
        binding = read(root/'versions'/previous/'binding.json')
        if binding['config_sha256'] != old['config_sha256'] or binding['source_fingerprint'] != old['source_fingerprint']:
            raise ValueError('Latest completed version does not bind the active protocol')
        for name, digest in read(root/'historical_references.json').items():
            if sha(name) != digest:raise ValueError('Historical input changed')
        r0 = {}
        for t in read(root/'teachers.json'):
            folder = root/'shared_r0'/t['teacher_id']; verify_snapshot(folder)
            r0[t['teacher_id']] = sha(folder/'COMPLETE.json')
        if old.get('shared_r0_sha256') != r0:raise ValueError('R0 binding changed')
        protected = {}
        for name in ['shared_r0', 'baselines', 'protocols', 'versions']:
            protected.update({name+'/'+n: meta['sha256'] for n, meta in inventory(root/name).items()})
        for audit in (root/'operations').glob('failed_version_rerun_*/failure_audit'):
            protected.update({audit.relative_to(root).as_posix()+'/'+n: meta['sha256'] for n, meta in inventory(audit).items()})
        for p in list(root.glob('FINALIZED_*.json')) + [root/n for n in
                ['config.json', 'PREPARED.json', 'ACCEPTANCE_COMPLETE.json', 'teachers.json', 'historical_references.json']]:
            protected[p.relative_to(root).as_posix()] = sha(p)
        revision = commit(); next_version = f'v{int(previous[1:])+1}'
        op = root/'operations'/f'budget_v30_after_{previous}_{revision[:8]}'
        folder = root/'protocols'/f'budget_v30_after_{previous}_{revision[:8]}'
        if op.exists() or folder.exists():raise ValueError('Budget operation exists; inspect its journal')
        journal = {'phase': 'staged', 'commit': revision, 'previous_version': previous, 'next_version': next_version,
                   'authorization': auth, 'review_progress': review, 'protected_sha256': protected,
                   'previous_protocol': old_dir.relative_to(root).as_posix(),
                   'old_pointer': read(root/'ACTIVE_PROTOCOL.json'), 'new_protocol': folder.relative_to(root).as_posix(),
                   'source_changes': changed_source, 'config_changes': sorted(changed_config),
                   'scientific_launch_approved': False, 'requires_native_acceptance': True}
        write(op/'journal.json', journal); write(folder/'config.json', new)
        prepared = {**old, 'config_sha256': sha(folder/'config.json'), 'source_fingerprint': current,
                    'commit': revision, 'versions': 25, 'total_outcomes_including_shared_r0': 25050,
                    'previous_protocol': old_dir.relative_to(root).as_posix(),
                    'previous_protocol_prepared_sha256': sha(old_dir/'PREPARED.json'),
                    'previous_protocol_acceptance_sha256': sha(old_dir/'ACCEPTANCE_COMPLETE.json'),
                    'budget_extension_operation': op.relative_to(root).as_posix(),
                    'native_probe_directory': f'operations/budget_v30_acceptance_{revision[:8]}',
                    'requires_native_acceptance': True, 'r0_resampling_allowed': False}
        write(folder/'PREPARED.json', prepared)
        verify_protected(root, protected)
        write(root/'ACTIVE_PROTOCOL.json', {'directory': folder.relative_to(root).as_posix(),
                                           'prepared_sha256': sha(folder/'PREPARED.json')})
        return {'phase': 'staged', 'operation': str(op), 'next_version': next_version,
                'requires_native_acceptance': True, 'launch_approved': False}


def verify_ready(root, next_version):
    root = Path(root).resolve(); folder = protocol_dir(root); p = read(folder/'PREPARED.json')
    op = (root/p['budget_extension_operation']).resolve()
    if op.parent != root/'operations' or op.is_symlink():raise ValueError('Unsafe budget operation')
    j = read(op/'journal.json'); a = read(acceptance_path(root))
    if (j['phase'] != 'accepted_for_review_check' or j['commit'] != p['commit'] or
            next_version not in read(folder/'config.json')['versions'] or
            int(next_version[1:]) < int(j['next_version'][1:]) or
            root/j['new_protocol'] != folder or j['acceptance_sha256'] != sha(acceptance_path(root)) or
            p['source_fingerprint'] != source_fingerprint() or
            a['source_fingerprint'] != source_fingerprint() or a['config_sha256'] != sha(folder/'config.json')):
        raise ValueError('Budget compatibility/native acceptance is not complete')
    verify_protected(root, j['protected_sha256'])
    return j


def finish(root, operation):
    root = Path(root).resolve(); op = Path(operation).resolve()
    if op.parent != root/'operations' or op.is_symlink():raise ValueError('Unsafe budget operation')
    with idle_locks(root):
        no_worker(root); assert_committed_runtime()
        j = read(op/'journal.json'); folder = protocol_dir(root)
        if j['phase'] not in {'staged','accepted_for_review_check'}:raise ValueError('Unknown budget operation phase')
        if j['commit'] != commit() or root/j['new_protocol'] != folder:raise ValueError('Use staged fixed runtime and protocol')
        a = read(acceptance_path(root))
        if (a['source_fingerprint'] != source_fingerprint() or a['config_sha256'] != sha(folder/'config.json') or
                a.get('teachers') != 2 or a.get('recovery_checks') != 2 or a.get('common_randomness') is not True or
                a.get('different_hypothesis_schedules') is not True or a.get('affinity_vjp') is not False):
            raise ValueError('Fresh two-teacher native acceptance required')
        prepared = read(folder/'PREPARED.json'); probe = root/prepared['native_probe_directory']
        if (a['results_sha256'] != sha(probe/'results.json') or
                a['gradient_checks_sha256'] != sha(folder/'validation/gradient_checks.json') or
                read(folder/'validation/gradient_checks.json').get('passed') is not True):
            raise ValueError('Native acceptance evidence changed')
        verify_protected(root, j['protected_sha256'])
        j.update(phase='accepted_for_review_check', acceptance_sha256=sha(acceptance_path(root)))
        write(op/'journal.json', j)
        verify_ready(root, j['next_version'])
        return {'phase': j['phase'], 'next_version': j['next_version'], 'launch_approved': False,
                'preserved_files': len(j['protected_sha256'])}
