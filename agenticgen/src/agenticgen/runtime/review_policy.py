"""Adopt the user-directed best-round gate at an idle completed-version boundary."""
from pathlib import Path

from agenticgen.adaptive_schedule.io import read, write, sha, commit, source_fingerprint
from agenticgen.adaptive_schedule.protocol_state import protocol_dir, acceptance_path
from agenticgen.adaptive_schedule.strategy_review import authorization, MODE, performance_context
from agenticgen.runtime.extend_budget import committed, assert_committed_runtime
from agenticgen.runtime.launch import idle_locks
from agenticgen.runtime.restart_failed_version import no_worker, verify_protected
from agenticgen.runtime.restart_failed_teacher import native_ready

ALLOWED_SOURCE_CHANGES = {'adaptive_schedule/group_review.py', 'adaptive_schedule/strategy_review.py',
                          'runtime/review_policy.py', 'cli.py'}


def stage(root, previous, authorization_path):
    root = Path(root).resolve()
    with idle_locks(root):
        no_worker(root); assert_committed_runtime()
        ref = committed(authorization_path); authorization(ref['path'], ref['sha256'])
        next_version = f'v{int(previous[1:])+1}'
        if previous != 'v7':
            raise ValueError('Initial best-round adoption is limited to completed V7')
        old_dir = protocol_dir(root); old = read(old_dir/'PREPARED.json')
        config = read(old_dir/'config.json'); accepted = read(old_dir/'ACCEPTANCE_COMPLETE.json')
        if old.get('master_review_policy'):
            raise ValueError('Review policy already adopted; do not repeat migration')
        if (config['versions'] != [f'v{i}' for i in range(6,31)]
                or old['config_sha256'] != sha(old_dir/'config.json')
                or old['teachers_sha256'] != sha(root/'teachers.json')
                or accepted['config_sha256'] != old['config_sha256']
                or accepted['source_fingerprint'] != old['source_fingerprint']):
            raise ValueError('Existing frozen protocol/acceptance changed')
        current = source_fingerprint()
        changes = {k for k in current.keys() | old['source_fingerprint'].keys()
                   if current.get(k) != old['source_fingerprint'].get(k)}
        if not changes or not changes <= ALLOWED_SOURCE_CHANGES:
            raise ValueError('Review adoption may not change scientific execution code')
        if {p.parent.name for p in (root/'versions').glob('*/binding.json')} != {'v6','v7'}:
            raise ValueError('Review adoption needs the completed V6/V7 boundary without a future run')
        protected = {}
        for version in ('v6','v7'):
            base = root/'versions'/version; final = root/f'FINALIZED_{version}.json'
            f = read(final); manifest = base/'review/manifest.json'; state = read(base/'pipeline.json')
            if (f.get('outcomes') != 1000 or f.get('rounds') != 5
                    or f['report_sha256'] != sha(base/'report.json')
                    or f['review_manifest_sha256'] != sha(manifest)
                    or read(manifest)['binding_sha256'] != sha(base/'binding.json')
                    or len(state['teachers']) != 2
                    or any(t['phase'] != 'complete' or t['round'] != 5 or t.get('error') for t in state['teachers'].values())):
                raise ValueError('Prior version is incomplete or its finalization changed')
            for p in (final, manifest, base/'report.json', base/'binding.json', base/'pipeline.json'):
                protected[p.relative_to(root).as_posix()] = sha(p)
            for teacher in state['teachers']:
                for r in range(1,6):
                    folder = base/teacher/f'round_{r:02d}'
                    for p in [folder/'wave'/n for n in ('COMPLETE.json','QUALITY_COMPLETE.json','CLEANUP_COMPLETE.json')] + [
                            folder/'council/COMPLETE.json', base/teacher/f'RETENTION_COMPLETE_{r:02d}.json']:
                        protected[p.relative_to(root).as_posix()] = sha(p)
        for p in [root/n for n in ('config.json','PREPARED.json','ACCEPTANCE_COMPLETE.json','teachers.json','historical_references.json')]:
            protected[p.relative_to(root).as_posix()] = sha(p)
        for p in (root/'protocols').rglob('*'):
            if p.is_file(): protected[p.relative_to(root).as_posix()] = sha(p)
        for teacher, digest in old.get('shared_r0_sha256', {}).items():
            p = root/'shared_r0'/teacher/'COMPLETE.json'
            if sha(p) != digest: raise ValueError('Shared R0 changed')
            protected[p.relative_to(root).as_posix()] = digest
        # Performance selection only; no full-session review is invoked here.
        selection = performance_context(root, previous)['teachers']
        op = root/'operations'/f'best_round_review_after_{previous}_{commit()[:8]}'
        folder = root/'protocols'/op.name
        if op.exists() or folder.exists():
            raise ValueError('Review adoption already staged; inspect existing transaction')
        journal = {'phase':'staged', 'commit':commit(), 'previous_version':previous, 'next_version':next_version,
                   'authorization':ref, 'new_protocol':folder.relative_to(root).as_posix(),
                   'old_pointer':read(root/'ACTIVE_PROTOCOL.json'), 'protected_sha256':protected,
                   'source_changes':sorted(changes), 'config_changes':[], 'best_rounds':selection,
                   'full_review_required':False, 'launch_approved':False}
        write(op/'journal.json', journal); write(folder/'config.json', config)
        p = {**old, 'commit':commit(), 'source_fingerprint':current,
             'config_sha256':sha(folder/'config.json'), 'previous_protocol':old_dir.relative_to(root).as_posix(),
             'previous_protocol_prepared_sha256':sha(old_dir/'PREPARED.json'),
             'previous_protocol_acceptance_sha256':sha(old_dir/'ACCEPTANCE_COMPLETE.json'),
             'master_review_policy':{'mode':MODE, 'authorization':ref, 'full_review_prohibited':True},
             'strategy_review_operation':op.relative_to(root).as_posix(),
             'native_probe_directory':f'operations/best_round_review_acceptance_{commit()[:8]}',
             'requires_native_acceptance':True, 'r0_resampling_allowed':False}
        for key in ('teacher_restart_operation','budget_extension_operation','failed_rerun_operation'):
            if key in p: p['previous_'+key] = p.pop(key)
        write(folder/'PREPARED.json', p); verify_protected(root, protected)
        write(root/'ACTIVE_PROTOCOL.json', {'directory':folder.relative_to(root).as_posix(),
                                           'prepared_sha256':sha(folder/'PREPARED.json')})
        return {'phase':'staged','operation':str(op),'next_version':next_version,'launch_approved':False}


def verify_ready(root, version):
    root = Path(root).resolve(); folder = protocol_dir(root); p = read(folder/'PREPARED.json')
    op = (root/p['strategy_review_operation']).resolve()
    if op.parent != root/'operations' or op.is_symlink(): raise ValueError('Unsafe review operation')
    j = read(op/'journal.json')
    if (j['phase'] != 'accepted_for_review_check' or j['commit'] != p['commit']
            or root/j['new_protocol'] != folder or version not in read(folder/'config.json')['versions']
            or int(version[1:]) < int(j['next_version'][1:])
            or p['master_review_policy'].get('authorization') != j['authorization']
            or j['acceptance_sha256'] != sha(acceptance_path(root))):
        raise ValueError('Review policy/native acceptance is incomplete')
    native_ready(root); verify_protected(root, j['protected_sha256'])
    return j


def finish(root, operation):
    root = Path(root).resolve(); op = Path(operation).resolve()
    if op.parent != root/'operations' or op.is_symlink(): raise ValueError('Unsafe review operation')
    with idle_locks(root):
        no_worker(root); assert_committed_runtime(); j = read(op/'journal.json')
        if (j['phase'] not in ('staged','accepted_for_review_check') or j['commit'] != commit()
                or root/j['new_protocol'] != protocol_dir(root)):
            raise ValueError('Use staged fixed runtime and protocol')
        native_ready(root); verify_protected(root, j['protected_sha256'])
        j.update(phase='accepted_for_review_check', acceptance_sha256=sha(acceptance_path(root)))
        write(op/'journal.json', j); verify_ready(root, j['next_version'])
        return {'phase':j['phase'],'next_version':j['next_version'],'launch_approved':False}
