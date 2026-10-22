"""Complete session coverage gate for every version transition, not approval by file existence."""
from pathlib import Path
import subprocess
from .io import read, write, sha, repository, commit, value_sha
from .group_run import SKILLS


def review_pack(root, version):
    root = Path(root); base = root/'versions'/version
    c = read(root/'config.json')
    items = []
    for teacher in read(root/'teachers.json'):
        name = teacher['teacher_id']; tb = base/name
        folders = [(0, 'bootstrap', tb/'bootstrap')]
        for r in range(1, c['rounds']+1):
            council = tb/f'round_{r:02d}'/'council'
            inventory = read(council/'MASTER_REVIEW_INPUTS.json')
            for path, digest in inventory['files_sha256'].items():
                if sha(council/path) != digest:
                    raise ValueError('Council evidence checksum changed')
            folders += [(r, f'G{g+1}', council/'analysts'/f'G{g+1}') for g in range(4)]
            folders.append((r, 'synthesis', council))
        for r, role, folder in folders:
            files = {}
            for p in folder.rglob('*'):
                if not p.is_file() or (role == 'synthesis' and 'analysts' in p.relative_to(folder).parts):
                    continue
                files[p.relative_to(root).as_posix()] = sha(p)
            identifier = f'{version}/{name}/r{r:02d}/{role}'
            items.append({'id': identifier, 'teacher': name, 'round': r, 'role': role,
                          'files_sha256': files, 'evidence_sha256': value_sha(files)})
    expected = len(read(root/'teachers.json'))*(1+5*c['rounds'])
    if len(items) != expected:
        raise ValueError('Missing logical sessions')
    shards = []
    for i in range(0, len(items), 5):
        path = base/'review'/f'shard_{i//5:03d}.json'
        write(path, {'sessions': items[i:i+5]})
        shards.append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path),
                       'session_ids': [x['id'] for x in items[i:i+5]]})
    manifest = {'version': version, 'config_sha256': sha(root/'config.json'),
                'binding_sha256': sha(base/'binding.json'), 'expected_sessions': expected,
                'expected_sampled_results': len(read(root/'teachers.json'))*c['rounds']*4*c.get('population_size',100),
                'shards': shards, 'requires_human_agent_review': True}
    write(base/'review/manifest.json', manifest)
    return manifest


def committed_file(path, digest, revision):
    relative = Path(path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Review artifact must be repository-relative')
    full = repository()/relative
    if sha(full) != digest:
        raise ValueError('Review artifact checksum mismatch')
    data = subprocess.check_output(['git', '-C', str(repository()), 'show', f'{revision}:{relative.as_posix()}'])
    import hashlib
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Review artifact is not committed in deployed revision')
    return full


def review_check(root, next_version):
    root = Path(root); config = read(root/'config.json')
    if next_version not in config['versions'] or next_version == 'v1':
        raise ValueError('Invalid reviewed-version transition')
    previous = f'v{int(next_version[1:])-1}'
    previous_root = Path(config['previous_protocol_root']) if next_version == config['versions'][0] and config.get('previous_protocol_root') else root
    finalized = previous_root/f'FINALIZED_{previous}.json'
    manifest_path = previous_root/'versions'/previous/'review/manifest.json'
    manifest = read(manifest_path)
    approval = read(root/'versions'/next_version/'review_approval.json')
    checks = {'previous_version': previous, 'next_version': next_version,
              'finalized_sha256': sha(finalized), 'manifest_sha256': sha(manifest_path), 'review_commit': commit()}
    if previous_root.resolve() != root.resolve():
        checks.update(previous_protocol_root=str(previous_root),
            new_prepared_sha256=sha(root/'PREPARED.json'),
            new_acceptance_sha256=sha(root/'ACCEPTANCE_COMPLETE.json'),
            objective_priority=config['agent_objective']['priority'])
    if any(approval.get(k) != v for k, v in checks.items()):
        raise ValueError('Review approval does not bind actual previous results and deployed commit')
    if read(finalized)['review_manifest_sha256'] != sha(manifest_path):
        raise ValueError('Finalized review manifest changed')
    for role, path in SKILLS.items():
        committed_file(path, approval['skill_sha256'][role], commit())
    committed_file(approval['review_document'], approval['review_document_sha256'], commit())
    expected, actual = {}, {}
    for shard in manifest['shards']:
        path = previous_root/shard['path']
        if sha(path) != shard['sha256']:
            raise ValueError('Review shard changed')
        for session in read(path)['sessions']:
            expected[session['id']] = session
            for name, digest in session['files_sha256'].items():
                if sha(previous_root/name) != digest:
                    raise ValueError('Actual prompt/tool evidence changed')
    for ref in approval['coverage']:
        path = committed_file(ref['path'], ref['sha256'], commit())
        for row in read(path)['sessions']:
            identifier = row['id']
            if identifier not in expected or identifier in actual:
                raise ValueError('Unexpected or duplicate reviewed session')
            source = expected[identifier]
            if row.get('evidence_sha256') != source['evidence_sha256']:
                raise ValueError('Review does not bind actual session evidence')
            for key in ['assessment', 'decision_update', 'counterexample_or_limitation']:
                if not isinstance(row.get(key), str) or len(row[key].strip()) < 30:
                    raise ValueError('Substantive individual review is required')
            citations = row.get('evidence_files', [])
            if not citations or not set(citations) <= set(source['files_sha256']):
                raise ValueError('Review cites unknown evidence')
            if not any('input' in p or 'request' in p for p in citations):
                raise ValueError('Review must cite actual prompt evidence')
            actual[identifier] = row
    if set(actual) != set(expected) or len(actual) != manifest['expected_sessions']:
        raise ValueError('Review coverage incomplete')
    return {'approved': True, 'previous': previous, 'next': next_version, 'sessions': len(actual)}
