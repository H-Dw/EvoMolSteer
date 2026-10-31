"""Read completed science and native provider receipts; archive only closed files."""
import fcntl
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path('/workspace/agenticgen_guidance_selected_retest_20261031')
OUT = ROOT / 'monitoring/20261101_0118'
OUT.mkdir(parents=True, exist_ok=True)


def read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def lock(path):
    if not path.exists():
        return 'absent'
    with path.open('r+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 'held'
        fcntl.flock(stream, fcntl.LOCK_UN)
        return 'free'


def lengths(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {'reason', 'improvement_reason', 'uncertainty'} and isinstance(child, str):
                yield len(child)
            else:
                yield from lengths(child)
    elif isinstance(value, list):
        for child in value:
            yield from lengths(child)


plan = read(ROOT / 'PLAN.json')
repo = Path(plan['repo'])
issues, jobs, selected = [], [], {}
old_final = Path('/workspace/agenticgen_guidance_comparison5_20261031/FINALIZED.json')
assert sha(old_final) == '9c6fd372b65f1ee05bd9828e41cdb8e48c7b9ff6d8000ff1c2da58c52e83a781'
for job in plan['jobs']:
    config = read(job['config'])
    root = Path(config['root'])
    versions = root / 'versions/GD01'
    attempt = read(root / 'attempts.json', {})
    roles, request_digests, http_attempts, completion_tokens = [], [], 0, 0
    for receipt_path in sorted(versions.glob('**/*.receipt.json')):
        receipt = read(receipt_path)
        if receipt.get('final_response_mode') != 'strict_tool_v1':
            issues.append({'receipt': str(receipt_path), 'error': 'non-strict final mode'})
            continue
        role = receipt_path.name.split('.')[0]
        answer_path = receipt_path.with_name(role + '.json')
        answer = read(answer_path)
        session = receipt_path.with_name(role + '.session')
        binding = read(session / 'BINDING.json')
        spec = read(receipt_path.with_name(role + '.prompt_source.json'))
        schema = spec['response_schema']
        assert set(schema['required']).issubset(answer)
        assert set(answer).issubset(schema['properties'])
        control = 'selected' if role == 'synthesis' else 'proposals'
        assert isinstance(answer[control], list)
        assert max(lengths(answer), default=0) <= 3000
        if control == 'selected':
            ids = answer[control]
            prop = schema['properties'][control]
            assert len(ids) == len(set(ids)) and len(ids) <= prop.get('maxItems', 3)
            assert all(i in prop['items'].get('enum', ids) for i in ids)
        if answer['decision'] in {'hold', 'no_future_execution'}:
            assert not answer[control]
        else:
            assert answer[control]
        assert sha(answer_path) == receipt['response_sha256']
        assert sha(session / 'BINDING.json') == receipt['binding_sha256']
        assert binding['final_response_transport']['mode'] == 'strict_tool_v1'
        assert binding['final_response_transport']['endpoint'] == 'https://api.deepseek.com/beta/chat/completions'
        assert receipt['final_tool'] == 'submit_final_response'
        assert receipt['complete_object_explicitly_submitted'] is True
        for name, digest in receipt['transcript_files'].items():
            assert sha(session / name) == digest
        request_count = response_count = accepted = 0
        for path in sorted(session.glob('step_*/REQUEST.json')):
            req = read(path)
            assert req['model'] == 'deepseek-flash' and req['max_tokens'] == 16384
            assert req['thinking'] == {'type': 'enabled'} and req['reasoning_effort'] == 'high'
            assert req['tool_choice'] == 'auto'
            assert req['tools'] and all(t['function'].get('strict') is True for t in req['tools'])
            final = [t['function'] for t in req['tools'] if t['function']['name'] == 'submit_final_response']
            assert len(final) == 1 and set(final[0]['parameters']['required']) == set(schema['required'])
            request_digests.append(sha(path))
            request_count += 1
        for path in sorted(session.glob('step_*/RESPONSE.json')):
            response = read(path)
            response_count += 1
            completion_tokens += response['usage']['completion_tokens']
        for path in sorted(session.glob('step_*/TOOL_*.json')):
            tool = read(path)
            if tool.get('result', {}).get('status') == 'final_submission_accepted':
                assert tool['tool_call_id'] == receipt['final_tool_call_id']
                assert json.loads(tool['call']['function']['arguments']) == answer
                assert tool['result']['response_sha256'] == sha(answer_path)
                accepted += 1
        assert accepted == 1
        http_attempts += len(list(session.glob('step_*/HTTP_*.json')))
        contexts = []
        for message in spec['messages']:
            if message['role'] == 'user':
                try:
                    context = json.loads(message['content'])
                except (ValueError, TypeError):
                    continue
                if isinstance(context, dict) and 'checkpoint_learning' in context:
                    contexts.append(context['checkpoint_learning'])
        roles.append({'path':str(receipt_path.relative_to(root)), 'role':role,
                      'requests':request_count, 'responses':response_count,
                      'explicit_control':answer[control], 'decision':answer['decision'],
                      'maximum_text_characters':max(lengths(answer), default=0),
                      'format_repairs':receipt['interface_repairs'],
                      'checkpoint_learning_in_actual_prompt':contexts})
    budgets, claims = [], []
    for path in sorted(versions.glob('**/POLICY.json')):
        policy = read(path)
        if 'max_model_calls_per_round' not in policy:
            continue
        assert policy['max_model_calls_per_round'] == 30 and policy['max_output_tokens_per_round'] == 491520
        rows = [read(p) for p in sorted(path.parent.glob('call_*.json'))]
        charged = sum(r.get('completion_tokens', r['reserved_output_tokens']) for r in rows)
        assert len(rows) <= 30 and charged <= 491520
        claims.extend(rows)
        budgets.append({'path':str(path.relative_to(root)), 'calls':len(rows), 'charged_output_tokens':charged})
    if roles:
        assert len(claims) == http_attempts
        assert Counter(c['request_sha256'] for c in claims) == Counter(request_digests)
    qualities = []
    for path in sorted((root / 'quality_queue/requests').glob('*.json')):
        req = read(path)
        folder = Path(req['folder'])
        assert folder.is_relative_to(root) and (folder / 'COMPLETE.json').exists()
        assert sha(folder / 'records.json') == req['binding']['records']
        assert sha(folder / 'molecules_all_built.sdf') == req['binding']['sdf']
        assert sha(req['teacher']['protein']) == req['binding']['protein']
        receipt = read(folder / 'QUALITY_COMPLETE.json')
        failure = read(folder / 'QUALITY_FAILED.json')
        if receipt:
            assert (receipt['attempts'], receipt['checked_attempts'], receipt['not_selected'], receipt['coverage']) == (50, 10, 40, .2)
            assert [(r['logical_seed'], r['logical_slot']) for r in receipt['selection']] == [(s,k) for s in [42,100045,200048,300051,400054] for k in [0,1]]
            assert sha(folder / 'quality.json') == receipt['quality_sha256']
            for name, digest in receipt['quality_implementation_sha256'].items():
                assert sha(repo / 'agenticgen/src/agenticgen/uniguide' / name) == digest
        if failure:
            issues.append({'quality_source':str(folder), 'failure':failure})
        qualities.append({'source':str(folder.relative_to(root)), 'complete':receipt is not None, 'failed':failure,
                          'receipt':receipt})
    worker = read(root / 'quality_queue/worker.json')
    worker_process = None
    if worker:
        assert worker['module_directory'] == str(repo / 'agenticgen/src/agenticgen/uniguide')
        for name, digest in worker['implementation_sha256'].items():
            assert sha(Path(worker['module_directory']) / name) == digest
        proc = Path('/proc') / str(worker['pid'])
        if (proc / 'cmdline').exists():
            command = (proc / 'cmdline').read_bytes().replace(b'\0',b' ').decode()
            environment = dict(x.split('=',1) for x in (proc / 'environ').read_bytes().decode().split('\0') if '=' in x)
            worker_process = {'command':command, 'cwd':str((proc / 'cwd').resolve()),
                              'numerics_and_device':{k:environment.get(k) for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
            assert str(root / 'quality_queue/runtime.json') in command
            assert worker_process['cwd'] == str(repo)
    round_paths = sorted(versions.glob('*/advice_*/ROUND_COMPLETE.json'))
    rounds = [{'path':str(p.relative_to(root)), 'checkpoint':read(p).get('checkpoint'),
               'checkpoint_learning':read(p).get('checkpoint_learning')} for p in round_paths]
    jobs.append({'id':job['id'], 'roles':roles, 'actual_http_attempts':http_attempts,
                 'actual_http_responses':sum(r['responses'] for r in roles),
                 'observed_completion_tokens':completion_tokens, 'budget_claims':len(claims),
                 'budgets':budgets, 'rounds':rounds,
                 'attempts':{s:sum(r['count'] for r in attempt.values() if r['stage']==s) for s in ['calibration','formal']},
                 'qualities':qualities, 'worker_process':worker_process,
                 'quality_status':read(root / 'quality_queue/status.json'),
                 'locks':{str(p.relative_to(root)):lock(p) for p in [root / '.executor.lock',root / '.campaign.lock',root / 'quality_queue/.worker.lock']}})
    closed = bool(read(root / 'SAMPLING_FINISHED.json')) and lock(root / '.executor.lock') == 'free'
    if closed:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink() or path.name.startswith('core') or 'launcher.log' in path.name or path.suffix == '.lock':
                continue
            rel = path.relative_to(root)
            if 'quality_queue' in rel.parts and path.name in {'status.json','worker.json'}:
                continue
            if path.suffix == '.log':
                continue
            selected[path.relative_to(ROOT).as_posix()] = path
        for path in sorted((ROOT / 'logs' / job['id']).glob('*')):
            if path.is_file() and 'launcher' not in path.name and path.suffix in {'.log','.json'}:
                selected[path.relative_to(ROOT).as_posix()] = path

mirror_root = Path(plan['durable_snapshot_root'])
mirror_state = read(mirror_root / 'MIRROR_STATE.json', {})
mirror_errors, mirror_lag = [], []
for name, row in mirror_state.items():
    destination = mirror_root / name
    source = ROOT / name
    if not destination.exists() or sha(destination) != row['sha256']:
        mirror_errors.append(name)
    if source.exists() and sha(source) != row['sha256']:
        mirror_lag.append(name)
report = {'timestamp':stamp(),'timezone':'Asia/Shanghai','plan_sha256':sha(ROOT/'PLAN.json'),
          'old_finalized_sha256':sha(old_final),'jobs':jobs,'issues':issues,
          'mirror':{'files':len(mirror_state),'copy_hash_errors':mirror_errors,'source_advanced_since_snapshot':mirror_lag,
                    'durable_status':read(ROOT/'DURABLE_SNAPSHOT.json')},
          'disk':[{'path':p,'free_bytes':shutil.disk_usage(p).free} for p in ['/root/private_data','/workspace']],
          'new_scientific_attempts_by_audit':0,'new_api_calls_by_audit':0}
write('runtime_audit.json',report)
assert not issues and not mirror_errors
for path in [ROOT/'PLAN.json', ROOT/'STATUS.json', ROOT/'HEARTBEAT.json', ROOT/'DURABLE_SNAPSHOT.json', OUT/'runtime_audit.json']:
    if path.exists():
        # Snapshot active suite metadata to this audit folder before archiving it.
        target = OUT / ('snapshot_' + path.name)
        shutil.copy2(path, target)
        selected[target.relative_to(ROOT).as_posix()] = target
snapshots = sorted((ROOT/'monitoring').glob('2026-11-01T*.json'))
if snapshots:
    selected[snapshots[-1].relative_to(ROOT).as_posix()] = snapshots[-1]
manifest = {}
archive = OUT / 'closed_artifacts.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
    for name,path in sorted(selected.items()):
        before = path.stat()
        digest = sha(path)
        tar.add(path,arcname=name,recursive=False)
        after = path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and sha(path)==digest, name
        manifest[name]={'sha256':digest,'bytes':before.st_size}
write('archive_manifest.json',{'timestamp':stamp(),'files':manifest,'scope':'closed first teacher; live second teacher excluded; safe snapshot metadata',
                               'full_hdf5_in_closed_job':True,'launcher_or_core_included':False})
complete={'timestamp':stamp(),'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
          'manifest_sha256':sha(OUT/'archive_manifest.json'),'files':len(manifest),'hdf5':sum(n.endswith('.h5') for n in manifest)}
write('backup_COMPLETE.json',complete)
print(json.dumps({'timestamp':report['timestamp'],'jobs':[{'id':j['id'],'rounds':len(j['rounds']),'attempts':j['attempts'],
                  'roles':len(j['roles']),'responses':j['actual_http_responses'],'claims':j['budget_claims'],
                  'quality_complete':sum(q['complete'] for q in j['qualities']),'quality_sources':len(j['qualities'])} for j in jobs],
                  'issues':issues,'mirror_errors':mirror_errors,'mirror_lag_count':len(mirror_lag),
                  'disk':report['disk'],'backup':complete},ensure_ascii=True),flush=True)
