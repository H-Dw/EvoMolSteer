"""Direct tests of the local reader's scope and completion semantics."""
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('remote_status_reader',
    Path(__file__).parents[2] / 'scripts/read_remote_status.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadOnlySFTP:
    def __init__(self, files):
        self.files = files
        self.opens = []
        self.redirects = {}
        self.entries = []
        self.folders = {}
        self.denied = set()

    def normalize(self, path):
        return self.redirects.get(path, path)

    def open(self, path, mode):
        assert mode == 'rb'
        self.opens.append(path)
        if path in self.denied:
            raise PermissionError(errno.EACCES, 'permission denied')
        if path not in self.files:
            raise FileNotFoundError(errno.ENOENT, 'missing')
        value = self.files[path]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        return io.BytesIO(value)

    def listdir_attr(self, path):
        if path in self.folders:
            return self.folders[path]
        assert path == module.ROOT + '/operations'
        return self.entries

    def exec_command(self, *args, **kwargs):
        raise AssertionError('Remote command execution is forbidden')

    def put(self, *args, **kwargs):
        raise AssertionError('Uploads are forbidden')


def encoded(value):
    return json.dumps(value).encode()


@pytest.fixture
def complete():
    root = module.ROOT
    files = {root + '/' + name: encoded({}) for name in module.permitted_files('v10')}
    files[root + '/launch.json'] = encoded({'version': 'v10', 'commit': 'a' * 40, 'pid': 123, 'time': 1000})
    files[root + '/versions/v10/pipeline.json'] = encoded({'teachers': {
        teacher: {'round': 5, 'phase': 'complete'} for teacher in module.TEACHERS}})
    files[root + '/versions/v10/PIPELINE_COMPLETE.json'] = encoded({
        'teacher_rounds': {teacher: 5 for teacher in module.TEACHERS}})
    for path in files:
        if path.endswith('/wave/COMPLETE.json') or path.endswith('/wave/QUALITY_COMPLETE.json'):
            files[path] = encoded({'attempts': 100})
    files[root + '/FINALIZED_v10.json'] = encoded({'version': 'v10', 'rounds': 5, 'outcomes': 1000,
        'report_sha256': hashlib.sha256(files[root + '/versions/v10/report.json']).hexdigest(),
        'review_manifest_sha256': hashlib.sha256(files[root + '/versions/v10/review/manifest.json']).hexdigest()})
    return ReadOnlySFTP(files)


def test_finalized_checks_both_teachers_five_rounds_and_hashes(complete):
    result = module.collect(complete, 'v10')
    assert result['state'] == 'finalized'
    assert result['completion_verified']
    assert not result['process']['worker_alive']
    assert not result['remote_execution'] and not result['remote_writes']
    assert all('/environ' not in path and not path.endswith('.log') for path in complete.opens)


@pytest.mark.parametrize('broken', ['quality', 'cleanup', 'retention', 'council', 'report', 'pipeline'])
def test_incomplete_or_changed_evidence_cannot_claim_completion(complete, broken):
    root = module.ROOT
    prefix = root + '/versions/v10/' + module.TEACHERS[0]
    paths = {'quality': prefix + '/round_05/wave/QUALITY_COMPLETE.json',
             'cleanup': prefix + '/round_05/wave/CLEANUP_COMPLETE.json',
             'retention': prefix + '/RETENTION_COMPLETE_05.json',
             'council': prefix + '/round_05/council/COMPLETE.json',
             'report': root + '/versions/v10/report.json',
             'pipeline': root + '/versions/v10/PIPELINE_COMPLETE.json'}
    del complete.files[paths[broken]]
    result = module.collect(complete, 'v10')
    assert result['state'] == 'inconsistent_completion'
    assert not result['completion_verified']


def test_startup_without_pipeline_is_running_not_complete():
    root = module.ROOT
    sftp = ReadOnlySFTP({root + '/launch.json': encoded({'version': 'v10', 'commit': 'a' * 40, 'pid': 123, 'time': 1000}),
        '/proc/123/cmdline': ('python\0-u\0-m\0agenticgen\0run\0--root\0' + root + '\0--version\0v10\0').encode(),
        '/proc/123/status': b'Name: python\nState:\tS (sleeping)\n'})
    result = module.collect(sftp, 'v10')
    assert result['state'] == 'running' and result['process']['command_matches']
    assert not result['completion_verified']
    sftp.files['/proc/123/cmdline'] = b'python\0unrelated.py\0'
    assert module.collect(sftp, 'v10')['state'] == 'needs_inspection'


def test_old_failures_are_not_current_failures(complete):
    complete.entries = [SimpleNamespace(filename='failure_1.json', st_mode=stat.S_IFREG, st_mtime=900),
                        SimpleNamespace(filename='failure_2.json', st_mode=stat.S_IFREG, st_mtime=1100)]
    result = module.collect(complete, 'v10')
    assert result['new_failure_receipts'] == [{'name': 'failure_2.json', 'mtime': 1100}]
    assert result['state'] == 'finalized'  # A receipt alone does not negate completion.


def test_permission_denial_is_not_reported_as_worker_exit(complete):
    complete.denied.add('/proc/123/cmdline')
    with pytest.raises(PermissionError):
        module.collect(complete, 'v10')


def test_only_current_failed_round_gets_targeted_diagnostics(complete):
    root = module.ROOT
    path = root + '/versions/v10/pipeline.json'
    state = json.loads(complete.files[path])
    teacher = module.TEACHERS[0]
    state['teachers'][teacher] = {'round': 4, 'phase': 'failed',
        'error': {'type': 'RuntimeError', 'message': 'An isolated analyst failed'}}
    complete.files[path] = encoded(state)
    diagnostic = root + '/versions/v10/' + teacher + '/round_04/council/analyst_failures.json'
    complete.files[diagnostic] = encoded({'G3': {'error_type': 'ValueError', 'message': 'Invalid proposal'}})
    result = module.collect(complete, 'v10')
    assert result['teachers'][teacher]['failed_analysts']['G3']['message'] == 'Invalid proposal'
    assert [p for p in complete.opens if p.endswith('analyst_failures.json')] == [diagnostic]


def test_error_summary_redacts_credentials_and_urls():
    result = module.error_summary({'type': 'HTTPError', 'message':
        'https://user:secret@example.test auth Bearer secret2 api_key=secret3 password=secret4'})
    assert 'secret' not in result['message']


@pytest.mark.parametrize('version', ['../v10', 'v31', 'V10', 'v010', 'v10;echo nope'])
def test_reject_out_of_scope_arguments(version):
    with pytest.raises(ValueError):
        module.permitted_files(version)


@pytest.mark.parametrize('path', ['../.ssh/id_rsa', 'versions/v10/../../config.json',
                                 'worker_v10.log', '/proc/123/environ', 'config.json'])
def test_reader_does_not_accept_arbitrary_paths(complete, path):
    with pytest.raises(ValueError):
        module.StatusReader(complete, 'v10').data(path)
    assert complete.opens == []


def test_symlink_escape_and_oversized_sources_fail_closed(complete):
    source = module.ROOT + '/launch.json'
    complete.redirects[source] = module.ROOT + '/config.json'
    with pytest.raises(ValueError):
        module.StatusReader(complete, 'v10').data('launch.json')
    complete.redirects.clear()
    complete.files[source] = b'x' * (module.MAX_BYTES + 1)
    with pytest.raises(ValueError):
        module.StatusReader(complete, 'v10').data('launch.json')


def test_launch_change_during_read_requires_another_snapshot(complete):
    path = module.ROOT + '/launch.json'
    complete.files[path] = [complete.files[path], encoded({'version': 'v11'})]
    result = module.collect(complete, 'v10')
    assert result['state'] == 'snapshot_changed' and not result['completion_verified']


def test_wrong_active_version_and_untrusted_pid_rejected(complete):
    path = module.ROOT + '/launch.json'
    launch = json.loads(complete.files[path])
    launch['version'] = 'v11'
    complete.files[path] = encoded(launch)
    with pytest.raises(ValueError):
        module.collect(complete, 'v10')
    launch.update(version='v10', pid='../environ')
    complete.files[path] = encoded(launch)
    with pytest.raises(ValueError):
        module.collect(complete, 'v10')


@pytest.fixture
def incomplete(complete):
    root = module.ROOT
    teacher = module.TEACHERS[0]
    state = {'teachers': {teacher: {'round': 4, 'phase': 'failed'},
                         module.TEACHERS[1]: {'round': 5, 'phase': 'complete'}}}
    complete.files[root + '/versions/v10/pipeline.json'] = encoded(state)
    del complete.files[root + '/FINALIZED_v10.json']
    council = root + '/versions/v10/' + teacher + '/round_04/council'
    complete.files[council + '/analyst_failures.json'] = encoded({
        'G2': {'error_type': 'ValueError', 'message': 'Incomplete JSON answer; no incumbent masquerading as success'}})
    folder = council + '/analysts/G2'
    content = b'{"learning_update":'
    request = encoded({'body': {'model': 'deepseek-flash', 'max_tokens': 65536,
                               'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}})
    receipt = encoded({'finish_reason': 'length', 'content_sha256': hashlib.sha256(content).hexdigest(),
                       'request_sha256': hashlib.sha256(request).hexdigest(), 'max_output_tokens': 65536,
                       'provider_request_id': 'test-id', 'usage': {'completion_tokens': 65536}})
    complete.files[folder + '/turn_02.incomplete_1.receipt.json'] = receipt
    complete.files[folder + '/turn_02.incomplete_1.txt'] = content
    complete.files[folder + '/turn_02.request.json'] = request
    complete.folders[folder] = [SimpleNamespace(filename=name, st_mode=stat.S_IFREG)
        for name in ('turn_02.incomplete_1.receipt.json', 'turn_02.request.json',
                     '../turn_02.incomplete_1.receipt.json', 'turn_02.raw_1.txt')]
    return complete, folder


def test_incomplete_diagnosis_is_opt_in_and_bound_to_failed_analyst(incomplete):
    sftp, folder = incomplete
    module.collect(sftp, 'v10')
    assert not any('/analysts/' in name for name in sftp.opens)
    result = module.collect(sftp, 'v10', incomplete_diagnostics=True)
    answers = result['teachers'][module.TEACHERS[0]]['incomplete_answers']['G2']
    assert len(answers) == 1 and answers[0]['receipt']['finish_reason'] == 'length'
    assert answers[0]['request_settings']['max_tokens'] == 65536
    assert answers[0]['content'] == '{"learning_update":'
    assert {name for name in sftp.opens if '/analysts/' in name} == {
        folder + '/turn_02.incomplete_1.receipt.json', folder + '/turn_02.incomplete_1.txt',
        folder + '/turn_02.request.json'}
    assert 'incomplete_answers' not in result['teachers'][module.TEACHERS[1]]


def test_invalid_answer_diagnosis_scopes_by_error_type_and_checks_hash(incomplete):
    sftp,folder=incomplete
    council=folder.rsplit('/analysts/',1)[0]
    sftp.files[council+'/analyst_failures.json']=encoded({'G2':{'error_type':'LLMInterfaceError','message':'invalid reference'}})
    name='turn_01.attempt_2.invalid.receipt.json'; content=b'{"evidence_ids":["wrong"]}'
    sftp.folders[folder]=[SimpleNamespace(filename=name,st_mode=stat.S_IFREG)]
    sftp.files[folder+'/'+name]=encoded({'content_sha256':hashlib.sha256(content).hexdigest(),'finish_reason':'stop'})
    sftp.files[folder+'/turn_01.attempt_2.invalid.txt']=content
    result=module.collect(sftp,'v10',interface_diagnostics=True)
    assert result['teachers'][module.TEACHERS[0]]['invalid_answers']['G2'][0]['content']==content.decode()
    assert not any('round_03/council/analysts/' in p for p in sftp.opens)
    sftp.files[folder+'/turn_01.attempt_2.invalid.txt']=b'changed'
    with pytest.raises(ValueError):module.collect(sftp,'v10',interface_diagnostics=True)


@pytest.mark.parametrize('problem', ['content_hash', 'request_hash', 'budget', 'redirect', 'count', 'live'])
def test_incomplete_diagnosis_fails_closed(incomplete, problem):
    sftp, folder = incomplete
    if problem == 'content_hash':
        sftp.files[folder + '/turn_02.incomplete_1.txt'] = b'changed'
    elif problem == 'request_hash':
        sftp.files[folder + '/turn_02.request.json'] = b'changed'
    elif problem == 'budget':
        name = folder + '/turn_02.incomplete_1.receipt.json'
        value = json.loads(sftp.files[name]); value['max_output_tokens'] = 32768
        sftp.files[name] = encoded(value)
    elif problem == 'redirect':
        sftp.redirects[folder] = folder.replace('G2', 'G1')
    elif problem == 'count':
        sftp.folders[folder] = [SimpleNamespace(filename=f'turn_02.incomplete_{i}.receipt.json',
                                              st_mode=stat.S_IFREG) for i in range(1, 18)]
    else:
        sftp.files['/proc/123/cmdline'] = ('python\0-m\0agenticgen\0run\0--root\0' + module.ROOT + '\0--version\0v10\0').encode()
        sftp.files['/proc/123/status'] = b'State:\tS (sleeping)\n'
    with pytest.raises(ValueError):
        module.collect(sftp, 'v10', incomplete_diagnostics=True)
