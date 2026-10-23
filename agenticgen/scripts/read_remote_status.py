"""Local, bounded SFTP status reader. No remote shell, upload or worker control.

Run with Python + Paramiko on the local machine; do not install into the GPU
runtime. The transport still uses normal Codex tool permissions. A denial must
not trigger a retry using the old unrestricted console or another transport.
"""
import argparse
import base64
import errno
import getpass
import hashlib
import json
from pathlib import Path
import re
import stat
import time


HOST = 'ssh.zzai.scnet.cn'
PORT = 11762
USER = 'root'
HOST_KEY = 'SHA256:AvD7IKadbL87TRyvyx0nrk7cSZE8di7nKSV1l1XiD8Q'
ROOT = '/opt/MolSteer/generated_datasets/agenticgen_5nn4_r12_h4_n25_core1hop_fixedstart_s42_20261022'
TEACHERS = ('5nn4_SC2_A_1017__top1', '5nn4_SC2_A_1017__top2')
MAX_BYTES = 8 * 1024 * 1024


def validate_version(version):
    if not re.fullmatch(r'v(?:[6-9]|[12][0-9]|30)', version):
        raise ValueError('Version must be v6..v30')
    return version


def permitted_files(version):
    validate_version(version)
    base = f'versions/{version}'
    names = {'launch.json', f'FINALIZED_{version}.json',
             f'{base}/pipeline.json', f'{base}/PIPELINE_COMPLETE.json',
             f'{base}/report.json', f'{base}/review/manifest.json'}
    for teacher in TEACHERS:
        for round_no in range(1, 6):
            prefix = f'{base}/{teacher}'
            names.add(f'{prefix}/RETENTION_COMPLETE_{round_no:02d}.json')
            names.add(f'{prefix}/round_{round_no:02d}/council/COMPLETE.json')
            names.add(f'{prefix}/round_{round_no:02d}/council/analyst_failures.json')
            for marker in ('COMPLETE', 'QUALITY_COMPLETE', 'CLEANUP_COMPLETE'):
                names.add(f'{prefix}/round_{round_no:02d}/wave/{marker}.json')
    return names


class StatusReader:
    """Allowlisted, size-bounded reads; the connection object is never exposed."""
    def __init__(self, sftp, version):
        self.sftp = sftp
        self.version = validate_version(version)
        self.allowed = permitted_files(version)
        self.hashes = {}

    def _bytes(self, path, limit, *, proc=False):
        try:
            resolved = self.sftp.normalize(path)
            if proc:
                if resolved != path:
                    raise ValueError('Process path resolved outside the requested file')
            elif resolved != path:
                raise ValueError('Result path resolved away from the permitted source')
            with self.sftp.open(path, 'rb') as handle:
                value = handle.read(limit + 1)
            if len(value) > limit:
                raise ValueError('Status source exceeds the bounded read size')
            return value
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return None
            raise  # Permission/network errors are unknown, not missing files.

    def data(self, relative):
        if relative not in self.allowed:
            raise ValueError('This file is not a permitted status source')
        value = self._bytes(ROOT + '/' + relative, MAX_BYTES)
        if value is not None:
            self.hashes[relative] = hashlib.sha256(value).hexdigest()
        return value

    def json(self, relative):
        value = self.data(relative)
        return json.loads(value) if value is not None else None

    def process(self, launch):
        pid = launch['pid']
        if type(pid) is not int or not 1 <= pid <= 4194304:
            raise ValueError('Invalid PID in launch receipt')
        command = self._bytes(f'/proc/{pid}/cmdline', 16384, proc=True)
        status = self._bytes(f'/proc/{pid}/status', 16384, proc=True)
        args = command.decode('utf-8', 'replace').split('\0') if command else []
        expected = ['-m', 'agenticgen', 'run', '--root', ROOT, '--version', self.version]
        matches = any(args[i:i + len(expected)] == expected for i in range(len(args)))
        state = next((line.split(':', 1)[1].strip() for line in
                      (status or b'').decode('utf-8', 'replace').splitlines()
                      if line.startswith('State:')), None)
        return {'pid': pid, 'exists': command is not None, 'command_matches': matches,
                'state': state, 'worker_alive': matches and state is not None and not state.startswith('Z')}

    def failures(self, launched):
        # CLI writes failure receipts in operations/, not at the result root.
        # Names/mtime only, no recursive scan and no reading failure payloads.
        rows = []
        for entry in self.sftp.listdir_attr(ROOT + '/operations'):
            if (re.fullmatch(r'failure_[0-9]+\.json', entry.filename)
                    and stat.S_ISREG(entry.st_mode) and entry.st_mtime >= int(launched)):
                rows.append({'name': entry.filename, 'mtime': entry.st_mtime})
        return sorted(rows, key=lambda row: row['mtime'])


def error_summary(error):
    if not isinstance(error, dict):
        return None
    message = str(error.get('message', ''))
    message = re.sub(r'https?://\S+', '[URL omitted]', message)
    message = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|password|token)\s*[:=]\s*)\S+',
                     '[credential omitted]', message)
    return {'type': str(error.get('type', error.get('error_type', 'unknown')))[:100],
            'message': message[:1200]}


def collect(sftp, version):
    reader = StatusReader(sftp, version)
    launch_bytes = reader.data('launch.json')
    launch = json.loads(launch_bytes) if launch_bytes else None
    if not launch or launch.get('version') != version:
        raise ValueError('Requested version differs from the active launch receipt')
    if not re.fullmatch(r'[0-9a-f]{40}', launch.get('commit', '')):
        raise ValueError('Invalid fixed commit in launch receipt')
    base = f'versions/{version}'
    pipeline = reader.json(f'{base}/pipeline.json')
    teacher_states = (pipeline or {}).get('teachers', {})
    if teacher_states and set(teacher_states) != set(TEACHERS):
        raise ValueError('Unexpected teacher identities')
    process = reader.process(launch)
    teachers = {}
    complete_markers = True
    for teacher in TEACHERS:
        rows = []
        for round_no in range(1, 6):
            prefix = f'{base}/{teacher}'
            folder = f'{prefix}/round_{round_no:02d}'
            sampling = reader.json(f'{folder}/wave/COMPLETE.json')
            quality = reader.json(f'{folder}/wave/QUALITY_COMPLETE.json')
            row = {'round': round_no,
                   'sampling': sampling is not None and sampling.get('attempts') == 100,
                   'quality': quality is not None and quality.get('attempts') == 100,
                   'cleanup': reader.json(f'{folder}/wave/CLEANUP_COMPLETE.json') is not None,
                   'council': reader.json(f'{folder}/council/COMPLETE.json') is not None,
                   'retention': reader.json(f'{prefix}/RETENTION_COMPLETE_{round_no:02d}.json') is not None}
            complete_markers &= all(row[key] for key in ('sampling', 'quality', 'cleanup', 'council', 'retention'))
            rows.append(row)
        current = teacher_states.get(teacher, {})
        teachers[teacher] = {'round': current.get('round'), 'phase': current.get('phase'),
                             'has_error': bool(current.get('error')), 'rounds': rows,
                             'error': error_summary(current.get('error'))}
        if current.get('phase') == 'failed' and type(current.get('round')) is int and 1 <= current['round'] <= 5:
            failures = reader.json(f"{base}/{teacher}/round_{current['round']:02d}/council/analyst_failures.json")
            if failures:
                teachers[teacher]['failed_analysts'] = {g: error_summary(failures[g])
                    for g in ('G1', 'G2', 'G3', 'G4') if g in failures}
    final_name = f'FINALIZED_{version}.json'
    final_bytes = reader.data(final_name)
    final = json.loads(final_bytes) if final_bytes is not None else None
    pipeline_complete = reader.json(f'{base}/PIPELINE_COMPLETE.json')
    verified = False
    if final is not None:
        report = reader.data(f'{base}/report.json')
        manifest = reader.data(f'{base}/review/manifest.json')
        verified = bool(report is not None and manifest is not None
            and final.get('version') == version and final.get('rounds') == 5
            and final.get('outcomes') == 1000
            and final.get('report_sha256') == hashlib.sha256(report).hexdigest()
            and final.get('review_manifest_sha256') == hashlib.sha256(manifest).hexdigest())
    # Do not announce completion from a torn snapshot or just from worker exit.
    stable = reader.data('launch.json') == launch_bytes and reader.data(final_name) == final_bytes
    finished = (stable and verified and complete_markers
                and (pipeline_complete or {}).get('teacher_rounds') == {t: 5 for t in TEACHERS}
                and all(t.get('phase') == 'complete' and t.get('round') == 5 for t in teacher_states.values())
                and len(teacher_states) == 2)
    failed = any(t['has_error'] or t['phase'] == 'failed' for t in teachers.values())
    state = ('snapshot_changed' if not stable else 'finalized' if finished else
             'inconsistent_completion' if final is not None else 'pipeline_failed' if failed else
             'running' if process['worker_alive'] else 'needs_inspection')
    return {'schema': 1, 'transport': 'bounded_read_only_sftp', 'observed_at': time.time(),
            'version': version, 'state': state, 'completion_verified': bool(finished),
            'launch': {k: launch[k] for k in ('version', 'commit', 'pid', 'time')},
            'process': process, 'teachers': teachers,
            'new_failure_receipts': reader.failures(launch['time']),
            'source_sha256': reader.hashes,
            'unavailable': {'disk_free_bytes': 'not exposed by this SFTP interface',
                            'raw_log_tail': 'excluded; inspect only for a specific failure'},
            'remote_execution': False, 'remote_writes': False}


def connect_and_collect(version):
    import paramiko  # Local optional dependency; never import the scientific runtime.

    class PinnedHost(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            actual = 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
            if actual != HOST_KEY:
                raise ValueError('SSH host key does not match the pinned server')

    validate_version(version)
    if not __import__('sys').stdin.isatty():
        raise ValueError('A terminal is required for non-echoing credential input')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHost())
    password = getpass.getpass('SSH password (not stored): ')
    try:
        client.connect(HOST, port=PORT, username=USER, password=password,
                       look_for_keys=False, allow_agent=False, timeout=25,
                       auth_timeout=25, banner_timeout=25)
        del password
        with client.open_sftp() as sftp:
            sftp.get_channel().settimeout(25)
            return collect(sftp, version)
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, type=validate_version)
    parser.add_argument('--output', type=Path, help='Optional local JSON receipt; never a remote destination')
    args = parser.parse_args()
    try:
        receipt = connect_and_collect(args.version)
    except Exception as exc:
        # Do not echo authenticated URLs, arbitrary remote text or credentials.
        print(json.dumps({'state': 'read_failed', 'error_type': type(exc).__name__,
                          'remote_execution': False, 'remote_writes': False}))
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_sha256'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
