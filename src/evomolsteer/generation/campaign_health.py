"""Read-only campaign health checks; completion requires native GPU evidence.

This module does not schedule, sleep, launch inference, extract archives or
alter progress markers. A caller decides when to perform the next check.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tarfile


OOM = re.compile(r'out of memory|torch\.(?:cuda\.)?OutOfMemoryError|'
                 r'(?:CUDA|HIP|ROCm).*\bOOM\b|(?:^|\s)MemoryError\b', re.I)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Campaign evidence is outside its output root')
    return path


def _documents(root, state, record):
    relative = record['dataset']
    names = ['learning_dataset_manifest.json',
             'results/' + state['config']['campaign'] + '/runtime.json']
    directory = _inside(root, relative)
    if all((directory / name).is_file() for name in names):
        return [_json(directory / name) for name in names], 'directory'
    archive = next((a for a in state['archives']
                    if a['path'] == record.get('archive')), None)
    if archive is None or not archive['metadata'].get('verified'):
        raise ValueError('No completed directory or verified archive locator')
    # Member keys retain the original target directory, including attempt suffixes.
    key = next(k for k, v in archive['directories'].items() if v == relative)
    wanted = {'targets/' + key + '/' + name: i for i, name in enumerate(names)}
    documents = [None] * len(names)
    with tarfile.open(_inside(root, archive['path']), 'r|gz') as stream:
        for member in stream:
            if member.name in wanted:
                if not member.isfile() or member.size > 8 * 1024 * 1024:
                    raise ValueError('Invalid native evidence member')
                documents[wanted[member.name]] = json.load(stream.extractfile(member))
                if all(document is not None for document in documents):
                    break
    if any(document is None for document in documents):
        raise ValueError('Native completion/runtime evidence is absent from archive')
    return documents, 'verified_archive'


def _native_proof(root, state, row, record):
    if record['state'] not in ('complete', 'archived'):
        return None
    progress = root / 'generation_progress'
    if not (progress / (row['key'] + '.finished')).is_file():
        raise ValueError('Completed target has no finished marker')
    if any((progress / (row['key'] + suffix)).exists() for suffix in ('.running', '.error')):
        raise ValueError('Completed target has conflicting progress markers')
    cfg = state['config']
    summary = record['summary']
    if summary['attempted_slots'] != cfg['samples'] or summary['built_slots'] < 1:
        raise ValueError('Target has no usable decoded molecule or incomplete slot coverage')
    if not record['attempts'] or record['attempts'][-1]['state'] != 'complete':
        raise ValueError('Target attempt did not complete')
    (manifest, runtime), location = _documents(root, state, record)
    verification = manifest.get('verification', {})
    if manifest.get('state') != 'complete' or verification.get('verified') is not True:
        raise ValueError('Native dataset verification did not complete')
    if verification.get('records') != cfg['samples']:
        raise ValueError('Native verified candidate count differs from campaign')
    if not manifest.get('checkpoint_sha256') or not manifest.get('flowr_sampler_sha256'):
        raise ValueError('Missing model or native sampler provenance')
    if 'evomolsteer.generation.steer_worker' not in manifest.get('command', []):
        raise ValueError('Completion was not produced by the native FLOWR worker')
    if not runtime.get('device') or not (runtime.get('cuda') or runtime.get('hip')):
        raise ValueError('No native GPU runtime evidence')
    if not Path(runtime['flowr_package']).resolve().is_relative_to(Path(cfg['flowr_root']).resolve()):
        raise ValueError('Runtime imported a different FLOWR program')
    if runtime.get('gradient_guidance') is not False or manifest.get('gradient_guidance') is not False:
        raise ValueError('Unexpected reward gradient in Steer collection')
    if runtime.get('selection_window') != [cfg['window_start'], cfg['window_end']]:
        raise ValueError('Runtime selection window differs from request')
    return {'target_id': row['target_id'], 'source': location,
            'attempted_slots': summary['attempted_slots'], 'built_slots': summary['built_slots'],
            'checkpoint_sha256': manifest['checkpoint_sha256'], 'device': runtime['device'],
            'torch': runtime.get('torch'), 'evomolsteer_commit': runtime.get('evomolsteer_commit')}


def check_campaign(dataset, *, controller_log=None, pid=None):
    root = Path(dataset).resolve()
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    report = {'dataset': str(root), 'status': 'pending', 'normal_target_verified': False,
              'oom': [], 'issues': [], 'free_disk_bytes': disk.free}
    if pid is not None:
        try:
            os.kill(pid, 0)
            alive = True
            if os.name != 'nt':
                stat = Path(f'/proc/{pid}/stat').read_text()
                alive = stat.rsplit(')', 1)[1].split()[0] != 'Z'
                command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
                alive = alive and b'run_steer_targets.py' in command
        except (ProcessLookupError, FileNotFoundError):
            alive = False
        report['controller_pid'] = pid
        report['controller_alive'] = alive
    logs = [Path(controller_log)] if controller_log else []
    logs += sorted((root / 'generation_progress').glob('*.error'))
    for log in logs:
        if not log.is_file():
            continue
        with log.open(encoding='utf-8', errors='replace') as stream:
            for line_number, line in enumerate(stream, 1):
                if OOM.search(line):
                    report['oom'].append({'path': str(log), 'line': line_number, 'text': line.strip()[:500]})
                    if len(report['oom']) >= 20:
                        break
    path = root / 'campaign_state.json'
    if path.is_file():
        state = _json(path)
        report['campaign_status'] = state['status']
        report['counts'] = {name: sum(r['state'] == name for r in state['targets'].values())
                            for name in ('pending', 'running', 'complete', 'archived', 'failed')}
        report['archives'] = [{k: a[k] for k in ('path', 'target_ids', 'retirement')}
                              for a in state['archives']]
        if state.get('archive_error'):
            report['issues'].append({'archive_error': state['archive_error']})
        for row in state['catalog']:
            record = state['targets'][row['target_id']]
            if record['state'] == 'failed':
                report['issues'].append({'target_id': row['target_id'], 'error': record.get('error')})
        for row in state['catalog']:
            try:
                proof = _native_proof(root, state, row, state['targets'][row['target_id']])
            except (ValueError, KeyError, OSError, StopIteration, tarfile.TarError) as error:
                report['issues'].append({'target_id': row['target_id'], 'evidence_error': str(error)})
                continue
            if proof:
                report['native_target'] = proof
                report['normal_target_verified'] = True
                break
    if pid is not None and not report['controller_alive'] and not report['normal_target_verified']:
        report['issues'].append({'controller': 'Exited before a native target completed'})
    if report['oom'] or report['issues']:
        report['status'] = 'attention'
    elif report['normal_target_verified']:
        report['status'] = 'normal'
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--controller-log')
    parser.add_argument('--pid', type=int)
    args = parser.parse_args(argv)
    report = check_campaign(args.dataset, controller_log=args.controller_log, pid=args.pid)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return {'normal': 0, 'pending': 1, 'attention': 2}[report['status']]
