"""Observe this engineering run without importing CUDA or changing execution state."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def inspect(audit):
    audit = Path(audit)
    root = Path(read(audit / 'STATUS.json')['root'])
    processes = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if str(audit) in command and 'inspect_engineering.py' not in command:
                processes.append(dict(pid=int(path.name), state=(path / 'stat').read_text().split()[2], command=command))
        except (OSError, UnicodeError):
            continue
    config = read(audit / 'config.json') or {}
    locks = {}
    paths = [root / '.executor.lock']
    if config:
        paths.append(Path(config['generation_context_root']) / '.executor.lock')
    for path in paths:
        locks[str(path)] = 'absent'
        if path.exists():
            with path.open('r+') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locks[str(path)] = 'free'
                except BlockingIOError:
                    locks[str(path)] = 'held'
    traces = sorted(root.rglob('steps.json'), key=lambda p: p.stat().st_mtime)[-3:]
    resources = {str(path.relative_to(root)): read(path) for path in
                 root.glob('acceptance/*/candidate_batch/resource_stability/RESOURCES.json')}
    protected = read(audit / 'SOURCE_PRESERVATION.json') or {}
    info = dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        audit=str(audit), root=str(root), status=read(audit / 'STATUS.json'),
        processes=processes, locks=locks, failed=read(audit / 'FAILED.json'), complete=read(audit / 'COMPLETE.json'),
        acceptance=read(root / 'acceptance/COMPLETE.json'),
        latest_steps=[dict(path=str(p.relative_to(root)), count=len(rows), last_step=rows[-1]['step'])
                      for p in traces for rows in [read(p)] if rows],
        resources=resources, protected_files_count=len(protected),
        protected_files_changed=[name for name, digest in protected.items() if sha(Path(name)) != digest],
        unexpected_scientific_outputs=[name for name in ('attempts.json', 'calibration', 'versions') if (root / name).exists()],
        runtime_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=audit / 'checkout', text=True).strip(),
        runtime_dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=audit / 'checkout', text=True).strip())
    folder = audit / 'monitoring'
    folder.mkdir(exist_ok=True)
    target = folder / (datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '.json')
    target.write_text(json.dumps(info, ensure_ascii=True, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(target), **info), ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit')
    inspect(parser.parse_args().audit)
