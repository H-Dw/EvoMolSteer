"""Archive this closed suite, referencing only a previously locally verified index."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, fcntl, hashlib, io, json, os, shutil, tarfile

ROOT = Path('/workspace/agenticgen_box_exclusion_20261102')
COMMIT = 'c79627ad35670862694fb6dd59dab7272a9e7c9d'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, allow_nan=False)
        handle.write('\n')


def exclusion(path):
    name = path.name.lower()
    if 'launcher.log' in name:
        return 'remote_only_launcher'
    if name == 'core' or name.startswith('core.') or name.startswith('core_'):
        return 'remote_only_core'
    if name.endswith(('.ckpt', '.safetensors')) or name in ('pytorch_model.bin', 'model.pt', 'weights.pt'):
        return 'model_weight'
    if name.endswith(('.tar.gz', '.bundle')):
        return 'prior_archive_no_recursive_backup'
    if '__pycache__' in path.parts or name.endswith('.pyc'):
        return 'python_cache'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--previous', required=True)
    parser.add_argument('--inspection', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert out.parent == Path('/workspace/uniguide_audits') and out.name.startswith('box-exclusion-final-')
    out.mkdir(parents=True, exist_ok=True)
    write(out/'ARCHIVE_CLAIM.json', dict(timestamp=now(), pid=os.getpid(), source=str(ROOT), code_sha256=sha(Path(__file__))))
    plan, inspection = read(ROOT/'PLAN.json'), read(Path(args.inspection))
    assert plan['commit'] == inspection['actual_commit'] == COMMIT
    assert not inspection['runtime_source_changed'] and not inspection['protected_files_changed']
    assert (ROOT/'SUITE_FINISHED.json').exists()
    previous = read(Path(args.previous))
    assert previous['verified_locally']
    for job in plan['jobs']:
        base = Path(job['root'])
        assert read(ROOT/'job_results'/f"{job['id']}.json")['status'] == 'completed'
        assert (base/'QUALITY_WORKER_STOPPED.json').exists()
        assert sha(Path(job['config'])) == job['config_sha256']
        for request in (base/'quality_queue/requests').glob('*.json'):
            source = Path(read(request)['folder'])
            assert (source/'QUALITY_COMPLETE.json').exists() and not (source/'QUALITY_FAILED.json').exists()
    # Exclusive nonblocking lock probes do not create files or erase lock metadata.
    lock_paths = list(ROOT.rglob('*.lock')) + [Path(plan['global_executor_lock'])]
    for path in lock_paths:
        if path.exists():
            with path.open('rb') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
    audit = Path('/root/private_data/uniguide_box_exclusion_20261102/audits/box-exclusion-c79627ad')
    configurations = Path('/workspace/uniguide_runs/box-exclusion-c79627ad-configs')
    files, excluded, links = [], [], []
    for base in (ROOT, audit, configurations):
        for path in sorted(base.rglob('*')):
            reason = exclusion(path)
            if reason and (path.is_file() or path.is_symlink()):
                excluded.append(dict(path=str(path), reason=reason))
            elif path.is_symlink():
                links.append(dict(path=path.as_posix().lstrip('/'), target=os.readlink(path), dereferenced=False))
            elif path.is_file():
                files.append(path)
    before, references = {}, []
    for path in files:
        stat = path.stat()
        digest = sha(path)
        name = path.as_posix().lstrip('/')
        row = dict(path=name, bytes=stat.st_size, sha256=digest, source_mtime_ns=stat.st_mtime_ns)
        if previous['files'].get(name) == digest:
            references.append(row)
        else:
            before[path] = row
    write(out/'PREVIOUS_ARCHIVE_REFERENCES.json', dict(previous_index=str(args.previous),
        previous_index_sha256=sha(Path(args.previous)), previous_archive_sha256=previous['previous_archive_sha256'],
        verified_locally=True, files=references))
    write(out/'EXCLUDED_AND_SYMLINKS.json', dict(excluded=excluded, symlinks=links,
        experimental_checkpoint_010_100_and_terminal_pt_gz_preserved=True))
    for path in (out/'PREVIOUS_ARCHIVE_REFERENCES.json', out/'EXCLUDED_AND_SYMLINKS.json'):
        stat=path.stat()
        before[path]=dict(path=path.as_posix().lstrip('/'), bytes=stat.st_size, sha256=sha(path), source_mtime_ns=stat.st_mtime_ns)
    assert shutil.disk_usage('/workspace').free >= sum(row['bytes'] for row in before.values()) + (64<<20)
    archive = out/'completed_increment.tar.gz'
    manifest, content = [], {}
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for path, row in before.items():
            info = tarfile.TarInfo(row['path'])
            info.size, info.mtime, info.mode = row['bytes'], row['source_mtime_ns']//10**9, 0o600
            key=(row['sha256'], row['bytes'])
            if key in content:
                info.type, info.linkname, info.size = tarfile.LNKTYPE, content[key], 0
                tar.addfile(info)
            else:
                with path.open('rb') as handle:
                    tar.addfile(info, handle)
                content[key] = row['path']
            manifest.append(row)
        for link in links:
            info = tarfile.TarInfo(link['path'])
            info.type, info.linkname = tarfile.SYMTYPE, link['target']
            tar.addfile(info)
        for path, row in before.items():
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns, sha(path)) == (row['bytes'], row['source_mtime_ns'], row['sha256']), str(path)
        for row in references:
            path=Path('/'+row['path'])
            stat=path.stat()
            assert (stat.st_size, stat.st_mtime_ns, sha(path)) == (row['bytes'], row['source_mtime_ns'], row['sha256']), str(path)
        payload = (json.dumps(manifest, ensure_ascii=True, indent=2)+'\n').encode()
        info = tarfile.TarInfo('MANIFEST.json')
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    verified, verified_links, saved = {}, {}, None
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name == 'MANIFEST.json':
                saved = json.load(tar.extractfile(member))
            elif member.islnk():
                verified[member.name] = verified[member.linkname]
            elif member.issym():
                verified_links[member.name] = member.linkname
            else:
                assert member.isfile()
                count, h = 0, hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20), b''):
                        count += len(block)
                        h.update(block)
                verified[member.name] = (count, h.hexdigest())
    assert saved == manifest and len(verified) == len(manifest)
    assert verified_links == {link['path']:link['target'] for link in links}
    for row in manifest:
        assert verified[row['path']] == (row['bytes'], row['sha256'])
    write(out/'MANIFEST.json', manifest)
    receipt = dict(timestamp=now(), archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive),
        members=len(manifest), symlinks=len(links), hdf5_members=sum(row['path'].endswith('.h5') for row in manifest),
        previous_verified_local_files=len(references), all_members_verified=True, all_source_hashes_unchanged=True,
        previous_archive_sha256=previous['previous_archive_sha256'], suite_closed=True, own_workers_stopped=True,
        fixed_commit=COMMIT, plan_sha256=sha(ROOT/'PLAN.json'), launcher_core_weights_included=False,
        source_files_deleted=False, moved_to_private_data=False, downloaded=False, new_sampling=0, llm_calls=0)
    write(out/'COLLECT_COMPLETE.json', receipt)
    print(json.dumps(receipt, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
