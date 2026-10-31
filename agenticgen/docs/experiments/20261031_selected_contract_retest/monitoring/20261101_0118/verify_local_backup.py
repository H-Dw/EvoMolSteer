"""Stream-verify every archived file without extracting large raw artifacts."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath

folder = Path(__file__).resolve().parent
complete = json.loads((folder / 'backup_COMPLETE.json').read_text(encoding='utf-8'))
manifest_path = folder / 'archive_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))['files']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


archive = folder / 'closed_artifacts.tar.gz'
assert archive.stat().st_size == complete['archive_bytes']
assert sha(archive) == complete['archive_sha256']
assert sha(manifest_path) == complete['manifest_sha256']
verified = {}
with tarfile.open(archive, 'r|gz') as tar:
    for member in tar:
        path = PurePosixPath(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        assert 'launcher.log' not in path.name and not path.name.startswith('core')
        assert path.suffix != '.lock' and member.name in manifest and member.name not in verified
        source = tar.extractfile(member)
        h = hashlib.sha256()
        size = 0
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
            size += len(block)
        assert size == manifest[member.name]['bytes'] == member.size
        assert h.hexdigest() == manifest[member.name]['sha256'], member.name
        verified[member.name] = size
assert set(verified) == set(manifest) and len(verified) == complete['files']
receipt = {'timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),
           'timezone':'Asia/Shanghai','archive':archive.name,'archive_sha256':complete['archive_sha256'],
           'archive_bytes':archive.stat().st_size,'manifest_sha256':complete['manifest_sha256'],
           'all_member_hashes_verified':True,'verified_files':len(verified),
           'verified_uncompressed_bytes':sum(verified.values()),
           'verified_hdf5':sum(n.endswith('.h5') for n in verified),
           'launcher_or_core_downloaded':False,'live_second_teacher_archived':False,
           'remote_results_deleted':False,'full_suite_finished':False}
(folder / 'local_backup_receipt.json').write_text(json.dumps(receipt, ensure_ascii=True, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, ensure_ascii=True))
