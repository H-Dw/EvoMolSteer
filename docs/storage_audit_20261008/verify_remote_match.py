"""Freshly verify every single-arm local source file against the remote hashes."""
import argparse
import json
from pathlib import Path

from evomolsteer.io import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--hashes', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--archive-report', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.input).resolve()
    rows = json.loads(Path(a.hashes).read_text())
    for r in rows:
        src = (root/r['path']).resolve()
        if not src.is_relative_to(root) or src.stat().st_size != r['bytes'] or digest(src) != r['sha256']:
            raise AssertionError('Fresh remote/local source mismatch: ' + r['path'])
    archive = Path(a.archive)
    report = json.loads(Path(a.archive_report).read_text())
    if archive.stat().st_size != report['window_archive_bytes'] or digest(archive) != report['window_archive_sha256']:
        raise AssertionError('Verified archive changed')
    record = {'remote_source_files_freshly_sha256_verified': len(rows),
              'source_bytes': sum(r['bytes'] for r in rows), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': digest(archive), 'originals_modified': False,
              'source_code_sha256': digest(__file__)}
    write_json(a.output, record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
