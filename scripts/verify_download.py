"""Verify remote digest, safely unpack, verify every immutable experiment file."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json,safe_extract,validate_checksums,digest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--manifest',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();m=read_json(a.manifest);archive=Path(a.archive)
    if archive.stat().st_size!=m['bytes']:raise ValueError('Archive size mismatch')
    safe_extract(archive,a.destination,m['sha256']);root=Path(a.destination)/Path(m['source']).name
    count=validate_checksums(root);state=read_json(root/'job_status.json')
    assert state['stage']=='complete'
    for name in ['run_all.exit','finalize.exit','quality_watch.exit']:assert (root/name).read_text().strip()=='0'
    write_json(Path(a.destination).parent/'download_manifest.json',{
        **m,'local_archive':str(archive.resolve()),'local_root':str(root.resolve()),'internal_files_verified':count,'verified':True})
    print('Verified archive and',count,'internal files')
