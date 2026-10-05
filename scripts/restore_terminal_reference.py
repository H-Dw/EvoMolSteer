import argparse
import hashlib
import json
from pathlib import Path
import tarfile

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    path=Path(a.archive);out=Path(a.output).resolve();meta=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    if out.exists():raise FileExistsError(out)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['archive_sha256']:raise ValueError('Archive hash')
    expected={r['path']:r for r in meta['files']}
    with tarfile.open(path) as archive:
        if {m.name for m in archive.getmembers()}!=set(expected):raise ValueError('Member set')
        for m in archive.getmembers():
            if not m.isfile() or not (out/m.name).resolve().is_relative_to(out):raise ValueError('Unsafe archive member')
            data=archive.extractfile(m).read()
            if hashlib.sha256(data).hexdigest()!=expected[m.name]['sha256']:raise ValueError(m.name)
            dest=out/m.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    print(json.dumps({'verified':True,'files':len(expected),'destination':str(out)}))
