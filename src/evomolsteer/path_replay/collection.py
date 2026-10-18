"""Stream a compact, independently verifiable result snapshot without remote duplication."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

from evomolsteer.io import digest


def collect(cfg,work,output):
    work=Path(work);members=[]
    for name in ('methods','tapes','analysis','logs','diagnostics'):
        root=work/name
        if root.exists(): members += [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    members += [p for p in work.glob('*.json') if p.is_file() and not p.is_symlink()]
    members += [p for p in work.glob('*.log') if p.is_file() and not p.is_symlink()]
    # Read and hash the exact bytes packed, not a potentially changing queue log
    # before the archive is emitted. Terminal result artifacts are immutable.
    inventory={};target=sys.stdout.buffer if str(output)=='-' else Path(output).open('wb')
    try:
        with tarfile.open(fileobj=target,mode='w|gz') as archive:
            for p in sorted(set(members)):
                data=p.read_bytes();name=p.relative_to(work).as_posix();info=tarfile.TarInfo(name);info.size=len(data)
                archive.addfile(info,io.BytesIO(data));inventory[name]={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            for name,value in [('protocol.json',cfg),('collection_manifest.json',inventory)]:
                data=json.dumps(value,sort_keys=True,indent=2).encode();info=tarfile.TarInfo(name);info.size=len(data)
                archive.addfile(info,io.BytesIO(data))
    finally:
        if target is not sys.stdout.buffer: target.close()


def unpack_snapshot(path,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with tarfile.open(path,'r:gz') as archive:
        manifest=json.load(archive.extractfile('collection_manifest.json'))
        for member in archive.getmembers():
            if not member.isfile() or not (output/member.name).resolve().is_relative_to(output):
                raise ValueError('Unsafe collection member')
            data=archive.extractfile(member).read()
            if member.name in manifest:
                row=manifest[member.name]
                if len(data)!=row['size'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('Snapshot member checksum mismatch: '+member.name)
            elif member.name not in ('protocol.json','collection_manifest.json'): raise ValueError('Unlisted snapshot member')
        actual={m.name for m in archive.getmembers()}
        if not set(manifest).issubset(actual): raise ValueError('Snapshot is missing inventoried artifacts')
        # A new snapshot replaces mutable queue observations, but immutable
        # result checksums are validated again when read by the analysis layer.
        archive.extractall(output,filter='data')
    return {'files':len(manifest),'archive_sha256':digest(path),'output':str(output)}
