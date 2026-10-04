"""Package a completed campaign with per-file hashes; never delete source data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from evomolsteer.io import read_json,digest,write_json


def archive(root,campaign,output):
    root=Path(root).resolve();run=(root/'results'/campaign).resolve();output=Path(output).resolve()
    if not run.is_relative_to(root/'results') or run.name!=campaign:raise ValueError('One campaign name required')
    if read_json(run/'COMPLETE.json')['status']!='complete':raise ValueError('Incomplete campaign')
    if output.exists():raise FileExistsError(output)
    if output.is_relative_to(run) or output.is_relative_to(root/'inputs'):raise ValueError('Archive must be outside its sources')
    config=read_json(run/'config.json');options=config['experiment']
    expected=options['n']//options['batch']
    for arm in options['arms'].split(','):
        completed=sorted((run/arm).glob('batch_*/COMPLETE.json'))
        if len(completed)!=expected:raise ValueError('Missing completed batch for '+arm)
        if any(read_json(p)['n']!=options['batch'] for p in completed):raise ValueError('Incomplete candidate batch')
    files=[]
    for directory in [root/'inputs',run]:
        for p in sorted(directory.rglob('*')):
            if p.is_symlink():raise ValueError('Symlinks are not archived')
            if p.is_file():files.append(p)
    checks={p.relative_to(root).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in files}
    manifest={'format':'evomolsteer.generation_archive.v1','campaign':campaign,'complete':True,
              'code_commit':config.get('extension',{}).get('code_commit'),'files':checks}
    data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial')
    if partial.exists():raise FileExistsError(partial)
    with tarfile.open(partial,'w:gz',compresslevel=3) as tf:
        info=tarfile.TarInfo('GENERATION_ARCHIVE_MANIFEST.json');info.size=len(data);tf.addfile(info,io.BytesIO(data))
        for p in files:
            rel=p.relative_to(root).as_posix()
            if digest(p)!=checks[rel]['sha256']:raise ValueError('Source changed while archiving: '+rel)
            tf.add(p,arcname=rel,recursive=False)
    # Read archive payloads back to verify transport contents, not just metadata.
    with tarfile.open(partial,'r:gz') as tf:
        for name,record in checks.items():
            h=hashlib.sha256();stream=tf.extractfile(name)
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=record['sha256']:raise ValueError('Archive roundtrip mismatch: '+name)
    partial.rename(output)
    result={'archive':str(output),'archive_sha256':digest(output),'bytes':output.stat().st_size,
            'files':len(files),'source_bytes':sum(v['bytes'] for v in checks.values()),'source_deleted':False}
    write_json(output.with_name(output.name+'.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(archive(a.dataset,a.campaign,a.output),indent=2))
