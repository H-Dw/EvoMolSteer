"""Streaming, byte-verified tar.gz groups and recoverable retirement of owned data."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import tarfile

from ..io import digest,read_json
from .transactions import atomic_json,checked_path,unlink_verified

FORMAT='evomolsteer.steer_target_archive.v1'
MANIFEST='STEER_TARGET_ARCHIVE_MANIFEST.json'


def _hash_stream(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _member(name,size):
    info=tarfile.TarInfo(name);info.size=size;info.mode=0o644
    info.uid=info.gid=0;info.mtime=0
    return info


def _safe_member(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or str(path)!=name:
        raise ValueError('Unsafe archive path: '+name)
    return path


def verify_target_archive(archive,metadata=None):
    """Verify every payload without extracting it or trusting only a file size."""
    archive=Path(archive)
    meta=read_json(metadata) if isinstance(metadata,(str,Path)) else metadata
    if meta and (archive.stat().st_size!=meta['archive_bytes'] or digest(archive)!=meta['archive_sha256']):
        raise ValueError('Archive transport checksum mismatch')
    names=set();actual={};manifest=None
    # One forward decompression pass, including the gzip footer. Random tar
    # seeks could repeatedly decompress a large group, especially with Windows
    # case-insensitive source ordering and case-sensitive JSON key ordering.
    with gzip.open(archive,'rb') as gz:
        with tarfile.open(fileobj=gz,mode='r|') as tf:
            for m in tf:
                _safe_member(m.name)
                if m.name in names:raise ValueError('Duplicate tar member')
                names.add(m.name)
                if not m.isfile():raise ValueError('Links/directories are not valid payloads: '+m.name)
                if m.name==MANIFEST:manifest=json.load(tf.extractfile(m))
                else:actual[m.name]={'bytes':m.size,'sha256':_hash_stream(tf.extractfile(m))}
        for block in iter(lambda:gz.read(1024*1024),b''):pass
    if manifest is None:raise ValueError('Archive manifest is missing')
    if manifest.get('format')!=FORMAT or not manifest.get('complete'):
        raise ValueError('Unsupported/incomplete archive')
    checks=manifest['files'];targets=manifest['targets']
    if names!=set(checks)|{MANIFEST} or MANIFEST in checks:
        raise ValueError('Archive members differ from its declared payload')
    for name,r in checks.items():
        p=_safe_member(name)
        if len(p.parts)<3 or p.parts[0]!='targets' or p.parts[1] not in targets:
            raise ValueError('Payload is outside its target directory')
        if actual[name]!=r:raise ValueError('Payload checksum mismatch: '+name)
    source_bytes=sum(r['bytes'] for r in checks.values())
    if meta and (source_bytes!=meta['source_bytes'] or len(checks)!=meta['files']):
        raise ValueError('Sidecar payload size/count mismatch')
    size=archive.stat().st_size
    return {'verified':True,'archive_sha256':digest(archive),'archive_bytes':size,
            'source_bytes':source_bytes,'saved_payload_bytes':source_bytes-size,
            'reduction_percent':100*(source_bytes-size)/source_bytes if source_bytes else None,
            'files':len(checks),'target_count':len(targets),'manifest':manifest}


def archive_target_directories(directories,output,*,target_ids=None,compression_level=6,scope='completed_steer_targets'):
    """Low-level archive of immutable directories. Generation completion is
    checked by the caller; scope also permits clearly labeled stored-data tests.
    No source is removed by this function.
    """
    if type(compression_level) is not int or not 1<=compression_level<=9:
        raise ValueError('gzip level must be an integer in [1,9]')
    if not directories:raise ValueError('At least one target directory is required')
    output=Path(output).resolve();partial=output.with_name(output.name+'.partial')
    sidecar=Path(str(output)+'.json')
    if output.exists() or partial.exists() or sidecar.exists():raise FileExistsError(output)
    files={};sources={};targets={}
    for key,value in sorted(directories.items()):
        if len(_safe_member('targets/'+key).parts)!=2 or key in ('.','..'):
            raise ValueError('One safe target directory key required')
        root=Path(value).resolve()
        if not root.is_dir() or output.is_relative_to(root):raise ValueError('Archive must be outside source directories')
        if any(root==old or root.is_relative_to(old) or old.is_relative_to(root) for old in sources.values()):
            raise ValueError('Archive sources must be disjoint')
        sources[key]=root;targets[key]={'target_id':(target_ids or {}).get(key,key)}
        for path in sorted(root.rglob('*')):
            checked_path(path,root)
            if not path.is_file():continue
            name='targets/'+key+'/'+path.relative_to(root).as_posix()
            _safe_member(name)
            files[name]={'sha256':digest(path),'bytes':path.stat().st_size}
        if not any(n.startswith('targets/'+key+'/') for n in files):raise ValueError('Empty target directory')
    manifest={'format':FORMAT,'complete':True,'scope':scope,'targets':targets,'files':files,
              'compression_level':compression_level,'array_losslessness':'All retained files preserved byte for byte'}
    data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    # No intermediate uncompressed tar; normalize headers for repeatable bytes.
    with partial.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=compression_level) as gz:
            with tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT) as tf:
                tf.addfile(_member(MANIFEST,len(data)),io.BytesIO(data))
                for name,r in files.items():
                    p=PurePosixPath(name);root=sources[p.parts[1]];path=root.joinpath(*p.parts[2:])
                    if digest(path)!=r['sha256'] or path.stat().st_size!=r['bytes']:
                        raise ValueError('Source changed during archive creation: '+name)
                    with path.open('rb') as f:tf.addfile(_member(name,r['bytes']),f)
        raw.flush();os.fsync(raw.fileno())
    result=verify_target_archive(partial)
    # Verification precedes publication. Sidecar remains available after retirement.
    partial.rename(output)
    result.pop('manifest')
    result.update(format=FORMAT,archive=str(output),source_deleted=False,compression_level=compression_level,scope=scope)
    atomic_json(sidecar,result)
    result['sidecar_bytes']=sidecar.stat().st_size
    result['net_reduction_percent_with_sidecar']=100*(result['source_bytes']-result['archive_bytes']-result['sidecar_bytes'])/result['source_bytes']
    return result


def retire_archived_directories(archive,metadata,directories,*,owned_root):
    """Retire only verified files inside the controller's explicit output root.
    A subset may remain after an interrupted retirement; unmanifested files or
    changed bytes block deletion. Never removes biological source inputs.
    """
    verified=verify_target_archive(archive,metadata);checks=verified['manifest']['files']
    root=Path(owned_root).resolve();planned=[];folders=[]
    for key,value in directories.items():
        directory=checked_path(value,root)
        if key not in verified['manifest']['targets']:raise ValueError('Target absent from archive')
        if not directory.exists():continue
        for path in sorted(directory.rglob('*')):
            checked_path(path,directory)
            if path.is_dir():continue
            name='targets/'+key+'/'+path.relative_to(directory).as_posix()
            if name not in checks or digest(path)!=checks[name]['sha256'] or path.stat().st_size!=checks[name]['bytes']:
                raise ValueError('Unarchived/changed source blocks retirement: '+str(path))
            planned.append((path,checks[name]['sha256'],directory))
        folders.append(directory)
    # Validate all targets before retiring the first file; recheck each file at unlink.
    for path,sha,directory in planned:unlink_verified(path,sha,directory)
    for directory in folders:
        for path in sorted((p for p in directory.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.rmdir()
        directory.rmdir()
    return {'source_deleted':True,'files_retired':len(planned),'archive_sha256':verified['archive_sha256']}


def restore_target(archive,target,output,metadata=None):
    report=verify_target_archive(archive,metadata);manifest=report['manifest']
    matches=[key for key,row in manifest['targets'].items() if key==target or row['target_id']==target]
    if len(matches)!=1:raise ValueError('One matching target is required')
    key=matches[0];output=Path(output).resolve();partial=output.with_name(output.name+'.partial')
    if output.exists() or partial.exists():raise FileExistsError(output)
    partial.mkdir(parents=True)
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            name=member.name
            if name==MANIFEST:continue
            p=PurePosixPath(name)
            if p.parts[1]!=key:continue
            r=manifest['files'][name]
            dest=partial.joinpath(*p.parts[2:]);checked_path(dest,partial);dest.parent.mkdir(parents=True,exist_ok=True)
            with tf.extractfile(member) as src,dest.open('xb') as f:
                for block in iter(lambda:src.read(1024*1024),b''):f.write(block)
            if digest(dest)!=r['sha256']:raise ValueError('Restored file differs: '+name)
    partial.rename(output)
    return {'verified':True,'target_id':manifest['targets'][key]['target_id'],'output':str(output)}


def main(argv=None):
    p=argparse.ArgumentParser(description='Verify a Steer target group, optionally restore one target for event analysis')
    p.add_argument('--archive',required=True);p.add_argument('--metadata');p.add_argument('--target');p.add_argument('--output')
    a=p.parse_args(argv)
    if bool(a.target)!=bool(a.output):p.error('--target and --output must be provided together')
    result=restore_target(a.archive,a.target,a.output,a.metadata) if a.target else verify_target_archive(a.archive,a.metadata)
    result.pop('manifest',None);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
