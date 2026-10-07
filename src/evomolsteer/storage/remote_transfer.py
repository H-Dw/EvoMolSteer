"""Resumable transport of immutable remote files; integrity is checked separately.

Never treats equal file size as cryptographic verification. The caller must run
the archive/metadata checksum verifier before using or retiring the dataset.
"""
from pathlib import Path

def download_resumable(sftp,remote_path,local_path,block_size=32768,prefetch_requests=32):
    path=Path(local_path);path.parent.mkdir(parents=True,exist_ok=True)
    size=int(sftp.stat(remote_path).st_size);offset=path.stat().st_size if path.exists() else 0
    if offset>size:raise ValueError('Local partial file is longer than immutable remote file')
    resumed=offset
    if offset<size:
        with sftp.open(remote_path,'rb') as source,path.open('ab') as target:
            if hasattr(source,'settimeout'):source.settimeout(60)
            source.seek(offset)
            if hasattr(source,'prefetch'):source.prefetch(file_size=size,max_concurrent_requests=prefetch_requests)
            while offset<size:
                chunk=source.read(min(block_size,size-offset))
                if not chunk:raise OSError('Remote file ended before expected size; preserve partial download')
                target.write(chunk);offset+=len(chunk)
    if path.stat().st_size!=size:raise ValueError('Incomplete transport')
    return {'remote_bytes':size,'resumed_bytes':resumed,'downloaded_bytes':size-resumed,'checksum_verified':False}
