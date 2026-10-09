"""Resumable HTTP range downloads with a mandatory published checksum."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request


def checksum(path,algorithm='md5'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download_verified(url,destination,*,size,md5,workers=8,chunk_size=8*1024*1024,proxy=None):
    """Accept only exact byte ranges and publish only a fully verified file.

    Completed chunks survive interruption. A sidecar freezes source/size/checksum
    before resuming; the final checksum also validates any reused chunks.
    """
    destination=Path(destination)
    if type(size) is not int or size<1 or not re.fullmatch('[0-9a-f]{32}',md5):
        raise ValueError('Published positive size and MD5 are required')
    if type(workers) is not int or not 1<=workers<=64 or type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Invalid range download concurrency/chunk size')
    if destination.exists():
        if destination.stat().st_size!=size or checksum(destination)!=md5:
            raise ValueError('Existing archive differs from the published checksum: '+str(destination))
        return {'path':str(destination),'bytes':size,'md5':md5,'verified':True,'reused':True}
    destination.parent.mkdir(parents=True,exist_ok=True)
    parts=destination.with_name(destination.name+'.ranges');parts.mkdir(exist_ok=True)
    frozen={'url':url,'size':size,'md5':md5,'chunk_size':chunk_size}
    meta=parts/'source.json'
    if meta.exists() and json.loads(meta.read_text())!=frozen:
        raise ValueError('Cannot resume ranges from a different source')
    meta.write_text(json.dumps(frozen,indent=2))

    def fetch(index):
        lo=index*chunk_size;hi=min(size-1,lo+chunk_size-1);expected=hi-lo+1
        done=parts/f'{index:06d}.part'
        if done.exists() and done.stat().st_size==expected:return index
        temporary=done.with_suffix('.writing')
        for attempt in range(5):
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler(
                    {'https':proxy,'http':proxy} if proxy else {}))
                request=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
                with opener.open(request,timeout=45) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':
                        raise ValueError('Server returned an incorrect byte range')
                    with temporary.open('wb') as output:
                        remaining=expected
                        while remaining:
                            block=response.read(min(256*1024,remaining))
                            if not block:raise IOError('Truncated byte range')
                            output.write(block);remaining-=len(block)
                        if response.read(1):raise ValueError('Oversized byte range')
                os.replace(temporary,done)
                return index
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt==4:raise
                time.sleep(min(2**attempt,8))
        raise AssertionError('Unreachable')

    count=(size+chunk_size-1)//chunk_size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fetch,i) for i in range(count)]
        for finished,future in enumerate(as_completed(futures),1):
            future.result()
            if finished%10==0 or finished==count:
                print(json.dumps({'event':'download_progress','archive':destination.name,
                                  'completed_ranges':finished,'total_ranges':count}),flush=True)
    assembled=destination.with_name(destination.name+'.assembling')
    with assembled.open('wb') as output:
        for i in range(count):
            with (parts/f'{i:06d}.part').open('rb') as source:shutil.copyfileobj(source,output,1024*1024)
    if assembled.stat().st_size!=size or checksum(assembled)!=md5:
        assembled.unlink(missing_ok=True)
        raise ValueError('Downloaded archive failed the published checksum')
    os.replace(assembled,destination)
    shutil.rmtree(parts)
    return {'path':str(destination),'bytes':size,'md5':md5,'verified':True,'reused':False}
