"""Durable, narrowly scoped file publication and retirement for generated data."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid
from ..io import clean,digest


def atomic_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8',newline='\n') as handle:
        handle.write(json.dumps(clean(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)


@contextmanager
def exclusive_lock(path):
    """OS locks release on process death; the one-byte lock file stays reusable."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    handle = path.open('a+b')
    if handle.seek(0,os.SEEK_END)==0:
        handle.write(b'\0');handle.flush()
    handle.seek(0)
    acquired=False
    try:
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError as error:
            raise RuntimeError('Another writer owns '+str(path)) from error
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


def checked_path(path,root):
    original,root = Path(path),Path(root).resolve()
    resolved = original.resolve()
    if original.is_symlink() or not resolved.is_relative_to(root) or resolved==root:
        raise ValueError('File is outside its declared dataset or is a symlink: '+str(original))
    # A symlink/junction in any ancestor must not redirect deletion.
    current = original.absolute().parent
    while current!=root and current!=current.parent:
        if current.is_symlink() or (hasattr(current,'is_junction') and current.is_junction()):
            raise ValueError('Redirected dataset path: '+str(current))
        current=current.parent
    return resolved


def unlink_verified(path,expected,root):
    path = checked_path(path,root)
    if digest(path)!=expected:
        raise ValueError('Source changed; refusing to delete '+str(path))
    path.unlink()
