"""Run or resume the authorized fifty-round CK2 ancestry retrieval study."""
from contextlib import contextmanager
import errno
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
from evomolsteer.lineage_retrieval import orchestrate


def _study_path(argv: list[str] | None = None) -> Path:
    """Resolve --study as orchestrate.main does, before it executes."""
    args = list(sys.argv[1:] if argv is None else argv)
    value = orchestrate.DEFAULT_STUDY
    for index, arg in enumerate(args):
        if arg == '--study' and index + 1 < len(args):
            value = args[index + 1]
            break
        if arg.startswith('--study='):
            value = arg.partition('=')[2]
            break
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


@contextmanager
def exclusive_study_lock(study: str | Path):
    """Hold a nonblocking OS lock; the persistent file is never truncated.

    The kernel releases an advisory lock on process exit or termination. A
    leftover zero-byte lock file is harmless and needs no PID/timer heuristic.
    """
    study_path = Path(study).expanduser().resolve()
    lock_path = study_path / 'runtime' / '.run.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open('a+b')
    try:
        # Windows byte-range locks require a byte to exist. Append only when
        # empty; concurrent initializers may append an extra NUL, harmlessly.
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'CK2 study is already locked by another process: {study_path}') from exc
            def unlock():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise RuntimeError(f'CK2 study is already locked by another process: {study_path}') from exc
                raise
            def unlock():
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield lock_path
        finally:
            unlock()
    finally:
        stream.close()


def main() -> None:
    # Help is read-only and should not create runtime directories or lock files.
    if any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        orchestrate.main()
        return
    with exclusive_study_lock(_study_path()):
        orchestrate.main()


if __name__ == '__main__':
    main()
