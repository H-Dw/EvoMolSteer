"""Exclusive worker lock extracted without the legacy runner."""
import contextlib
from pathlib import Path

@contextlib.contextmanager
def lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
