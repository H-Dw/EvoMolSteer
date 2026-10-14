"""Real artifact I/O past the Windows legacy path limit."""
import os
from pathlib import Path

from evomolsteer.lineage_retrieval.runner import _atomic


def io_path(path):
    name = str(path.absolute())
    if os.name == 'nt' and not name.startswith('\\\\?\\'):
        return Path('\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name)
    return Path(name)


def test_atomic_native_event_file_survives_long_path_and_replacement(tmp_path):
    parent = tmp_path / 'native_audit'
    owned = [parent]
    while len(str(parent / 'attempt-1.events.jsonl.tmp')) <= 290:
        parent = parent / ('h' * 32)
        owned.append(parent)
    target = parent / 'attempt-1.events.jsonl'
    assert len(str(target.with_name(target.name + '.tmp'))) > 260
    assert parent.resolve().is_relative_to(tmp_path.resolve())
    try:
        _atomic(target, b'first event')
        assert io_path(target).read_bytes() == b'first event'
        _atomic(target, b'replaced event')
        assert io_path(target).read_bytes() == b'replaced event'
        assert not io_path(target.with_name(target.name + '.tmp')).exists()
    finally:
        if io_path(target).exists():
            io_path(target).unlink()
        # Remove only the empty, explicitly constructed test directories.
        for directory in reversed(owned):
            if io_path(directory).exists():
                io_path(directory).rmdir()
