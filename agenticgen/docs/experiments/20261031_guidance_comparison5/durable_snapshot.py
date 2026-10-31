"""Mirror safe audit/terminal artifacts onto NFS; leave full HDF5 on the run disk."""

import hashlib
import json
import shutil
import time
from pathlib import Path

_last_snapshot = 0.0


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Durable mirror must be outside the active result tree")
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / "MIRROR_STATE.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "launcher.log" or path.name.startswith("core"):
            continue
        if path.suffix not in (
            ".json",
            ".jsonl",
            ".csv",
            ".sdf",
            ".log",
        ) and not path.name.endswith(".pt.gz"):
            continue
        relative = path.relative_to(source).as_posix()
        before = path.stat()
        signature = dict(size=before.st_size, mtime_ns=before.st_mtime_ns)
        previous = state.get(relative, {})
        if all(previous.get(k) == v for k, v in signature.items()) and sha(
            path
        ) == previous.get("sha256"):
            continue
        if shutil.disk_usage(destination).free < before.st_size + (64 << 20):
            raise OSError("Insufficient NFS space for the next durable artifact")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".copying")
        shutil.copy2(path, temporary)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            temporary.unlink()
            continue  # Mutable progress/log file is copied on the next snapshot.
        digest = sha(temporary)
        if sha(path) != digest:
            temporary.unlink()
            continue
        temporary.replace(target)
        state[relative] = dict(signature, sha256=digest)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(state_path)
    return dict(
        files=len(state),
        bytes=sum(row["size"] for row in state.values()),
        full_hdf5_included=False,
        launcher_log_included=False,
    )


def snapshot_suite(suite, *, force=False):
    global _last_snapshot
    if not force and time.monotonic() - _last_snapshot < 120:
        return
    _last_snapshot = time.monotonic()
    from agenticgen.uniguide.io import read, write
    from datetime import datetime, timezone, timedelta

    plan = read(Path(suite) / "PLAN.json")
    try:
        result = dict(status="mirrored", **mirror(suite, plan["durable_snapshot_root"]))
    except (OSError, ValueError) as error:
        result = dict(status="error", error=repr(error), sampling_not_restarted=True)
    write(
        Path(suite) / "DURABLE_SNAPSHOT.json",
        dict(
            timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(), **result
        ),
    )
