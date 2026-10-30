"""Atomic artifacts, content bindings and an attempt ledger."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frozen(path, value):
    if Path(path).exists():
        if value_sha(read(path)) != value_sha(value):
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        write(path, value)
    return value


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(importlib.import_module(name), "__version__", "source"))


class Budget:
    """Reserve before sampling. Failed attempts remain charged; resume does not refill."""
    def __init__(self, root):
        self.path = Path(root) / "attempts.json"

    def reserve(self, run_id, stage, count, binding):
        caps = {"calibration": 5000, "r0": 100, "formal": 30000}
        limits_path = self.path.parent / "ATTEMPT_LIMITS.json"
        limits = read(limits_path) if limits_path.exists() else {}
        caps.update(limits.get("stage_caps", {}))
        if stage not in caps or type(count) is not int or count <= 0:
            raise ValueError("Invalid scientific attempt reservation")
        rows = read(self.path) if self.path.exists() else {}
        item = {"stage": stage, "count": count, "binding": binding}
        per_teacher = limits.get("per_teacher_stage_caps", {}).get(stage)
        if per_teacher is not None:
            scope = read(self.path.parent / "BUDGET_SCOPE.json")
            if scope.get("stage") != stage or not scope.get("teacher_id"):
                raise ValueError("Scientific reservation requires a teacher budget scope")
            item["teacher_id"] = scope["teacher_id"]
        if run_id in rows:
            if rows[run_id] != item:
                raise ValueError("Attempt ID reused with a different experiment")
            return
        if sum(x["count"] for x in rows.values() if x["stage"] == stage) + count > caps[stage]:
            raise ValueError("Scientific attempt budget exhausted")
        if per_teacher is not None:
            same_stage = [x for x in rows.values() if x["stage"] == stage]
            if any(not x.get("teacher_id") for x in same_stage):
                raise ValueError("Unattributed teacher attempt budget")
            used = sum(x["count"] for x in same_stage if x["teacher_id"] == item["teacher_id"])
            if used + count > per_teacher:
                raise ValueError("Teacher scientific attempt budget exhausted")
        rows[run_id] = item
        write(self.path, rows)

