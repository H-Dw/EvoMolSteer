"""One-use, content-bound OOM continuations; numerical failures are never retried."""
import json
import os
from pathlib import Path

from .io import frozen, read, sha, value_sha

POLICY = "clean-gpu-process-oom-v1"


def json_value(value):
    return json.loads(json.dumps(value, default=lambda x: str(x) if isinstance(x, Path) else x))


def request_binding(config, teacher, condition, jobs):
    return value_sha(json_value(dict(config=config, teacher=teacher, condition=condition, jobs=jobs)))


class GpuOOMRestartRequired(RuntimeError):
    def __init__(self, recovery):
        self.recovery = str(recovery)
        super().__init__("Confirmed device OOM requires a new GPU process: " + self.recovery)


def unit_files(root, unit):
    folder = unit["folder"]
    paths = [folder / "manifest.json", root / "slot_registry" / (unit["fingerprint"] + ".json")]
    paths += [folder / name for name in ("COMPLETE.json", "FAILED.json") if (folder / name).exists()]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def issue_recovery(config, teacher, condition, jobs, units, oom_path, capacity):
    root = Path(config["root"])
    record = dict(policy=POLICY, failed_pid=os.getpid(),
        request_sha256=request_binding(config, teacher, condition, jobs),
        next_capacity=capacity, oom_path=str(oom_path), oom_sha256=sha(oom_path),
        ledger_sha256=sha(root / "attempts.json"),
        units={u["id"]: dict(identity=u["identity"], files=unit_files(root, u)) for u in units},
        no_new_reservations=True, no_numerical_failure_retry=True)
    path = oom_path.parent / "CLEAN_PROCESS_RECOVERY.json"
    frozen(path, record)
    return path


def validate_recovery(path, config, teacher, condition, jobs):
    """Claim once, before reusing any interrupted reservation in the new process."""
    path = Path(path)
    record = read(path)
    root = Path(config["root"])
    if record["policy"] != POLICY or record["request_sha256"] != request_binding(config, teacher, condition, jobs):
        raise ValueError("OOM continuation belongs to another request")
    if record["failed_pid"] == os.getpid():
        raise RuntimeError("OOM continuation requires a different GPU process")
    exited = read(path.with_name("OLD_PROCESS_EXITED.json"))
    if exited.get("pid") != record["failed_pid"] or not exited.get("process_reaped") or exited.get("exit_code") != 75:
        raise RuntimeError("Failed GPU process must exit before OOM continuation")
    if sha(record["oom_path"]) != record["oom_sha256"] or sha(root / "attempts.json") != record["ledger_sha256"]:
        raise ValueError("OOM evidence or attempt ledger changed")
    oom = read(record["oom_path"])
    if not oom.get("clean_gpu_process_required") or oom["next_capacity"] != record["next_capacity"]:
        raise ValueError("No confirmed OOM continuation")
    for item in record["units"].values():
        for name, digest in item["files"].items():
            if sha(root / name) != digest:
                raise ValueError("OOM reservation or terminal evidence changed: " + name)
    claim = dict(recovery_sha256=sha(path), failed_pid=record["failed_pid"], new_pid=os.getpid(),
                 capacity=record["next_capacity"])
    with path.with_name("RECOVERY_CLAIM.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, sort_keys=True)
    return record
