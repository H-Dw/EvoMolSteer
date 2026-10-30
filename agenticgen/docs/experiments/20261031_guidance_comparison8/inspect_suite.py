"""Read real processes, locks and artifacts without importing a GPU runtime."""

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path


def read(path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lock_status(path):
    if not path.exists():
        return "absent"
    with path.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "held"
        fcntl.flock(stream, fcntl.LOCK_UN)
        return "free"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(Path(args.plan))
    root = Path(plan["root"])
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if int(proc.name) == os.getpid():
                continue
            command = (
                (proc / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if plan["repo"] in command or str(root) in command:
                status = (proc / "stat").read_text().split()
                processes.append(
                    dict(
                        pid=int(proc.name),
                        state=status[2],
                        parent_pid=int(status[3]),
                        command=command,
                    )
                )
        except (OSError, IndexError):
            pass
    expected = read(root / "SOURCE_PRESERVATION.json", {})
    changed = [
        path
        for path, digest in expected.items()
        if not Path(path).exists() or sha(path) != digest
    ]
    jobs = []
    for job in plan["jobs"]:
        config = read(Path(job["config"]))
        folder = Path(config["root"])
        attempts = read(folder / "attempts.json", {})
        steps = (
            sorted(folder.glob("**/progress.json"), key=lambda p: p.stat().st_mtime)[
                -6:
            ]
            if folder.exists()
            else []
        )
        logs = root / "logs" / job["id"]
        latest = (
            sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime)[-1:]
            if logs.exists()
            else []
        )
        logtail = (
            latest[0].read_bytes()[-2500:].decode(errors="replace") if latest else None
        )
        versions = folder / "versions/GD01"
        jobs.append(
            dict(
                id=job["id"],
                experiment=job["experiment"],
                teacher=job["teacher"],
                root=str(folder),
                result=read(root / "job_results" / (job["id"] + ".json")),
                locks={
                    str(p): lock_status(p)
                    for p in (
                        folder / ".executor.lock",
                        folder / "quality_queue/.worker.lock",
                    )
                },
                acceptance_complete=(folder / "acceptance/COMPLETE.json").exists(),
                calibrated_domains=list(read(folder / "calibration/domains.json", {})),
                reserved_attempts={
                    stage: sum(
                        r["count"] for r in attempts.values() if r["stage"] == stage
                    )
                    for stage in ("calibration", "formal")
                },
                completed_rounds=len(
                    list(versions.glob("*/advice_*/ROUND_COMPLETE.json"))
                ),
                llm_role_receipts=len(list(versions.glob("**/*.receipt.json"))),
                actual_http_responses=len(list(versions.glob("**/RESPONSE.json"))),
                progress=[
                    dict(path=str(p.relative_to(folder)), value=read(p)) for p in steps
                ],
                sampling_finished=read(folder / "SAMPLING_FINISHED.json"),
                failed=read(folder / "CAMPAIGN_FAILED.json"),
                quality_worker=read(folder / "quality_queue/worker.json"),
                quality_status=read(folder / "quality_queue/status.json"),
                latest_log=str(latest[0]) if latest else None,
                stage_log_tail=logtail,
            )
        )
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    report = dict(
        timestamp=stamp,
        timezone="Asia/Shanghai",
        plan_sha256=sha(args.plan),
        commit=plan["commit"],
        status=read(root / "STATUS.json"),
        heartbeat=read(root / "HEARTBEAT.json"),
        processes=processes,
        suite_lock=lock_status(root / ".suite.lock"),
        global_executor_lock=lock_status(
            Path(plan["generation_context_root"]) / ".executor.lock"
        ),
        preserved_files=len(expected),
        protected_files_changed=changed,
        jobs=jobs,
        all_jobs_attempted=read(root / "ALL_JOBS_ATTEMPTED.json"),
        suite_finished=read(root / "SUITE_FINISHED.json"),
    )
    destination = (
        root / "monitoring" / (stamp.replace(":", "").replace("+", "_") + ".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dict(snapshot=str(destination), **report), ensure_ascii=True))


if __name__ == "__main__":
    main()
