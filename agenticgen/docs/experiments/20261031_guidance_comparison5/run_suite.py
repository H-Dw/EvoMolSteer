"""Sequential independent teacher jobs; one failure never cancels the remaining jobs."""

import argparse
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from agenticgen.uniguide.io import read, write, frozen, sha


def durable_snapshot(suite, *, force=False):
    from durable_snapshot import snapshot_suite

    snapshot_suite(suite, force=force)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def execute_sequence(jobs, run, record):
    results = []
    for job in jobs:
        try:
            result = dict(
                job_id=job["id"], status="completed", result=run(job), timestamp=now()
            )
        except Exception as error:
            result = dict(
                job_id=job["id"],
                status="failed_skipped",
                error=repr(error),
                traceback=traceback.format_exc(),
                timestamp=now(),
                source_artifacts_preserved=True,
                automatic_retry=False,
            )
        results.append(result)
        record(job, result)
    return results


def check_preservation(suite):
    expected = read(suite / "SOURCE_PRESERVATION.json")
    changed = [name for name, digest in expected.items() if sha(name) != digest]
    if changed:
        raise ValueError("Protected historical source changed: " + repr(changed[:5]))


def quality_sources(config):
    root = Path(config["root"])
    sources = set()
    for path in (root / "versions/GD01").glob("*/advice_*/ROUND_COMPLETE.json"):
        sources.update(r["id"] for r in read(path)["groups"])
    final = root / "versions/GD01/FINALIZED.json"
    if final.exists():
        sources.update(read(final)["quality_source_groups"])
    # Keep completed groups from a failed, partly executed round as well.
    from agenticgen.uniguide.geometry_sampling import validate_receipt

    for path in (root / "versions/GD01").glob("*/iteration_*/*/COMPLETE.json"):
        sources.add(validate_receipt(path.parent)["id"])
    return sorted(sources)


def submit_quality(config):
    from agenticgen.uniguide.quality_queue import enqueue

    root = Path(config["root"])
    teachers = {
        s["teacher"]["target"]: s["teacher"] for s in read(root / "SUBJECTS.json")
    }
    errors = {}
    for source in quality_sources(config):
        folder = root / source
        if (folder / "QUALITY_COMPLETE.json").exists() or (
            folder / "QUALITY_FAILED.json"
        ).exists():
            continue
        try:
            row = read(folder / "aggregate.json")
            if row["attempts"] != 50:
                continue
            enqueue(config, folder, teachers[row["teacher_id"]])
        except Exception as error:
            errors[source] = dict(
                error=repr(error), does_not_imply_molecular_failure=True
            )
    if errors:
        write(root / "QUALITY_ENQUEUE_ERRORS.json", errors)


def validate_job_config(config, job):
    rounds = job["rounds"]
    if (
        type(rounds) is not int
        or rounds < 1
        or config["rounds"] != rounds
        or len(config["conditions"]) != 1
        or config["agent_tools"]["skill_release"] != job["skill_release"]
        or config["task_scheduling"] != "ug04-930db032-serial"
    ):
        raise ValueError(
            "Single-teacher configuration differs from its launch contract"
        )
    return rounds


def run_job(repo, suite, job):
    config_path = Path(job["config"])
    config = read(config_path)
    root = Path(config["root"])
    if root.exists():
        raise RuntimeError(
            "A job requires a fresh result root; preserve claims and queue a new repair attempt"
        )
    if sha(config_path) != job["config_sha256"]:
        raise ValueError("Job configuration changed after deployment")
    rounds = validate_job_config(config, job)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Interactive API environment not loaded")
    logdir = suite / "logs" / job["id"]
    logdir.mkdir(parents=True, exist_ok=False)
    stages = [
        ("prepare", ["prepare"]),
        ("acceptance", ["acceptance"]),
    ]
    if config.get("postprocessing_recovery"):
        stages.append(("recover_promotion", ["recover-promotion"]))
    stages.extend((("calibrate", ["calibrate"]), ("formal", ["run-version", "GD01"])))
    try:
        for stage, arguments in stages:
            check_preservation(suite)
            log = logdir / (stage + ".log")
            command = [
                sys.executable,
                "-u",
                str(repo / "agenticgen/scripts/run_uniguide_geometry.py"),
                "--config",
                str(config_path),
                *arguments,
            ]
            with log.open("xb") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                state = dict(
                    timestamp=now(),
                    job_id=job["id"],
                    stage=stage,
                    status="running",
                    child_pid=process.pid,
                    command=command,
                    log=str(log),
                    root=str(root),
                )
                write(suite / "STATUS.json", state)
                while process.poll() is None:
                    if stage == "formal":
                        submit_quality(config)
                    write(
                        suite / "HEARTBEAT.json",
                        dict(
                            state, timestamp=now(), child_alive=process.poll() is None
                        ),
                    )
                    durable_snapshot(suite)
                    time.sleep(20)
            if stage == "formal":
                submit_quality(config)
            frozen(
                logdir / (stage.upper() + "_EXIT.json"),
                dict(
                    timestamp=now(), returncode=process.returncode, log_sha256=sha(log)
                ),
            )
            if process.returncode:
                raise RuntimeError(
                    f"{stage} exited {process.returncode}; preserved log {log}"
                )
            if stage == "prepare":
                subjects = read(root / "SUBJECTS.json")
                if (
                    len(subjects) != 1
                    or subjects[0]["teacher"]["target"] != job["teacher"]
                ):
                    raise ValueError("Prepared teacher identity differs")
                if len(subjects[0]["condition"]["controlled_atom_ids"]) != 9:
                    raise ValueError(
                        "Only the nine original direct PLIP atoms are authorized"
                    )
                module = repo / "agenticgen/src/agenticgen/uniguide"
                frozen(
                    root / "QUALITY_MODULE_BINDING.json",
                    dict(
                        commit=job["commit"],
                        module_directory=str(module),
                        implementation_sha256={
                            name: sha(module / name)
                            for name in (
                                "quality_queue.py",
                                "quality.py",
                                "profiling.py",
                            )
                        },
                    ),
                )
        final = read(root / "versions/GD01/FINALIZED.json")
        if (
            final["completed_rounds"] != rounds
            or job["teacher"] not in final["teachers"]
        ):
            raise ValueError("Configured teacher evaluation is incomplete")
        frozen(
            root / "SAMPLING_FINISHED.json",
            dict(
                timestamp=now(),
                rounds=rounds,
                source_sha256=sha(root / "versions/GD01/FINALIZED.json"),
            ),
        )
        check_preservation(suite)
        return dict(
            root=str(root),
            sampling_complete=True,
            rounds=rounds,
            quality_sources=quality_sources(config),
            quality_wait_is_independent=True,
        )
    except Exception as error:
        if "process" in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if (root / "GEOMETRY_CONFIG.json").exists():
            frozen(
                root / "CAMPAIGN_FAILED.json",
                dict(
                    timestamp=now(),
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    automatic_retry=False,
                    source_artifacts_preserved=True,
                ),
            )
        raise


def stop_idle_worker(config):
    root = Path(config["root"])
    path = root / "quality_queue/worker.json"
    status = root / "quality_queue/status.json"
    if not path.exists() or not status.exists():
        return
    worker, state = read(path), read(status)
    proc = Path("/proc") / str(worker["pid"])
    if state["active"] or state["pending"] or not (proc / "cmdline").exists():
        return
    command = (proc / "cmdline").read_bytes().split(b"\0")
    if (
        b"agenticgen.uniguide.quality_queue" in command
        and str(root / "quality_queue/runtime.json").encode() in command
    ):
        os.kill(worker["pid"], signal.SIGTERM)
        write(
            root / "QUALITY_WORKER_STOPPED.json",
            dict(timestamp=now(), pid=worker["pid"], idle=True),
        )


def main():
    import fcntl
    from agenticgen.uniguide.quality_queue import snapshot

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    repo, suite = Path(plan["repo"]), Path(plan["root"])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != plan["commit"]:
        raise ValueError("Fixed suite commit differs from the launch plan")
    with (
        (suite / ".suite.lock").open("a+") as own,
        Path(plan["global_executor_lock"]).open("a+") as global_lock,
    ):
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan.get("wait_for_global_lock", False):
            write(
                suite / "STATUS.json",
                dict(
                    timestamp=now(),
                    stage="waiting_for_global_lock",
                    status="queued",
                    pid=os.getpid(),
                    no_gpu_process_started=True,
                ),
            )
            fcntl.flock(global_lock, fcntl.LOCK_EX)
        else:
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen(
            suite / "SUITE_CLAIM.json",
            dict(
                timestamp=now(),
                pid=os.getpid(),
                plan_sha256=sha(args.plan),
                script_sha256=sha(__file__),
                execution_commit=plan["commit"],
            ),
        )

        def record(job, result):
            frozen(suite / "job_results" / (job["id"] + ".json"), result)
            write(suite / "STATUS.json", dict(result, stage="job_finished"))
            durable_snapshot(suite, force=True)

        results = execute_sequence(
            plan["jobs"], lambda job: run_job(repo, suite, job), record
        )
        frozen(
            suite / "ALL_JOBS_ATTEMPTED.json",
            dict(
                timestamp=now(),
                results=results,
                failed_jobs=[
                    r["job_id"] for r in results if r["status"] != "completed"
                ],
            ),
        )
        # Sampling order is independent of slow CPU diagnostics and failed siblings.
        while True:
            reports = {}
            for job in plan["jobs"]:
                config = read(job["config"])
                root = Path(config["root"])
                if not (root / "SUBJECTS.json").exists():
                    continue
                submit_quality(config)
                report = snapshot(root, quality_sources(config))
                reports[job["id"]] = report
            write(suite / "QUALITY_STATUS.json", reports)
            durable_snapshot(suite)
            if not any(r["pending"] for r in reports.values()):
                break
            time.sleep(20)
        check_preservation(suite)
        frozen(
            suite / "SUITE_FINISHED.json",
            dict(
                timestamp=now(),
                jobs=results,
                quality=reports,
                actual_gpt_review_required=True,
                automatic_scientific_retry=False,
            ),
        )
        write(
            suite / "STATUS.json",
            dict(
                timestamp=now(),
                stage="finished",
                status="awaiting_review_or_engineering_repair",
            ),
        )
        durable_snapshot(suite, force=True)


if __name__ == "__main__":
    main()
