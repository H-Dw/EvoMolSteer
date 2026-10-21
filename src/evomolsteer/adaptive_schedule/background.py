"""Persistent remote supervisor; a cohort boundary waits for the main-agent skill review."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from .io import DEFAULT_ROOT, write, read, commit, repository, source_fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--skill-version", choices=["v1", "v2"], default="v1")
    p.add_argument("--skill", type=Path)
    opt = p.parse_args()
    root = opt.root
    if not (root / "PREPARED.json").exists():
        raise RuntimeError("Prepared experiment missing. Explicitly prepare a new named cohort; never recreate a lost V1 implicitly")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY missing from supervisor environment")
    import fcntl
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "launch.json", {"pid": os.getpid(), "start_time": time.time(), "commit": commit(),
                                    "repository": str(repository()), "version": opt.skill_version,
                                    "ssh_port": 10104, "api_key_present": True})
        base = [sys.executable, "-B", "-m", "evomolsteer.adaptive_schedule.cli", "--root", str(root)]
        if not (root / "VALIDATED.json").exists():
            write(root / "status.json", {"phase": "validating", "time": time.time()})
            subprocess.run(base + ["validate"], check=True)
        gradient_path = root / "validation/gradient_checks.json"
        gradient = read(gradient_path) if gradient_path.exists() else {}
        if not gradient.get("passed") or gradient.get("source_fingerprint") != source_fingerprint():
            write(root / "status.json", {"phase": "checking_gradients", "time": time.time()})
            subprocess.run([sys.executable, "-B", str(repository() / "scripts/check_plip_adaptive_gradients.py"),
                            "--root", str(root)], check=True)
        command = base + ["run", "--skill-version", opt.skill_version, "--resume"]
        if opt.skill:
            command += ["--skill", str(opt.skill)]
        from .retention import enabled
        quality = None if enabled(read(root / "config.json")) else subprocess.Popen(
            base + ["profile", "--skill-version", opt.skill_version, "--watch"])
        worker = subprocess.Popen(command)
        write(root / "workers.json", {"gpu_pid": worker.pid, "quality_pid": quality.pid if quality else None,
                                      "quality_mode": "watcher" if quality else "synchronous_before_prune", "time": time.time()})
        exit_code = worker.wait()
        if exit_code:
            old = read(root / "status.json") if (root / "status.json").exists() else {}
            write(root / "status.json", {**old, "phase": "attention", "worker_exit_code": exit_code, "time": time.time()})
        quality_exit = quality.wait() if quality else 0
        subprocess.run(base + ["analyze", "--skill-version", opt.skill_version], check=True)
        subprocess.run(base + ["review-pack", "--skill-version", opt.skill_version], check=True)
        write(root / f"supervisor_{opt.skill_version}_exit.json",
              {"worker_exit": exit_code, "quality_exit": quality_exit, "time": time.time()})
        if exit_code or quality_exit:
            raise SystemExit(exit_code or quality_exit)


if __name__ == "__main__":
    main()
