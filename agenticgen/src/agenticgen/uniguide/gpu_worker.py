"""Persistent teacher GPU child and CPU coordinator; only confirmed OOM restarts it.

Requests and results are immutable files. stdin carries paths, never tensor data or
credentials. EOF terminates an orphaned idle child. An abort is terminal, not OOM.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .io import read, write, frozen, sha, value_sha
from .gpu_recovery import GpuOOMRestartRequired, json_value


def inherited_executor_locks(config):
    """Keep the parent's advisory GPU locks alive if the coordinator crashes."""
    if sys.platform != "linux":
        return ()
    names = {str(Path(config[key]).resolve() / name) for key in ("root", "generation_context_root")
             if key in config for name in (".executor.lock", ".campaign.lock")}
    descriptors = []
    for link in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(link) in names:
                descriptors.append(int(link.name))
        except FileNotFoundError:
            continue
    return tuple(descriptors)


class TeacherProcess:
    def __init__(self, config, teacher):
        self.config, self.teacher = json_value(config), json_value(teacher)
        self.folder = Path(config["root"]) / "gpu_execution" / teacher["target"] / str(time.time_ns())
        frozen(self.folder / "config.json", self.config)
        frozen(self.folder / "teacher.json", self.teacher)
        self.process = None
        self.log = None
        self.ordinal = 0
        self.closed = False

    def _start(self):
        if self.closed:
            raise RuntimeError("Teacher GPU process is closed")
        if self.process is not None:
            raise RuntimeError("Previous GPU process has not been reaped")
        self.ordinal += 1
        self.log = (self.folder / f"worker_{self.ordinal:03d}.log").open("wb")
        environment = dict(os.environ)
        environment.pop("DEEPSEEK_API_KEY", None)
        options = dict(pass_fds=inherited_executor_locks(self.config)) if os.name == "posix" else {}
        self.process = subprocess.Popen(self._command(),
            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
            text=True, env=environment, **options)
        frozen(self.folder / f"process_{self.ordinal:03d}.json", dict(pid=self.process.pid,
            parent_pid=os.getpid(), config_sha256=sha(self.folder / "config.json")))

    def _command(self):
        return [sys.executable, "-u", "-m", "agenticgen.uniguide.gpu_worker",
                "--folder", str(self.folder), "--ordinal", str(self.ordinal)]

    def _reap(self, expected=None):
        process = self.process
        if process is None:
            return
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
        self.log.close()
        self.process = self.log = None
        frozen(self.folder / f"exit_{self.ordinal:03d}.json", dict(pid=process.pid, exit_code=code))
        if expected is not None and code != expected:
            raise RuntimeError(f"GPU process {process.pid} exited {code}, expected {expected}; no automatic retry")

    def execute_candidate_jobs(self, condition, jobs):
        if not jobs:
            return []
        recovery = None
        for _ in range(4):  # Bounded 50 -> 40 -> 20 -> 10; no retry of numerical errors.
            if self.process is None:
                self._start()
            request = self.folder / "requests" / (str(time.time_ns()) + ".json")
            frozen(request, json_value(dict(condition=condition, jobs=jobs, recovery=recovery)))
            result_path = request.with_suffix(".result.json")
            try:
                self.process.stdin.write(str(request) + "\n")
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._reap()
                raise RuntimeError("Teacher GPU process exited before request; no automatic retry")
            while not result_path.exists():
                if self.process.poll() is not None:
                    self._reap()
                    raise RuntimeError("Teacher GPU process aborted; preserve requests and reservations")
                time.sleep(.1)
            result = read(result_path)
            if result["request_sha256"] != sha(request):
                raise ValueError("GPU result request binding changed")
            if result["status"] == "complete":
                return [FloatingPointError(r["numerical_failure"]) if "numerical_failure" in r else r
                        for r in result["results"]]
            if result["status"] != "oom":
                self._reap(expected=1)
                raise RuntimeError("GPU request failed: " + result["error"])
            old_pid = self.process.pid
            self._reap(expected=75)  # A new child cannot start while the failed context still exists.
            recovery = result["recovery"]
            frozen(Path(recovery).with_name("OLD_PROCESS_EXITED.json"),
                dict(pid=old_pid, exit_code=75, process_reaped=True, coordinator_pid=os.getpid()))
        raise RuntimeError("GPU OOM capacity ladder exhausted")

    def close(self, *, check_exit=True):
        if not self.closed:
            self.closed = True
            self._reap(expected=0 if check_exit else None)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(check_exit=kind is None)


def serve(folder, ordinal):
    from .prepare import load_model
    from .gpu_execution import bind_executor
    from .geometry_batch import execute_many
    folder = Path(folder)
    config, teacher = read(folder / "config.json"), read(folder / "teacher.json")
    model = load_model(config, teacher)
    with bind_executor(model, config) as executor:
        frozen(folder / f"ready_{ordinal:03d}.json", dict(pid=os.getpid(),
            config_sha256=sha(folder / "config.json"), teacher_id=teacher["target"],
            candidate_batch_size=50, gpu_workers=executor.workers))
        for line in sys.stdin:
            path = Path(line.strip())
            request = read(path)
            response = dict(request_sha256=sha(path), pid=os.getpid())
            try:
                rows = execute_many(model, config, teacher, request["condition"], request["jobs"],
                    _recovery=request["recovery"])
                response.update(status="complete", results=[
                    dict(numerical_failure=str(r)) if isinstance(r, FloatingPointError) else r for r in rows])
            except GpuOOMRestartRequired as error:
                response.update(status="oom", recovery=error.recovery)
                write(path.with_suffix(".result.json"), response)
                # Never run CUDA cleanup in the failed context; the OS releases it.
                os._exit(75)
            except Exception as error:
                response.update(status="failed", error=f"{type(error).__name__}: {error}",
                                traceback=traceback.format_exc())
                write(path.with_suffix(".result.json"), response)
                return 1
            write(path.with_suffix(".result.json"), response)
            if request["recovery"]:
                frozen(Path(request["recovery"]).with_name("RECOVERY_COMPLETE.json"),
                    dict(new_pid=os.getpid(), result_sha256=sha(path.with_suffix(".result.json")),
                         no_new_reservations=True, no_numerical_failure_retry=True))
    frozen(folder / f"closed_{ordinal:03d}.json", dict(pid=os.getpid(),
        executor_closed=executor.closed, worker_pools_remaining=len(executor.slots),
        workspaces_cleared=executor.workspaces_cleared))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    arguments = parser.parse_args()
    raise SystemExit(serve(arguments.folder, arguments.ordinal))
