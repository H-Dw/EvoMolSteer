from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from agenticgen.uniguide import gpu_execution as gpu
from agenticgen.uniguide.gpu_resources import plateau
from agenticgen.uniguide.gpu_worker import TeacherProcess
from agenticgen.uniguide.io import read, write


def test_configuration_keeps_population_and_concurrency_separate():
    assert gpu.settings({}) == {"workers": 1}
    assert gpu.settings({"population": 50}) == {"workers": 1}
    for value in (0, 6, 1.0, True, "2"):
        with pytest.raises(ValueError):
            gpu.settings({"gpu_execution": {"workers": value}})
    with pytest.raises(ValueError):
        gpu.settings({"gpu_execution": {"workers": 1, "batch_size": 50}})


def test_serial_executor_uses_owner_and_creates_no_stream_or_thread(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kw: pytest.fail("Serial path created a side stream"))
    identity = threading.get_ident()
    with gpu.TeacherGpuExecutor(1) as executor:
        for step in range(100):
            assert executor.map(lambda item: (item, threading.get_ident()), range(5)) == [(i, identity) for i in range(5)]
        assert executor.slots == []
    with pytest.raises(RuntimeError, match="closed"):
        executor.map(lambda x: x, [1])


def fake_cuda(monkeypatch):
    events = []
    class Stream:
        device = 0
        def __init__(self, **kw): events.append(("create", id(self)))
        def wait_stream(self, other): events.append(("wait", id(self), id(other)))
        def synchronize(self): events.append(("join", id(self)))
    parent = Stream()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: parent)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "device", lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda *a: nullcontext())
    monkeypatch.setattr(gpu, "clear_workspaces", lambda: None)
    monkeypatch.setattr(gpu, "record_tree", lambda value, stream: events.append(("record", id(stream))))
    return parent, events


def test_fixed_threads_and_streams_reused_and_outputs_owned_by_parent(monkeypatch):
    parent, events = fake_cuda(monkeypatch)
    with gpu.TeacherGpuExecutor(2) as executor:
        calls = []
        for _ in range(20):
            calls.append(executor.map(lambda i: (i, threading.get_ident(), torch.is_grad_enabled()), range(5), inputs=("source",)))
        assert len(executor.slots) == 2
        assert len({row[1] for call in calls for row in call}) == 2
        assert all(call == calls[0] for call in calls)
        assert all(row[2] for row in calls[0])
        assert sum(event[0] == "create" for event in events) == 3  # parent plus two fixed streams
        assert sum(event == ("record", id(parent)) for event in events) == 20
    assert not executor.slots


def test_all_workers_join_before_error_and_failed_executor_cannot_be_reused(monkeypatch):
    fake_cuda(monkeypatch)
    finished = []
    first_pair = threading.Barrier(2)
    def operation(i):
        first_pair.wait(timeout=5)
        if i == 0:
            raise torch.cuda.OutOfMemoryError("fixture")
        time.sleep(.02)
        finished.append(i)
    with gpu.TeacherGpuExecutor(2) as executor:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            executor.map(operation, range(4))
        assert finished == [1]  # In-flight work joined; queued work never reached the GPU.
        with pytest.raises(RuntimeError, match="clean process"):
            executor.map(lambda i: i, [0])


def test_plateau_requires_real_counts_and_rejects_growth():
    base = dict(allocated_bytes=1000, reserved_bytes=2000, blas_workspace_count=2,
        blaslt_workspace_count=0, native_gpu_queue_count=4, persistent_worker_pools=0,
        persistent_streams=0, unavailable={})
    rows = [dict(base) for _ in range(8)]
    assert plateau(rows)["plateau_observed"]
    rows[-1]["blas_workspace_count"] += 1
    with pytest.raises(ValueError, match="plateau"):
        plateau(rows)
    rows[-1] = dict(base, unavailable={"queue": "not exposed"})
    with pytest.raises(ValueError, match="unavailable"):
        plateau(rows)
    rows[-1] = dict(base, reserved_bytes=100 << 20)
    with pytest.raises(ValueError, match="plateau"):
        plateau(rows)


def test_coordinator_reaps_oom_process_then_starts_distinct_pid_and_keeps_worker(tmp_path, monkeypatch):
    fixture = tmp_path / "worker.py"
    fixture.write_text('''import os,sys,json,pathlib
folder=pathlib.Path(sys.argv[1]);ordinal=int(sys.argv[2])
assert 'DEEPSEEK_API_KEY' not in os.environ
from agenticgen.uniguide.io import read,write,sha
for line in sys.stdin:
 p=pathlib.Path(line.strip());request=read(p)
 result=dict(request_sha256=sha(p),pid=os.getpid())
 if ordinal==1:
  recovery=folder/'OOM_RECOVERY.json';write(recovery,dict(failed_pid=os.getpid()))
  result.update(status='oom',recovery=str(recovery));write(p.with_suffix('.result.json'),result)
  os._exit(75)
 old=read(folder/'OLD_PROCESS_EXITED.json')
 assert old['pid']!=os.getpid() and old['process_reaped']
 result.update(status='complete',results=[dict(pid=os.getpid(),exact_request=request['jobs'])])
 write(p.with_suffix('.result.json'),result)
''', encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_secret_not_a_real_key")
    monkeypatch.setattr(TeacherProcess, "_command", lambda self: [sys.executable, "-u", str(fixture), str(self.folder), str(self.ordinal)])
    with TeacherProcess(dict(root=str(tmp_path)), dict(target="teacher")) as teacher:
        jobs = [dict(schedule=[1, 2, 3, 4], folder="fixed-slots")]
        first = teacher.execute_candidate_jobs({}, jobs)[0]
        second = teacher.execute_candidate_jobs({}, jobs)[0]
        assert first == second and first["exact_request"] == jobs
        assert read(teacher.folder / "exit_001.json")["exit_code"] == 75
        assert first["pid"] != read(teacher.folder / "process_001.json")["pid"]
        folder = teacher.folder
    assert read(folder / "exit_002.json")["exit_code"] == 0


def test_aborted_gpu_process_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(TeacherProcess, "_command", lambda self: [sys.executable, "-c", "import os;os._exit(134)"])
    with TeacherProcess(dict(root=str(tmp_path)), dict(target="teacher")) as teacher:
        with pytest.raises(RuntimeError, match="aborted"):
            teacher.execute_candidate_jobs({}, [{}])
        assert teacher.ordinal == 1
