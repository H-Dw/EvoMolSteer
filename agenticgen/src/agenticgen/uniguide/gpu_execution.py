"""Teacher-scoped GPU scheduling; candidate population and concurrency are separate."""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import threading

import torch

POLICY = "teacher-persistent-gpu-execution-v1"


def settings(config):
    supplied = config.get("gpu_execution", {})
    if not isinstance(supplied, dict) or set(supplied) - {"workers"}:
        raise ValueError("gpu_execution accepts only workers")
    workers = supplied.get("workers", 1)
    if type(workers) is not int or not 1 <= workers <= 5:
        raise ValueError("GPU workers must be an integer from 1 to 5")
    return {"workers": workers}


def tensors(value):
    """Traverse TensorDicts and ordinary tensor pytrees without flattening metadata."""
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping) or hasattr(value, "items"):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensors(child)


def record_tree(value, stream):
    for tensor in tensors(value):
        if tensor.is_cuda:
            tensor.record_stream(stream)


def clear_workspaces():
    """Call only after all GPU consumers have completed, never inside an active step."""
    if not torch.cuda.is_initialized():
        return False
    torch.cuda.synchronize()
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()
    gc.collect()
    torch.cuda.empty_cache()
    return clear is not None


@dataclass
class _Slot:
    pool: ThreadPoolExecutor
    stream: object


class _SkippedAfterFailure(RuntimeError):
    pass


class TeacherGpuExecutor:
    """One owner thread, fixed worker threads and fixed streams for one teacher.

    workers=1 uses the caller and its stream, reproducing UG04's serial layout.
    Parallel calls finish every producer before return and register returned
    allocations on the consumer stream. Thus asynchronous parent-side concatenation
    cannot release a side-stream allocation before its consumer finishes.
    """

    def __init__(self, workers=1):
        self.workers = settings({"gpu_execution": {"workers": workers}})["workers"]
        self.owner = threading.get_ident()
        self.owner_stream = torch.cuda.current_stream() if torch.cuda.is_initialized() else None
        self.slots = []
        self.closed = False
        self.poisoned = False
        self.workspaces_cleared = False

    def _start(self):
        if self.slots:
            return
        device = torch.cuda.current_device()
        for index in range(self.workers):
            stream = torch.cuda.Stream(device=device)
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"teacher-gpu-{index}")
            self.slots.append(_Slot(pool, stream))

    @staticmethod
    def _run(slot, parent, enabled, function, item, inputs, failed):
        if failed.is_set():
            raise _SkippedAfterFailure("Queued GPU work cancelled after a producer failed")
        try:
            with torch.cuda.device(slot.stream.device), torch.set_grad_enabled(enabled):
                slot.stream.wait_stream(parent)
                record_tree(inputs, slot.stream)
                with torch.cuda.stream(slot.stream):
                    try:
                        return function(item)
                    finally:
                        # Join even on OOM; no unfinished producer escapes this call.
                        slot.stream.synchronize()
        except Exception:
            failed.set()
            raise

    def map(self, function, items, *, inputs=()):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Nested or concurrent use of a teacher GPU executor")
        if self.closed or self.poisoned:
            raise RuntimeError("GPU executor is closed or requires a clean process")
        items = list(items)
        if self.workers == 1:
            try:
                if torch.cuda.is_initialized():
                    current = torch.cuda.current_stream()
                    if self.owner_stream is None:
                        self.owner_stream = current
                    if current != self.owner_stream:
                        raise RuntimeError("Serial teacher GPU stream changed during its lifetime")
                return [function(item) for item in items]
            except Exception:
                self.poisoned = True
                raise
        self._start()
        parent = torch.cuda.current_stream()
        enabled = torch.is_grad_enabled()
        failed = threading.Event()
        futures = [self.slots[i % self.workers].pool.submit(
            self._run, self.slots[i % self.workers], parent, enabled, function, item, inputs, failed
        ) for i, item in enumerate(items)]
        wait(futures)
        try:
            errors = [f.exception() for f in futures if f.exception() is not None
                      and not isinstance(f.exception(), _SkippedAfterFailure)]
            if errors:
                raise errors[0]
            results = [future.result() for future in futures]
            record_tree(results, parent)
            return results
        except Exception:
            self.poisoned = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for slot in self.slots:
            slot.pool.shutdown(wait=True)
        self.slots.clear()
        # A failed GPU process is discarded, not "repaired" by cache clearing.
        if not self.poisoned:
            self.workspaces_cleared = clear_workspaces()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.poisoned |= kind is not None
        self.close()


@contextmanager
def bind_executor(model, config):
    executor = TeacherGpuExecutor(**settings(config))
    if getattr(model, "_ug_gpu_executor", None) is not None:
        raise RuntimeError("Teacher GPU executor already installed")
    model._ug_gpu_executor = executor
    try:
        with executor:
            yield executor
    finally:
        del model._ug_gpu_executor


@contextmanager
def execution_model(loader, config, teacher):
    """Production coordinators own a persistent child; fixtures can supply a model."""
    from .prepare import load_model
    from .gpu_worker import TeacherProcess

    if loader is load_model:
        with TeacherProcess(config, teacher) as model:
            yield model
    else:
        model = loader(config, teacher)
        if not hasattr(model, "integrator"):
            yield model
        else:
            with bind_executor(model, config):
                yield model
