"""Actual side-stream producers, asynchronous parent consumers, allocator churn."""
import gc
import threading
import pytest
import torch

from agenticgen.uniguide.gpu_execution import TeacherGpuExecutor
from agenticgen.uniguide.gpu_resources import NativeCounters, snapshot, plateau


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fixed_stream_vjp_lifetimes_and_native_resource_plateau(tmp_path):
    # Deterministic tensors; this test does not draw from any random generator.
    base = torch.arange(512 * 512, device="cuda", dtype=torch.float32).reshape(512, 512) / (512 * 512)
    reference = (base @ base.T).sum().item()
    counters = NativeCounters(tmp_path)
    resources, threads, checksums = [], [], []
    with TeacherGpuExecutor(2) as executor:
        for batch in range(8):
            ids = set()
            for step in range(100):
                def operation(index):
                    x = base.detach().clone().requires_grad_(True)
                    output = x @ base.T
                    gradient = torch.autograd.grad(output.sum(), x)[0]
                    return {"output": output.detach(), "gradient": gradient,
                            "thread": threading.get_ident()}
                produced = executor.map(operation, [0, 1], inputs=base)
                ids.update(p["thread"] for p in produced)
                # Parent work is enqueued after the producer futures complete.
                # Drop side-stream tensors before the consumer executes, then
                # allocate new side-stream buffers at the same shape.
                consumer = torch.cat([p["output"] for p in produced])
                del produced
                churn = executor.map(lambda i: torch.empty_like(base).fill_(-7), [0, 1], inputs=base)
                actual = consumer[:512].sum().item()
                assert actual == reference
                checksums.append(actual)
                del churn, consumer
            threads.append(ids)
            gc.collect()
            resources.append(snapshot(executor, counters))
        assert all(ids == threads[0] and len(ids) == 2 for ids in threads)
        assert all(value == reference for value in checksums)
        assert plateau(resources)["plateau_observed"]
