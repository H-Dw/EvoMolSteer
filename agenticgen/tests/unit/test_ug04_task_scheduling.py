import threading

import pytest
import torch

from agenticgen.uniguide import chunking
from agenticgen.uniguide.geometry_batch import CandidateControl
from agenticgen.uniguide.task_scheduling import POLICY, configure


def test_configuration_exposes_only_the_ug04_schedule():
    assert configure({}) == {"task_scheduling": POLICY}
    for settings in ({"workers": 1}, {"workers": 5}, {"workers": 0}):
        with pytest.raises(ValueError, match="removed"):
            configure({"gpu_execution": settings})
    with pytest.raises(ValueError, match="Only UG04"):
        configure({"task_scheduling": "persistent"})


@pytest.mark.parametrize("count", [10, 25, 50])
def test_chunked_network_and_input_gradient_run_serially_on_the_caller(count, monkeypatch):
    owner = threading.get_ident()
    threads, shapes = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("UG04 scheduling must not create a side stream")

    monkeypatch.setattr(torch.cuda, "Stream", forbidden)

    def network(batch):
        threads.append(threading.get_ident())
        shapes.append(len(batch["coords"]))
        return {"coords": batch["coords"].square()}

    x = torch.arange(count * 9, dtype=torch.float32).reshape(count, 3, 3).requires_grad_()
    output = chunking.chunk_call(network, ({"coords": x},), {}, count, pad_to=10)
    gradient = torch.autograd.grad(output["coords"].sum(), x)[0]
    assert threads == [owner] * ((count + 9) // 10)
    assert shapes == [10] * len(threads)
    assert torch.equal(output["coords"], x.square())
    assert torch.equal(gradient, 2 * x)


def test_controller_executes_one_ten_row_block_at_a_time_and_stops_on_failure():
    control = object.__new__(CandidateControl)
    owner = threading.get_ident()
    calls = []

    def block(start):
        assert threading.get_ident() == owner
        calls.append(start)
        if start == 20:
            raise RuntimeError("block fixture")
        return start

    with pytest.raises(RuntimeError, match="block fixture"):
        control.execute_blocks(block, range(0, 50, 10))
    assert calls == [0, 10, 20]
