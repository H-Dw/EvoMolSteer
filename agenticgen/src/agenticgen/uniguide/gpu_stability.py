"""Repeated full native trajectories in one teacher executor; engineering only."""
import copy
import functools
import gc
from pathlib import Path
import h5py
import numpy as np
import torch

from . import sampler
from .geometry_batch import CandidateControl, batch_initial, DEFAULT_BATCH_SIZE
from .geometry_protocol import EFFECTIVE
from .gpu_resources import NativeCounters, snapshot, plateau
from .io import frozen, write
from .random_streams import capture, equal


def run_teacher(model, config, teacher, condition, folder, units, references, *, batches=8):
    if type(batches) is not int or batches < 6:
        raise ValueError("At least six consecutive full batches required")
    folder = Path(folder)
    executor = model._ug_gpu_executor
    counters = NativeCounters(folder / "counters")
    frozen(folder / "COUNTER_BINDING.json", dict(binding=counters.binding, unavailable=counters.errors))
    rows = []
    for batch in range(batches):
        selected = [copy.deepcopy(units[(i + batch) % len(units)])
                    for i in range(DEFAULT_BATCH_SIZE // 10)]
        for i, unit in enumerate(selected):
            unit["id"] = f"stability_{batch}_{i}"
        destination = folder / f"batch_{batch:03d}"
        factory = functools.partial(CandidateControl, condition=condition,
            scale_binding=selected[0]["scale"], units=selected, physical_rows=DEFAULT_BATCH_SIZE)
        before = capture()
        sampled = sampler.sample(model, config, teacher, selected[0]["scale"][EFFECTIVE],
            destination, DEFAULT_BATCH_SIZE, control_factory=factory, check_affinity=False,
            initial_override=batch_initial(config, teacher["target"], selected), stop_step=100)
        if sampled[3].failed_units:
            raise ValueError("Resource fixture produced a numerical failure")
        del sampled
        model._adaptive = None
        model.integrator._adaptive_observer = None
        if not equal(before, capture()):
            raise ValueError("Repeated resource batch consumed external RNG")
        with h5py.File(destination / "trace.h5") as combined:
            for i in range(len(selected)):
                with h5py.File(references[(i + batch) % len(units)] / "trace.h5") as single:
                    for name in single:
                        if not np.array_equal(single[name][:100], combined[name][:100, i * 10:(i + 1) * 10]):
                            raise ValueError("Repeated batch changed independent trajectory: " + name)
        gc.collect()  # Do not empty the allocator or workspace caches between batches.
        row = snapshot(executor, counters)
        row.update(batch=batch, candidate_batch_size=DEFAULT_BATCH_SIZE, full_steps=100,
            independent_trajectory_exact=True, external_rng_unchanged=True)
        rows.append(row)
        write(folder / "RESOURCES.json", rows)
    result = dict(plateau(rows), batches=batches, full_steps_per_batch=100,
        candidate_batch_size=DEFAULT_BATCH_SIZE, gpu_workers=executor.workers,
        independent_trajectories_exact=True, external_rng_unchanged=True,
        engineering_trajectory_evaluations=batches * DEFAULT_BATCH_SIZE,
        scientific_attempts=0, llm_calls=0)
    frozen(folder / "COMPLETE.json", result)
    return result
