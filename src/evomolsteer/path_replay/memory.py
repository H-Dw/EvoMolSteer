"""Lossless activation offload: keep the full B50 forward and original VJPs."""
from contextlib import nullcontext
import torch


def saved_tensor_context(policy):
    if policy=='native_device': return nullcontext()
    if policy=='saved_tensors_cpu_exact':
        # This changes storage only. No casts, checkpoint recomputation,
        # microbatching, parameter flag changes or reward substitutions.
        return torch.autograd.graph.save_on_cpu(pin_memory=False)
    raise ValueError('Unknown saved-tensor policy: '+str(policy))
