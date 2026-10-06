"""Exact RNG snapshots for paired inference and side-effect-free numerical audits."""
import random
import numpy as np
import torch


def rng_state():
    return {'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}


def set_rng(state):
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
