import copy
import torch

def take(value, indices, size):
    if torch.is_tensor(value):
        return value[indices].clone() if value.ndim and value.shape[0] == size else value.clone()
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {k: take(v, indices, size) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if len(value) == size:
            return type(value)(copy.deepcopy(value[i]) for i in indices)
        return type(value)(take(v, indices, size) for v in value)
    return copy.deepcopy(value)


def join(values, sizes, indices):
    """Select rows from heterogeneous full populations without inferring slot identity."""
    first = values[0]
    if torch.is_tensor(first):
        if first.ndim and all(v.shape[0] == size for v, size in zip(values, sizes)):
            return torch.cat([v[i:i+1] for v, i in zip(values, indices)], 0)
        if any(not torch.equal(first, v) for v in values[1:]):
            raise ValueError('Non-row tensor differs in mixed-parent recovery')
        return first.clone()
    if isinstance(first, dict) or hasattr(first, 'items'):
        return {k: join([v[k] for v in values], sizes, indices) for k in first.keys()}
    if isinstance(first, (list, tuple)):
        if all(len(v) == size for v, size in zip(values, sizes)):
            return type(first)(copy.deepcopy(v[i]) for v, i in zip(values, indices))
        return type(first)(join([v[i] for v in values], sizes, indices) for i in range(len(first)))
    if any(v != first for v in values[1:]):
        raise ValueError('Non-row metadata differs in mixed-parent recovery')
    return copy.deepcopy(first)

