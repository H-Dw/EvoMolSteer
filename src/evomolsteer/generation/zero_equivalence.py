"""Scientific tensor comparison, excluding paths, arm labels and timestamps."""
import gzip
from pathlib import Path
import numpy as np
import torch
from ..storage.arrays import byte_equal


def compare_final(native_path, zero_path):
    def load(path):
        with gzip.open(Path(path), 'rb') as stream:
            return torch.load(stream, map_location='cpu', weights_only=True)
    native, zero = load(native_path), load(zero_path)
    required = ['coords', 'atomics', 'bonds', 'charges', 'mask']
    optional = ['hybridization']
    fields = required + [key for key in optional if key in native or key in zero]
    checks = {key: key in native and key in zero and
              byte_equal(native[key].numpy(), zero[key].numpy()) for key in fields}
    # Heads are also scientific outputs. Record them without treating their
    # presence as an unconditional tensor schema requirement.
    for key in sorted(set(native.get('affinity', {})) | set(zero.get('affinity', {}))):
        left, right = native.get('affinity', {}).get(key), zero.get('affinity', {}).get(key)
        checks['affinity.' + key] = (torch.is_tensor(left) and torch.is_tensor(right)
                                     and byte_equal(left.numpy(), right.numpy()))
    return {'byte_equal': checks, 'passed': all(checks.values()),
            'coordinate_rms_model_units': float(np.sqrt(np.mean(
                np.square(native['coords'].numpy().astype(float) - zero['coords'].numpy().astype(float)))))}
