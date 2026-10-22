"""Lossless shared blocks for repeated council evidence, never selective omission."""
from collections import Counter
from .io import canonical, value_sha

REF = '$evidence_ref'


def pack(value, minimum_bytes=256):
    counts, values = Counter(), {}

    def visit(x):
        if isinstance(x, dict) and REF in x:
            raise ValueError('Reserved evidence reference in original input')
        if isinstance(x, (dict, list)):
            key = canonical(x)
            if len(key.encode('utf-8')) >= minimum_bytes:
                counts[key] += 1
                values[key] = x
            for child in (x.values() if isinstance(x, dict) else x):
                visit(child)
    visit(value)
    repeated = sorted((k for k, n in counts.items() if n > 1), key=lambda k: value_sha(values[k]))
    ids = {key: f'B{i:04d}' for i, key in enumerate(repeated)}

    def encode(x, skip=None):
        if isinstance(x, (dict, list)):
            key = canonical(x)
            if key in ids and key != skip:
                return {REF: ids[key]}
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            return [encode(v) for v in x]
        return x
    encoded = encode(value)
    blocks = {ids[k]: encode(values[k], skip=k) for k in repeated}
    if unpack(encoded, blocks) != value:
        raise ValueError('Lossless evidence round-trip failed')
    # No size improvement means keep the original representation.
    if len(canonical([encoded, blocks])) >= len(canonical([value, {}])):
        return value, {}
    return encoded, blocks


def unpack(value, blocks, active=()):
    if isinstance(value, dict):
        if set(value) == {REF}:
            key = value[REF]
            if key in active or key not in blocks:
                raise ValueError('Cyclic or missing evidence block')
            return unpack(blocks[key], blocks, active+(key,))
        return {k: unpack(v, blocks, active) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v, blocks, active) for v in value]
    return value
