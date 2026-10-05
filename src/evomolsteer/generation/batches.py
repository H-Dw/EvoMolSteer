"""Select fixed global batch indices without optimizing or resetting the seed."""
def resolve_batches(n,batch,indices=None):
    if n<1 or batch<1 or n%batch:
        raise ValueError('Complete batches required')
    values=list(range(n//batch)) if indices is None else [int(v) for v in indices.split(',')] if isinstance(indices,str) else list(indices)
    if not values or values!=sorted(set(values)) or any(v<0 for v in values) or len(values)*batch!=n:
        raise ValueError('Unique increasing nonnegative indices matching candidate count required')
    return values
