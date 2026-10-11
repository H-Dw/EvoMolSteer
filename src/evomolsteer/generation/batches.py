"""Select fixed global batch indices without optimizing or resetting the seed."""
def resolve_batches(n,batch,indices=None):
    if n<1 or batch<1 or n%batch:
        raise ValueError('Complete batches required')
    values=list(range(n//batch)) if indices is None else [int(v) for v in indices.split(',')] if isinstance(indices,str) else list(indices)
    if not values or values!=sorted(set(values)) or any(v<0 for v in values) or len(values)*batch!=n:
        raise ValueError('Unique increasing nonnegative indices matching candidate count required')
    return values
def loading_count(batch, indices, prior_seed_mode='legacy_stream'):
    """Indexed priors construct only one actual batch, not a discarded prefix."""
    if prior_seed_mode not in {'legacy_stream','indexed_batch'} or batch<=0 or not indices:
        raise ValueError('Valid prior policy, batch and indices required')
    return batch if prior_seed_mode=='indexed_batch' else (max(indices)+1)*batch


def indexed_pairs(args, target, off, interpolant, loader, indices, master_seed):
    """Use the upstream loader and unchanged prior law at a declared RNG seed.

    Different conditions get identical prior tensors by construction. This is
    a new explicit seed policy, not byte equality to legacy stream positions.
    The caller binds it before any scientific inference outcomes are observed.
    """
    import copy
    for batch in indices:
        scoped=copy.copy(args)
        scoped.seed=master_seed+100003*batch
        target_loader=loader(scoped,target,interpolant)
        off_loader=loader(scoped,off,interpolant)
        yield batch,(next(iter(target_loader)),next(iter(off_loader)))
