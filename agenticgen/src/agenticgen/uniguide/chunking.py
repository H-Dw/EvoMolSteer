"""Bound physical network batches without changing the logical native integrator."""
import types
import torch
from .batching import take

ROWS=10


def concurrent_map(function,items,workers=1,*,executor=None,inputs=()):
    """Reuse an explicit teacher executor; never construct transient GPU workers."""
    if executor is not None:return executor.map(function,items,inputs=inputs)
    if workers!=1:raise RuntimeError("Parallel execution requires a teacher-scoped executor")
    return [function(item) for item in items]


def concatenate(values,sizes):
    first=values[0]
    if torch.is_tensor(first):
        if first.ndim and all(v.shape[0]==n for v,n in zip(values,sizes)):
            return torch.cat(values,0)
        if any(not torch.equal(first,v) for v in values[1:]): raise ValueError("Non-row output differs between chunks")
        return first
    if isinstance(first,dict) or hasattr(first,"items"):
        return {k:concatenate([v[k] for v in values],sizes) for k in first.keys()}
    if isinstance(first,(tuple,list)):
        # Native fragment_mode is one metadata item per row. A 25-row
        # population has physical pieces of 10, 10 and 5; these must be
        # flattened, unlike structural lists such as the two time tensors.
        if all(len(v)==n for v,n in zip(values,sizes)):
            return type(first)(item for value in values for item in value)
        if any(len(v)!=len(first) for v in values):
            raise ValueError("Non-row output sequences have inconsistent lengths")
        return type(first)(concatenate([v[i] for v in values],sizes) for i in range(len(first)))
    if any(v!=first for v in values[1:]): raise ValueError("Non-row network metadata differs")
    return first


def chunk_call(function,args,kwargs,count,rows=ROWS,pad_to=None,workers=1,executor=None):
    if count<=rows and (pad_to is None or count==pad_to): return function(*args,**kwargs)
    def block(start):
        ids=list(range(start,min(start+rows,count)))
        actual=len(ids)
        if pad_to is not None:
            if actual>pad_to: raise ValueError("Physical rows exceed frozen padding shape")
            ids=(ids*((pad_to+actual-1)//actual))[:pad_to]
        output=function(*(take(a,ids,count) for a in args),**{k:take(v,ids,count) for k,v in kwargs.items()})
        return take(output,list(range(actual)),len(ids)),actual
    parts=concurrent_map(block,range(0,count,rows),workers,executor=executor,inputs=(args,kwargs))
    values,sizes=zip(*parts)
    return concatenate(values,sizes)


def install(model):
    if hasattr(model,"_ug_original_forward"): raise ValueError("Chunking already installed")
    forward=model.forward
    model._ug_original_forward=forward
    def bounded_forward(this,*args,**kwargs):
        batch=args[0] if args else kwargs["batch"]
        rows=getattr(model,"_ug_physical_rows",ROWS)
        return chunk_call(forward,args,kwargs,len(batch["coords"]),rows=rows,pad_to=rows,
            executor=getattr(model,"_ug_gpu_executor",None))
    model.forward=types.MethodType(bounded_forward,model)
    encoding=model.gen.get_pocket_encoding
    def bounded_encoding(this,*args,**kwargs):
        coords=args[0] if args else kwargs["pocket_coords"]
        rows=getattr(model,"_ug_physical_rows",ROWS)
        return chunk_call(encoding,args,kwargs,len(coords),rows=rows,pad_to=rows,
            executor=getattr(model,"_ug_gpu_executor",None))
    model.gen.get_pocket_encoding=types.MethodType(bounded_encoding,model.gen)
