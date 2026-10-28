"""Bound physical network batches without changing the logical native integrator."""
import types
import torch
from .batching import take

ROWS=10


def concurrent_map(function,items,workers=1):
    """Keep GEMM/VJP shapes fixed while several units occupy the GPU concurrently."""
    items=list(items)
    if workers<=1 or len(items)<=1:return [function(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    device=torch.cuda.current_device();parent=torch.cuda.current_stream(device)
    enabled=torch.is_grad_enabled()
    def run(item):
        with torch.cuda.device(device),torch.set_grad_enabled(enabled):
            stream=torch.cuda.Stream(device=device);stream.wait_stream(parent)
            with torch.cuda.stream(stream):
                try:return function(item)
                finally:stream.synchronize()
    with ThreadPoolExecutor(max_workers=min(workers,len(items))) as pool:
        # All worker streams finish before returning or propagating an OOM.
        return list(pool.map(run,items))


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


def chunk_call(function,args,kwargs,count,rows=ROWS,pad_to=None,workers=1):
    if count<=rows and (pad_to is None or count==pad_to): return function(*args,**kwargs)
    def block(start):
        ids=list(range(start,min(start+rows,count)))
        actual=len(ids)
        if pad_to is not None:
            if actual>pad_to: raise ValueError("Physical rows exceed frozen padding shape")
            ids=(ids*((pad_to+actual-1)//actual))[:pad_to]
        output=function(*(take(a,ids,count) for a in args),**{k:take(v,ids,count) for k,v in kwargs.items()})
        return take(output,list(range(actual)),len(ids)),actual
    parts=concurrent_map(block,range(0,count,rows),workers)
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
            workers=getattr(model,"_ug_parallel_workers",1))
    model.forward=types.MethodType(bounded_forward,model)
    encoding=model.gen.get_pocket_encoding
    def bounded_encoding(this,*args,**kwargs):
        coords=args[0] if args else kwargs["pocket_coords"]
        rows=getattr(model,"_ug_physical_rows",ROWS)
        return chunk_call(encoding,args,kwargs,len(coords),rows=rows,pad_to=rows,
            workers=getattr(model,"_ug_parallel_workers",1))
    model.gen.get_pocket_encoding=types.MethodType(bounded_encoding,model.gen)
