"""Bound physical network batches without changing the logical native integrator."""
import types
import torch
from .batching import take

ROWS=10


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
        return type(first)(concatenate([v[i] for v in values],sizes) for i in range(len(first)))
    if any(v!=first for v in values[1:]): raise ValueError("Non-row network metadata differs")
    return first


def chunk_call(function,args,kwargs,count,rows=ROWS):
    if count<=rows: return function(*args,**kwargs)
    values=[];sizes=[]
    for start in range(0,count,rows):
        ids=list(range(start,min(start+rows,count)))
        values.append(function(*(take(a,ids,count) for a in args),**{k:take(v,ids,count) for k,v in kwargs.items()}))
        sizes.append(len(ids))
    return concatenate(values,sizes)


def install(model):
    if hasattr(model,"_ug_original_forward"): raise ValueError("Chunking already installed")
    forward=model.forward
    model._ug_original_forward=forward
    def bounded_forward(this,*args,**kwargs):
        batch=args[0] if args else kwargs["batch"]
        return chunk_call(forward,args,kwargs,len(batch["coords"]))
    model.forward=types.MethodType(bounded_forward,model)
    encoding=model.gen.get_pocket_encoding
    def bounded_encoding(this,*args,**kwargs):
        coords=args[0] if args else kwargs["pocket_coords"]
        return chunk_call(encoding,args,kwargs,len(coords))
    model.gen.get_pocket_encoding=types.MethodType(bounded_encoding,model.gen)
