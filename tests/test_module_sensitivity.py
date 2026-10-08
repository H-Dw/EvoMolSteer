from types import SimpleNamespace
import torch
from evomolsteer.generation.module_sensitivity import DecoderSensitivity

def test_hooks_record_existing_reverse_without_changing_values_or_gradient():
    decoder=torch.nn.Module();decoder.coord_emb=torch.nn.Linear(3,4,bias=False)
    model=SimpleNamespace(gen=SimpleNamespace(ligand_dec=decoder))
    x=torch.arange(12,dtype=torch.float32).reshape(1,4,3).requires_grad_(True)
    a=decoder.coord_emb(x);ga,=torch.autograd.grad(a.square().sum(),x)
    hooks=DecoderSensitivity(model)
    b=decoder.coord_emb(x);gb,=torch.autograd.grad(b.square().sum(),x)
    assert torch.equal(a,b) and torch.equal(ga,gb)
    assert hooks.records['gen.ligand_dec.coord_emb/0']['finite']
    hooks.close();assert not hooks.handles
