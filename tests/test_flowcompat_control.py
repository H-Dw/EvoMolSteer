import torch
from evomolsteer.generation.flowcompat_control import control_geometry

def test_null_and_flow_decomposition():
    g=torch.tensor([[[1.,2.,3.],[2.,1.,0.]]]);v=torch.ones_like(g);mask=torch.ones((1,2),dtype=torch.bool)
    same,h,_=control_geometry(g,v,mask,.3,[.2,.6],{})
    assert same is g and torch.equal(h,torch.ones(1))
    adjusted,h,_=control_geometry(g,v,mask,.3,[.2,.6],{'parallel_component_scale':0.})
    assert abs(float((adjusted*v).sum()))<1e-6
    assert torch.allclose(adjusted+(g*v).sum()/v.square().sum()*v,g)

def test_dynamic_window_and_gain_gate():
    g=torch.ones((1,2,3));m=torch.ones((1,2),dtype=torch.bool)
    _,a,_=control_geometry(g,g,m,.25,[0,.5],{'time_envelope_power':1.})
    _,b,_=control_geometry(g,g,m,.5,[0,1],{'time_envelope_power':1.})
    assert torch.equal(a,b) and float(a)==.5
    _,gate,_=control_geometry(g,g,m,0,[0,.5],{'jacobian_gain_saturation':1.},torch.full((1,),3**.5))
    assert torch.allclose(gate,torch.tensor([.5]))
