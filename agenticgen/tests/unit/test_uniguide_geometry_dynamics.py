"""Remote engineering contracts for raw999 and per-molecule direction-preserving caps."""
import inspect
import textwrap
from types import SimpleNamespace
import pytest
import torch
from agenticgen.uniguide.geometry_dynamics import policy,limit_increment,Raw999Sampling,LIMIT,WINDOWS
from agenticgen.uniguide.geometry_protocol import expand_effective,displacement


def config(**kw):return dict(output_policy="raw999",**{LIMIT:3.},**kw)


@pytest.mark.parametrize("cap",[.25,3.,8.])
def test_candidate_limit_keeps_direction_healthy_rows_and_rng(cap):
    delta=torch.tensor([[[.03,.04,0.],[.01,0.,0.]],[[3.,4.,0.],[0.,0.,2.]]])
    original=delta.clone();mask=torch.ones(2,2,dtype=torch.bool);rng=torch.get_rng_state().clone()
    bounded,audit=limit_increment(delta,mask,2.,cap)
    assert torch.equal(bounded[0],delta[0]) and torch.equal(delta,original)
    assert bounded.double().norm(dim=-1).amax()*2<=cap
    factor=audit["guidance_increment_scale_factor"][1]
    assert 0<factor<1 and torch.allclose(bounded[1],delta[1]*factor)
    assert audit["guidance_increment_limited"]==[False,True]
    assert torch.equal(rng,torch.get_rng_state())
    separate=torch.cat([limit_increment(delta[i:i+1],mask[i:i+1],2.,cap)[0] for i in range(2)])
    assert torch.equal(bounded,separate)


def test_large_finite_vector_mask_zero_and_exact_boundary():
    delta=torch.tensor([[[1e30,1e30,1e30],[float('nan'),0.,0.]],[[3.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]]])
    mask=torch.tensor([[1,0],[1,1],[1,1]],dtype=torch.bool)
    bounded,audit=limit_increment(delta,mask,1.,3.)
    assert torch.isfinite(bounded).all() and bounded.double().norm(dim=-1).max()<=3
    assert torch.equal(bounded[1:],delta[1:]) and torch.equal(bounded[0,1],torch.zeros(3))
    assert audit["unconstrained_max_atom_increment_A"][0]>1e30
    assert audit["guidance_increment_limited"]==[True,False,False]
    with pytest.raises(FloatingPointError):limit_increment(delta,torch.ones_like(mask),1.,3.)


@pytest.mark.parametrize("bad",[None,0,-1,True,float('inf'),float('nan'),'3'])
def test_config_requires_an_explicit_positive_limit(bad):
    with pytest.raises(ValueError):policy(dict(output_policy='raw999',**{LIMIT:bad}))


def test_window_and_output_contract():
    assert policy(config())[LIMIT]==3
    scales=expand_effective([1,2,3,4])
    assert scales[:10]==[0]*10 and scales[10:26]==[1]*16 and scales[76:]==[4]*24
    assert WINDOWS[-1]==(76,99)
    for change in (dict(output_policy='native_final_corrector'),dict(windows=[[10,25],[26,50],[51,75],[76,90]])):
        with pytest.raises(ValueError):policy(dict(config(),**change))


def test_calibration_observes_unconstrained_growth():
    rows=[dict(increment_vector_rms_A=.5,max_atom_increment_A=3.,
        unconstrained_increment_vector_rms_A=50.,unconstrained_max_atom_increment_A=1000.)]*4
    assert displacement(dict(window_diagnostics=rows))==dict(increment_vector_rms_A=50.,max_atom_increment_A=1000.)


class Dummy:
    def __init__(self):self.final_calls=0
    def _get_predictions(self,x):return x,None
    def _generate_selective(self,steps=100):
        time_points=torch.linspace(0,1,steps+1)
        step_sizes=[t1-t0 for t0,t1 in zip(time_points[:-1],time_points[1:])]
        times=[torch.zeros(1)]
        for i,step_size in enumerate(step_sizes):
            out_target={'coords':torch.tensor([float(i)])}
            predicted_target,cond_batch=self._get_predictions(out_target)
            predicted_untarget,_=self._get_predictions(out_target)
            predicted_target['coords'].fill_(-100.)
            times=[times[0]+step_size]
        eps=-1e-4
        self.final_calls+=1
        predicted_target={k:v.cpu().detach() for k,v in predicted_target.items()}
        return predicted_target


def test_geometry_uses_last_uncovered_prediction_and_rejects_changed_source(tmp_path):
    model=Dummy();controller=Raw999Sampling();controller.model=model;model._adaptive=controller
    controller.folder=tmp_path;controller.resume=None
    source=tmp_path/'provenance/native';source.mkdir(parents=True)
    (source/'instrumented_selective.py').write_text(textwrap.dedent(inspect.getsource(Dummy._generate_selective)))
    result=controller.sampling_function(dict(config(),root=str(tmp_path)))()
    assert result['coords'].item()==99. and model.final_calls==0
    assert controller.last_loop['network_times']==pytest.approx([.98901],abs=1e-6)
    assert controller.raw_output_checkpoint()['raw_output_prediction']['target']['coords'].item()==99.
    controller.write_output_policy()
    (source/'raw999_selective.py').write_text('changed')
    with pytest.raises(ValueError):controller.sampling_function(dict(config(),root=str(tmp_path)))
