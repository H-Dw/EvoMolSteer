"""A local derivative needs adjacent converged stencils, not a lucky secant."""
import math
import numpy as np
import h5py
import pytest
from agenticgen.uniguide.derivative_acceptance import verify,EPSILONS
from agenticgen.uniguide.io import write


def objective(estimates,identities=None):
    def evaluate(offset):
        i=EPSILONS.index(abs(offset))
        return dict(value=4.+offset*estimates[i],identity=(identities or ['base']*4)[i])
    return evaluate


def test_bounded_refinement_preserves_failed_coarse_probes():
    # At the origin f'=1; large stencils see curvature, not the local tangent.
    result=verify(lambda x:dict(value=4.+x+1000*x**3,identity='base'),1.,'base')
    assert result['passed'] and result['accepted_epsilons']==[.005,.0025]
    assert result['probes'][0]['relative_error']>.05
    assert result['tolerance']==.05 and result['convergence_tolerance']==.025


@pytest.mark.parametrize('estimates,identities',[
    ([1.2]*4,None),  # consistently wrong AD
    ([1.2,1.,1.2,1.2],None),  # isolated lucky match
    ([1.04,.96,1.04,.96],None),  # both within 5%, but not converged
    ([1.]*4,['changed']*4),  # a different Hungarian branch
    ([1.]*4,['base','changed','base','changed']),
])
def test_rejects_wrong_unconverged_or_nonlocal_derivatives(estimates,identities):
    result=verify(objective(estimates,identities),1.,'base')
    assert not result['passed'] and len(result['probes'])==4


@pytest.mark.parametrize('value',[math.nan,math.inf,-math.inf])
def test_nonfinite_is_fatal_without_refinement(value):
    calls=[]
    def evaluate(x):calls.append(x);return dict(value=value,identity='base')
    result=verify(evaluate,1.,'base')
    assert not result['passed'] and len(calls)==2
    with pytest.raises(ValueError):verify(evaluate,value,'base')


@pytest.mark.parametrize('first',[1,10])
@pytest.mark.parametrize('fault',[None,'capped','healthy_changed','delta_changed'])
def test_uncapped_fixture_observes_exactly_the_first_active_update(tmp_path,monkeypatch,first,fault):
    from agenticgen.uniguide import geometry_batch_acceptance as acceptance
    from agenticgen.uniguide import guidance_endpoints
    monkeypatch.setattr(guidance_endpoints,'update_bounds',lambda c:[first,98])
    monkeypatch.setattr(acceptance,'scale_binding',lambda values,**kw:{acceptance.EFFECTIVE:values})
    monkeypatch.setattr(acceptance,'batch_initial',lambda *a:{})
    separate=tmp_path/'separate';separate.mkdir()
    with h5py.File(separate/'trace.h5','w') as trace:
        trace['coords']=np.zeros((100,10,2,3),dtype=np.float32)
    def sample(model,config,teacher,schedule,folder,count,**kw):
        assert count==20 and kw['stop_step']==first+1
        stress=kw['control_factory'].keywords['units'][1]['scale'][acceptance.EFFECTIVE]
        assert stress==[1e6]*4
        folder.mkdir();n=kw['stop_step']
        rows=[dict(max_atom_increment_A=[0.]*20) for _ in range(n)]
        rows[first]['max_atom_increment_A'][10]=2. if fault=='capped' else 4.
        write(folder/'steps.json',rows)
        with h5py.File(folder/'trace.h5','w') as trace:
            xyz=np.zeros((n,20,2,3),dtype=np.float32)
            if fault=='healthy_changed':xyz[first,0,0,0]=1.
            delta=np.zeros_like(xyz);delta[first,10,0,0]=4.
            trace['coords']=xyz;trace['guidance_delta_A']=delta
            if fault=='delta_changed':delta[first,10,0,0]=5.
            trace['unconstrained_guidance_delta_A']=delta.astype(np.float64)
    monkeypatch.setattr(acceptance.sampler,'sample',sample)
    units=[dict(id='healthy',slots=[0,1],scale={acceptance.EFFECTIVE:[1.]*4})]
    run=lambda:acceptance.verify_uncapped_guidance(None,{},dict(target='teacher'),{},tmp_path,units,separate)
    if fault:
        with pytest.raises(AssertionError):run()
    else:
        result=run()
        assert result['stress_step_index']==first and result['stress_steps']==first+1
