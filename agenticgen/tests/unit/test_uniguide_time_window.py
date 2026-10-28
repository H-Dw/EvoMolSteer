"""Remote-only experiment-contract tests; no fixture is scientific evidence."""
import inspect
import textwrap
import types

import pytest
import torch

from agenticgen.uniguide import time_window_experiment as experiment
from agenticgen.uniguide.io import read,write
from agenticgen.uniguide.condition import guidance_delta
from test_uniguide_joint_endpoint import Dummy


def test_factorial_contains_nine_new_and_three_reused_arms_per_teacher():
    all_arms=experiment.arms()
    reused={'gradient__native1__window','chemical__native1__window','chemical__raw999__window'}
    assert len(all_arms)==12 and len(set(all_arms)-reused)==9
    assert all(p['chemistry'] in ('gradient','chemical') for p in all_arms.values())


def test_full_coverage_extends_edges_without_changing_original_window():
    schedule=[1.,2.,3.,4.]
    old=experiment.expand_coverage(schedule,'window')
    new=experiment.expand_coverage(schedule,'full_interior')
    assert old[10:91]==new[10:91] and old[0]==new[0]==0
    assert new[1:10]==[1.]*9 and new[91:]==[4.]*9
    assert sum(bool(x) for x in old)==81 and sum(bool(x) for x in new)==99
    with pytest.raises(ValueError):experiment.expand_coverage(schedule,'literal_closed_interval')


@pytest.mark.parametrize('endpoint,time,last,correctors',[
    ('native1',.99,999.,1),('raw1',.99,99.,0),('raw999',.98901,99.,0)])
def test_grid_and_output_are_independently_identified(endpoint,time,last,correctors):
    source=textwrap.dedent(inspect.getsource(Dummy._generate_selective))
    function,rendered=experiment.compile_policy(Dummy()._generate_selective,source,endpoint)
    model=Dummy();result=types.MethodType(function,model)()
    assert result['coords'].item()==last and model.final_calls==correctors
    assert len(model._adaptive.raw_times)==100
    assert model._adaptive.raw_times[-1]==pytest.approx(time,abs=1e-6)


def test_endpoint_zero_remains_excluded_in_original_vjp_formula():
    gradient=torch.ones(1,2,3);valid=torch.ones(1,2)
    zero,_=guidance_delta(gradient,torch.tensor([0.]),.01,[0.],0.,valid)
    assert not zero.any()
    with pytest.raises(ValueError,match='undefined'):
        guidance_delta(gradient,torch.tensor([0.]),.01,[1.],0.,valid)
    early,_=guidance_delta(gradient,torch.tensor([.01]),.01,[1.],0.,valid)
    assert torch.isfinite(early).all() and (early<0).all()


def test_budget_preserves_failed_attempts_and_rejects_unlisted_or_rebound_groups(tmp_path):
    reused={'gradient__native1__window','chemical__native1__window','chemical__raw999__window'}
    new=sorted(set(experiment.arms())-reused)
    write(tmp_path/'STUDY.json',dict(controls={'a':{},'b':{}},new_arms=new))
    budget=experiment.StudyBudget(tmp_path)
    for t in ['a','b']:
        for arm in new:budget.reserve(f'groups/{t}/{arm}','formal',50,arm)
    budget.reserve('groups/a/'+new[0],'formal',50,new[0])
    assert sum(r['count'] for r in read(tmp_path/'attempts.json').values())==900
    with pytest.raises(ValueError):budget.reserve('groups/a/'+new[0],'formal',50,'changed')
    with pytest.raises(ValueError):budget.reserve('groups/a/gradient__native1__window','formal',50,'reused')


def test_paired_seed_statistics_keep_all_slots_and_do_not_impute_failures():
    base=[dict(logical_seed=s,logical_slot=i,condition_rmsd_A=1.) for s in experiment.SEEDS for i in range(10)]
    treatment=[dict(r,condition_rmsd_A=1.2) for r in base]
    result=experiment.paired_statistics(base,treatment,'condition_rmsd_A')
    assert result['mean_delta']==pytest.approx(.2) and result['clusters']==5 and result['paired_slots']==50
    assert result['practical_rmsd_difference'] and result['interval_excludes_zero']
    assert experiment.paired_statistics(base,treatment[:-1],'condition_rmsd_A')['status']=='unavailable_incomplete_population'
    treatment[0]['condition_rmsd_A']=float('nan')
    assert experiment.paired_statistics(base,treatment,'condition_rmsd_A')['status']=='unavailable_nonfinite_or_unmeasured'


def test_no_chemical_path_uses_bound_scientific_control_when_sampler_constructor_is_wrapped(monkeypatch):
    control=object.__new__(experiment.TimeWindowControl)
    control.chemistry='gradient';control.reference=None
    monkeypatch.setattr(experiment.FrozenControl,'after',lambda self,value,step:dict(value,step=step))
    monkeypatch.setattr(experiment.sampler,'Control',lambda *a,**k:None)
    assert control.after({'x':1},5)=={'x':1,'step':5}
