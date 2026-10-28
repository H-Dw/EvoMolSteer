"""Remote-only contract checks; synthetic data is not experimental evidence."""
import inspect
import textwrap
import types

import pytest
import torch

from agenticgen.uniguide import late_window_experiment as late
from agenticgen.uniguide import time_window_experiment as base
from agenticgen.uniguide.io import read, write
from test_uniguide_joint_endpoint import Dummy


def test_late_extension_is_only_steps_91_through_99():
    values=[181.,724.,181.,64.]
    reference=base.expand_coverage(values,'window')
    extended=base.expand_coverage(values,'late_only')
    assert extended[:91]==reference[:91]
    assert extended[:10]==[0.]*10
    assert extended[91:100]==[64.]*9
    assert [i for i in range(100) if reference[i]!=extended[i]]==list(range(91,100))
    assert len(extended)==100 and sum(bool(x) for x in extended)==90


def test_only_four_new_arms_per_teacher_and_no_native_output():
    arms=late.arms()
    assert len(arms)==12
    assert sum(v['coverage']=='late_only' for v in arms.values())==4
    assert {v['endpoint'] for v in arms.values()}=={'raw1','raw999'}
    for a,b,label in late.contrasts():
        assert a in arms and b in arms
    assert len(late.contrasts())==14


def test_four_hundred_attempt_ledger_cannot_rerun_or_change_binding(tmp_path):
    new=[k for k,v in late.arms().items() if v['coverage']=='late_only']
    write(tmp_path/'STUDY.json',dict(policy=late.POLICY,controls={'a':{},'b':{}},new_arms=new,new_attempt_cap=400))
    budget=base.StudyBudget(tmp_path)
    for t in ['a','b']:
        for arm in new:budget.reserve(f'groups/{t}/{arm}','formal',50,arm)
    budget.reserve('groups/a/'+new[0],'formal',50,new[0])
    ledger=read(tmp_path/'attempts.json')
    assert len(ledger)==8 and sum(r['count'] for r in ledger.values())==400
    assert all(r['stage']==late.POLICY for r in ledger.values())
    with pytest.raises(ValueError):budget.reserve('groups/a/'+new[0],'formal',50,'replacement')
    with pytest.raises(ValueError):budget.reserve('groups/a/gradient__raw1__window','formal',50,'control')
    with pytest.raises(ValueError):budget.reserve('groups/a/'+new[0],'formal',10,'small_retry')
    # Corrupted or oversized ledgers cannot authorize additional samples.
    ledger.pop('groups/b/'+new[-1]);ledger['unknown_failed_group']={'count':50}
    write(tmp_path/'attempts.json',ledger)
    with pytest.raises(ValueError,match='400-attempt'):
        budget.reserve('groups/b/'+new[-1],'formal',50,new[-1])


@pytest.mark.parametrize('endpoint,last_time',[('raw1',.99),('raw999',.98901)])
def test_last_uncovered_prediction_is_returned_without_terminal_forward(endpoint,last_time):
    model=Dummy();source=textwrap.dedent(inspect.getsource(Dummy._generate_selective))
    compiled,_=base.compile_policy(model._generate_selective,source,endpoint)
    output=types.MethodType(compiled,model)()
    assert model.final_calls==0 and output['coords'].item()==99.
    assert len(model._adaptive.raw_times)==100
    assert model._adaptive.raw_times[-1]==pytest.approx(last_time,abs=1e-6)


def test_report_keeps_failed_reference_unavailable_instead_of_resampling(tmp_path):
    ref=tmp_path/'old_failure';ref.mkdir()
    write(ref/'FAILED.json',dict(scientific_failure=True,charged_attempts=50))
    a='gradient__raw1__full_interior';b='gradient__raw1__late_only'
    write(tmp_path/'STUDY.json',dict(policy=late.POLICY,controls={'teacher':{}},
        arms={a:{},b:{}},references={'teacher':{a:str(ref)}},protected_files={},
        reused_attempts=0,referenced_failed_attempts=50,statistics={},
        contrasts=[(b,a,'early_extension_with_late_fixed')]))
    base.report(tmp_path)
    result=read(tmp_path/'RESULTS.json')
    assert result['new_attempts']==0 and result['referenced_failed_attempts']==50
    assert result['groups']['teacher/'+a]['reused']
    assert result['groups']['teacher/'+a]['status']=='failed'
    assert result['comparisons'][0]['paired_metrics'] is None
    assert not (tmp_path/'attempts.json').exists()


def test_raw_return_capture_precedes_step_update_in_compiled_program():
    source=textwrap.dedent(inspect.getsource(Dummy._generate_selective))
    _,rendered=base.compile_policy(Dummy()._generate_selective,source,'raw999')
    assert 'last_loop_target' in rendered
    assert rendered.index('capture_loop_prediction')<rendered.index('predicted_target = self._adaptive.last_loop_target')
