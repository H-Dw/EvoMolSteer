"""Full/drop dispatch, independent confirmation and immutable historical plans."""
import csv
import importlib
import shutil
from pathlib import Path

import pytest

from evomolsteer.ablation.protocol import (
    PLAN,FULL_DROP_PLAN,create_full_drop_plan,reserved_batches,validate_full_drop_protocol,validate_plan)
from evomolsteer.ablation.statistics import classify_module
from evomolsteer.ablation.modules import assemble
from evomolsteer.io import digest,read_json

REPO=Path(__file__).resolve().parents[1]


def test_new_plan_has_only_full_drop_and_keeps_frozen_history(tmp_path):
    shutil.copytree(REPO/PLAN,tmp_path/PLAN)
    before={p.relative_to(tmp_path):digest(p) for p in (tmp_path/PLAN).rglob('*') if p.is_file()}
    root=create_full_drop_plan(tmp_path)
    validate_plan(tmp_path,FULL_DROP_PLAN)
    assert all(digest(tmp_path/p)==sha for p,sha in before.items())
    p=read_json(root/'protocol.json');validate_full_drop_protocol(p)
    assert p['evaluation']['confirmation_families']==68
    assert len(reserved_batches(p))==300 and min(reserved_batches(p))==4100
    assert p['temperature_policy']['parameters']==read_json(tmp_path/PLAN/'protocol.json')['temperature_policy']['parameters']
    rows=list(csv.DictReader((root/'conditions.csv').open(encoding='utf-8')))
    arms=[x for x in rows if x['role']!='none']
    assert len(arms)==34 and {x['intervention'] for x in arms}=={'full','drop'}
    for mid in p['module_ids']:
        assert {x['intervention'] for x in arms if x['module']==mid}=={'full','drop'}
    modules=read_json(root/'module_registry.json')['modules']
    assert all('alternative_instruction' not in m for m in modules)
    for role in ['Analyst','Designer']:
        ids=[m['id'] for m in modules if m['role']==role]
        assert len(assemble(modules,role,ids))>0
    assert create_full_drop_plan(tmp_path)==root


def test_runner_rejects_legacy_three_arm_activation():
    with pytest.raises(ValueError,match='full/drop-only'):
        validate_full_drop_protocol(read_json(REPO/PLAN/'protocol.json'))


@pytest.mark.parametrize('ci,result',[
    ([.06,.10],'AFFINITY_NECESSARY_IN_THIS_CONTEXT'),
    ([-.01,.01],'PERFORMANCE_EQUIVALENT_IN_THIS_CONTEXT'),
    ([-.10,-.03],'NEGATIVE_IN_THIS_CONTEXT'),
    ([-.10,.10],'INCONCLUSIVE')])
def test_two_arm_classification_needs_no_replacement_result(ci,result):
    assert classify_module({'simultaneous_CI':ci},functional_valid=True)==result


@pytest.fixture
def runner(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    mod=importlib.import_module('run_scientific_ablation_study')
    value=mod.ScientificStudy(tmp_path,{'study':'study','config_output':'config'})
    value.modules=[{'id':'A1.1','role':'Analyst','lane':'G'}]
    value.protocol=read_json(REPO/FULL_DROP_PLAN/'protocol.json')
    value.state.update(pruning=[],conditions={})
    monkeypatch.setattr(value,'commit',lambda *a:None)
    monkeypatch.setattr(value,'write_report',lambda:None)
    return value


@pytest.mark.parametrize('final_context',[False,True])
def test_initial_and_final_context_dispatch_only_full_and_drop(runner,monkeypatch,final_context):
    seen=[];batches=[]
    def condition(case,role,active,incumbent,**kwargs):
        assert 'alternative' not in kwargs
        seen.append((case,active))
        return {'case':case,'functional':{'assessments':[{'module':'A1.1','score':2} for _ in range(6)]}}
    def frames(c,b):batches.append(b);return ['frame']
    def contrast(*a,**kw):return {'delta_pic50':0.,'simultaneous_CI':[0.,0.]}
    monkeypatch.setattr(runner,'condition',condition)
    monkeypatch.setattr(runner,'condition_frames',frames)
    monkeypatch.setattr(runner,'controls',lambda b:{})
    monkeypatch.setattr(runner,'frames',lambda *a:'frame')
    monkeypatch.setattr(runner,'contrast',contrast)
    monkeypatch.setattr(runner,'strain_contrast',lambda *a,**kw:{'status':'UNAVAILABLE'})
    mod=importlib.import_module('run_scientific_ablation_study')
    monkeypatch.setattr(mod,'summarize',lambda *a:{})
    monkeypatch.setattr(mod,'quality_envelope',lambda *a:{'passed':True})
    if final_context:
        runner._final_context_step(['A1.1'],[],{},[],'A1.1',2)
        assert all(b==list(range(5032,5040)) for b in batches)
        record=runner.state['final_context'][0]
    else:
        assert runner._sequential_step('A','Analyst',['A1.1'],{},'A1.1',0)==[]
        assert list(range(4200,4208)) in batches
        record=runner.state['pruning'][0]
    assert [x[0].rsplit('_',1)[-1] for x in seen]==['full','drop']
    assert seen[0][1]==['A1.1'] and seen[1][1]==[]
    assert not any('alternative' in k for k in record)


@pytest.mark.parametrize('case',['A_A1.1_alternative','A_empty','D-G_empty','D-U_empty'])
def test_undeclared_conditions_cannot_reach_the_api(runner,case):
    with pytest.raises(ValueError,match='Only full/drop'):
        runner.condition(case,'Analyst',[],{})
