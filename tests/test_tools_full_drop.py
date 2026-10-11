import csv
import shutil
from pathlib import Path

import pytest

from evomolsteer.ablation.protocol import (PLAN, FULL_DROP_PLAN, TOOLS_FULL_DROP_PLAN,
    create_tools_full_drop_plan, validate_plan, reserved_batches)
from evomolsteer.agent_workflow.ablation_adapter import analyst_source
from evomolsteer.agent_workflow.diagnostics import fixture_sources
from evomolsteer.agent_workflow.sources import EvidenceTools
from evomolsteer.io import digest, read_json, write_json

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / 'docs/experiments/m2_full_drop_ablation_20261012'


def test_uniform_plan_preserves_history_and_scientific_controls(tmp_path):
    for source in [PLAN, FULL_DROP_PLAN, 'skills/agent-json-interface']:
        shutil.copytree(REPO/source, tmp_path/source)
    before = digest(tmp_path/FULL_DROP_PLAN/'protocol.json')
    root = create_tools_full_drop_plan(tmp_path)
    validate_plan(tmp_path, TOOLS_FULL_DROP_PLAN)
    p = read_json(root/'protocol.json')
    old = read_json(tmp_path/FULL_DROP_PLAN/'protocol.json')
    assert digest(tmp_path/FULL_DROP_PLAN/'protocol.json') == before
    assert p['agent_interface'] == 'evidence-tools-v1'
    assert p['baseline'] == old['baseline']
    assert p['temperature_policy'] == old['temperature_policy']
    assert p['comparison_policy']['previous_study_results_used_for_selection'] is False
    assert set(reserved_batches(p)).isdisjoint(reserved_batches(old))
    assert len(reserved_batches(p)) == 300
    rows = list(csv.DictReader((root/'conditions.csv').open(encoding='utf-8')))
    assert len(rows) == 36
    assert {r['intervention'] for r in rows if r['role'] != 'none'} == {'full','drop'}
    assert p['budgets']['maximum_api_role_calls'] > 10*642 + 300
    assert create_tools_full_drop_plan(tmp_path) == root


@pytest.mark.parametrize('mode',['observed','sign_changed','missing'])
def test_diagnostic_sources_isolated_and_self_consistent(tmp_path,mode):
    sha = digest(OLD/'evidence/packet.json')
    tools = EvidenceTools(fixture_sources(OLD,tmp_path,contrast=mode))
    assert tools.packet['diagnostic_only'] is True
    assert digest(OLD/'evidence/packet.json') == sha
    assert set(tools.records) == set(tools.index['ids'])
    original = EvidenceTools(OLD/'evidence')
    assert len(tools.records) == (0 if mode == 'missing' else len(original.tables['regional_effects']))
    if mode == 'sign_changed':
        key = next(iter(tools.records))
        assert tools.records[key]['effect_z'] == -original.records[key]['effect_z']


def test_unit_fixture_changes_representation_not_physical_scale(tmp_path):
    a = EvidenceTools(fixture_sources(OLD,tmp_path/'a',unit='A'))
    nm = EvidenceTools(fixture_sources(OLD,tmp_path/'nm',unit='nm'))
    assert a.packet['tables'] == nm.packet['tables'] == {}
    assert a.packet['teacher_modes']['assignment_rms_quantiles'] == pytest.approx(
        [10*x for x in nm.packet['teacher_modes']['assignment_rms_quantiles']])


def fake_workflow(study):
    folder = study/'calls/selected_Analyst_fresh/chain_00'
    nested = folder/'Analyst.tools_v1'
    response = {'role':'Analyst','findings':[],'hypotheses':[],'limitations':[]}
    write_json(folder/'Analyst.json', response)
    write_json(nested/'Analyst.json', response)
    write_json(nested/'binding.json', {'source':'test fixture'})
    write_json(nested/'references.json', {'sources':{},'references':{}})
    write_json(nested/'completed.json', {'backend':'real_api','response_sha256':digest(nested/'Analyst.json'),
        'binding_sha256':digest(nested/'binding.json'),'references_sha256':digest(nested/'references.json')})
    relative = Path('docs/experiments')/study.name/'calls/selected_Analyst_fresh/chain_00/Analyst.json'
    write_json(study/'selected_Analyst_pool.json', {'outputs':[
        {'path':relative.as_posix(),'sha256':digest(folder/'Analyst.json')} for _ in range(6)]})
    return folder, response


def test_designer_cross_condition_uses_verified_frozen_source(tmp_path):
    study = tmp_path/'study'
    folder,response = fake_workflow(study)
    assert analyst_source(study,'D-G_D1.1_drop',0,response) == folder/'Analyst.tools_v1/Analyst.json'
    assert not (study/'calls/D-G_D1.1_drop/chain_00/Analyst.json').exists()
    write_json(folder/'Analyst.tools_v1/references.json',{'changed':True})
    with pytest.raises(ValueError,match='verified actual API'):
        analyst_source(study,'D-G_D1.1_drop',0,response)


def test_frozen_handoff_rejects_wrong_response_or_stale_hash(tmp_path):
    study = tmp_path/'study'
    folder,response = fake_workflow(study)
    with pytest.raises(ValueError,match='differs'):
        analyst_source(study,'D-G_full',0,{**response,'limitations':['fake']})
    write_json(folder/'Analyst.json',{**response,'limitations':['changed']})
    with pytest.raises(ValueError,match='changed'):
        analyst_source(study,'D-G_full',0,response)


def test_condition_names_do_not_reveal_deletion_to_role_and_two_wordings_delivered(tmp_path,monkeypatch):
    import evomolsteer.agent_workflow.ablation_adapter as adapter
    import evomolsteer.ablation.agents as agents
    import evomolsteer.ablation.protocol as protocol
    study=tmp_path/'study'
    for name in ['packet.json','index.json','continuous_functions.json','observable_definitions.json']:
        destination=study/'evidence'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OLD/'evidence'/name,destination)
    monkeypatch.setattr(protocol,'validate_activation',lambda *a:{'protocol':{
        'temperature_policy':{'parameters':{}},'lanes':{'G':{}}},'packet_sha256':'test'})
    monkeypatch.setattr(agents,'role_prompt',lambda *a:'Actual active modules only')
    tasks=[]
    def capture(role,tools,prompt,task,*args,**kwargs):
        tasks.append(task)
        return {'role':'Analyst','findings':[],'hypotheses':[],'limitations':[]}
    monkeypatch.setattr(adapter,'run_agent',capture)
    adapter.call_role_tools(REPO,study,'A_A1.1_full',0,'Analyst',[])
    adapter.call_role_tools(REPO,study,'A_A1.1_drop',3,'Analyst',[])
    assert tasks[0]['task_wording'] != tasks[1]['task_wording']
    for task in tasks:
        assert 'case' not in task and len(task['case_id']) == 16
        assert 'A1.1' not in str(task) and '_drop' not in str(task) and '_full' not in str(task)
