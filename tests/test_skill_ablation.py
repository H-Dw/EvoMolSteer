import copy
from pathlib import Path
import pytest
from evomolsteer.io import read_json,write_json,digest
from evomolsteer.skill_guidance import render,modules
from evomolsteer.continuous.skill_ablation import export_request,validate_response,compile_response,effective_signature,variants,instruction

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'docs/experiments/skill_ablation_20261007/study_v1'


def request(tmp_path):
    v=next(v for v in variants() if v['id']=='compact_compact')
    a=export_request(ROOT,STUDY/'evidence.json',tmp_path,v,'Analyst')
    response={'agent':'Analyst','request_sha256':digest(a),'observations':[], 'hypotheses':[], 'counterevidence':[], 'limitations':[]}
    write_json(tmp_path/'Analyst.response.json',response)
    return export_request(ROOT,STUDY/'evidence.json',tmp_path,v,'Designer',tmp_path/'Analyst.response.json')


def response(req):
    packet=read_json(STUDY/'evidence.json')
    return {'agent':'Designer','request_sha256':digest(req),'analyst_response_sha256':read_json(req)['bindings']['analyst_response_sha256'],
        'decision':'modify_existing','base_program_id':packet['incumbent_program_id'],
        'updates':{'native_rms_ratio':.27},'evidence_ids':['quality:'+packet['incumbent_program_id']],
        'rationale':'Declared dose counterfactual','failure_modes':['Possible regression']}


def test_literal_choice_is_not_repaired_to_historical_winner(tmp_path):
    req=request(tmp_path);r=response(req);path=tmp_path/'Designer.response.json';write_json(path,r)
    program=compile_response(ROOT,req,path,STUDY/'evidence.json',tmp_path/'compiled.json')
    assert program['native_rms_ratio']==.27
    # A different valid source choice is legal even when it is worse historically.
    packet=read_json(STUDY/'evidence.json');r['base_program_id']=next(k for k,v in packet['program_registry'].items() if v['reward_view']=='endpoint_direction')
    write_json(path,r);assert validate_response(req,path,STUDY/'evidence.json')


@pytest.mark.parametrize('edit',[
    {'request_sha256':'wrong'}, {'analyst_response_sha256':'wrong'}, {'base_program_id':'invented'},
    {'updates':{'native_rms_ratio':float('inf')}}, {'updates':{'affinity_head_gradient':True}},
    {'decision':'retain_existing'}])
def test_execution_contract_without_expected_answer(tmp_path,edit):
    req=request(tmp_path);r=response(req);r.update(edit);path=tmp_path/'Designer.response.json'
    if edit.get('updates',{}).get('native_rms_ratio')==float('inf'):
        path.write_text(__import__('json').dumps(r))
    else:write_json(path,r)
    with pytest.raises((ValueError,TypeError)):validate_response(req,path,STUDY/'evidence.json')


def test_advice_removal_changes_actual_instructions_not_contract():
    conditions={v['id']:v for v in variants()}
    advice=read_json(STUDY/'treatment_modules.json')
    for role in ('Analyst','Designer'):
        full=instruction(role,conditions['full_advice'],ROOT);core=instruction(role,conditions['compact_compact'],ROOT)
        assert full!=core
        for text in advice[role].values():assert text not in core
        with pytest.raises(ValueError):render(role,['unknown'])


def test_deployment_has_only_canonical_core_and_no_retired_advice():
    for role in ('Analyst','Designer'):
        assert modules(role)=={}
        assert render(role)==instruction(role,{'id':'standard_core','analyst':'compact','designer':'compact'},ROOT)


def test_unknown_execution_fields_prevent_false_deduplication():
    p=read_json(ROOT/'configs/experiments/ck2_affinity_geometry30_v1/backtrack_round26.json')
    altered=copy.deepcopy(p);altered['future_new_loss_parameter']=.2
    assert effective_signature(p)[0]!=effective_signature(altered)[0]
    altered=copy.deepcopy(p);altered['program_id']='different provenance'
    assert effective_signature(p)[0]==effective_signature(altered)[0]


def test_api_transport_loads_actual_instructions_and_bound_analyst(tmp_path,monkeypatch):
    import evomolsteer.agents as api
    req=request(tmp_path);r=response(req);captured={}
    class Result:
        def raise_for_status(self):pass
        def json(self):return {'choices':[{'message':{'content':__import__('json').dumps(r)}}]}
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,url,**kwargs):captured.update(kwargs['json']);return Result()
    monkeypatch.setattr(api.httpx,'Client',Client)
    for key,value in [('EVOMOLSTEER_BASE_URL','https://test.invalid'),('EVOMOLSTEER_MODEL','simulator'),('EVOMOLSTEER_API_KEY','unit-test-placeholder')]:monkeypatch.setenv(key,value)
    result=api.call_api(req,tmp_path)
    assert result['updates']=={'native_rms_ratio':.27}
    assert read_json(tmp_path/'reward_program.json')['native_rms_ratio']==.27
    assert read_json(req)['system_instruction'] in captured['messages'][0]['content']
    assert 'analyst_document' in captured['messages'][1]['content']
    assert captured['response_format']=={'type':'json_object'}
