"""Mock transport compatibility, not an LLM replay or strategy-effect test."""
import json
import httpx
import jsonschema
import pytest
from agenticgen.providers.deepseek_api import call_json, LLMInterfaceError, FORMAT_POLICY
from agenticgen.adaptive_schedule.io import read


@pytest.mark.parametrize('final_valid',[True,False])
def test_contract_then_syntax_failure_gets_bounded_complete_replacement(tmp_path,monkeypatch,final_valid):
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only')
    seen=[]
    contents=['{"value":0}', '{"value":1,{"value":2}]}', '{"value":2}' if final_valid else '{"value":0}']
    def handle(request):
        body=json.loads(request.content);seen.append(body)
        assert body['max_tokens']==98304 and body['response_format']=={'type':'json_object'}
        n=len(seen)
        if n==2:assert 'already executed' in body['messages'][-1]['content']
        if n==3:
            assert 'JSONDecodeError' in body['messages'][-1]['content']
            assert 'complete replacement object' in body['messages'][-1]['content']
        return httpx.Response(200,json={'id':f'synthetic-{n}','model':'deepseek-flash',
            'choices':[{'finish_reason':'stop','message':{'content':contents[n-1]}}],
            'usage':{'prompt_tokens':n,'completion_tokens':10*n}})
    def validator(result):
        if result['value']==0:raise jsonschema.ValidationError('Schedule already executed')
    output=tmp_path/'response.json'
    kwargs=dict(max_format_repairs=2,max_output_tokens=98304,response_validator=validator,
                transport=httpx.MockTransport(handle))
    schema={'type':'object','required':['value'],'properties':{'value':{'type':'integer'}},'additionalProperties':False}
    if final_valid:
        assert call_json([{'role':'user','content':'synthetic'}],schema,output,**kwargs)=={'value':2}
        assert len(read(output.with_suffix('.receipt.json'))['interface_repairs'])==2
    else:
        with pytest.raises(LLMInterfaceError):call_json([{'role':'user','content':'synthetic'}],schema,output,**kwargs)
        assert not output.exists()
    assert len(seen)==3
    invalid=list(tmp_path.glob('*.invalid.receipt.json'))
    assert len(invalid)==(2 if final_valid else 3)
    assert {read(p)['provider_request_id'] for p in invalid}=={f'synthetic-{i+1}' for i in range(len(invalid))}
    assert all(read(p)['format_policy']==FORMAT_POLICY for p in invalid)
    assert (tmp_path/'response.attempt_2.invalid.txt').read_text()==contents[1]


def test_truncation_still_stops_without_using_extra_format_budget(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only');seen=[]
    def handle(request):
        seen.append(1)
        return httpx.Response(200,json={'id':'synthetic-length','model':'deepseek-flash',
            'choices':[{'finish_reason':'length','message':{'content':'{"value":'}}],
            'usage':{'completion_tokens':98304}})
    with pytest.raises(ValueError,match='Incomplete JSON'):
        call_json([{'role':'user','content':'synthetic'}],{'type':'object'},tmp_path/'r.json',
                  max_format_repairs=2,max_output_tokens=98304,transport=httpx.MockTransport(handle))
    assert len(seen)==1 and not (tmp_path/'r.json').exists()
