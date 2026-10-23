import json
import httpx
import pytest
from agenticgen.adaptive_schedule.io import read
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.providers.deepseek_api import call_json


@pytest.mark.parametrize('finish',['stop','length'])
@pytest.mark.parametrize('budget',[65536,98304])
def test_budget_is_sent_and_truncated_answer_is_never_accepted(tmp_path,monkeypatch,finish,budget):
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only')
    def handle(request):
        body=json.loads(request.content)
        assert body['max_tokens']==budget
        return httpx.Response(200,json={'id':'synthetic','model':'deepseek-flash',
            'choices':[{'finish_reason':finish,'message':{'content':'{"ok":true}' if finish=='stop' else '{"ok":'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':20}})
    output=tmp_path/'response.json'
    kwargs=dict(max_output_tokens=budget,transport=httpx.MockTransport(handle))
    if finish=='stop':
        assert call_json([{'role':'user','content':'synthetic'}],{'type':'object'},output,**kwargs)=={'ok':True}
        assert read(output.with_suffix('.receipt.json'))['max_output_tokens']==budget
    else:
        with pytest.raises(ValueError,match='Incomplete JSON'):
            call_json([{'role':'user','content':'synthetic'}],{'type':'object'},output,**kwargs)
        assert not output.exists()
        receipt=read(next(tmp_path.glob('*.incomplete_*.receipt.json')))
        assert receipt['finish_reason']=='length' and receipt['usage']['completion_tokens']==20


def test_empty_answer_with_reasoning_budget_exhaustion_is_archived(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only')
    def handle(request):
        assert json.loads(request.content)['max_tokens'] == 98304
        return httpx.Response(200,json={'id':'reasoning-only','model':'deepseek-flash',
            'choices':[{'finish_reason':'length','message':{'content':''}}],
            'usage':{'completion_tokens':98304,'completion_tokens_details':{'reasoning_tokens':98304}}})
    output=tmp_path/'response.json'
    for _ in range(2):
        with pytest.raises(ValueError,match='Incomplete JSON'):
            call_json([{'role':'user','content':'synthetic'}],{'type':'object'},output,
                      max_output_tokens=98304,transport=httpx.MockTransport(handle))
    assert not output.exists() and len(list(tmp_path.glob('*.incomplete_*.receipt.json'))) == 2
    receipt=read(output.with_suffix('.incomplete_1.receipt.json'))
    assert receipt['usage']['completion_tokens_details']['reasoning_tokens']==98304
    assert output.with_suffix('.incomplete_1.txt').read_bytes()==b''


@pytest.mark.parametrize('budget',[True,511,98305,393216])
def test_budget_above_authorized_96k_is_rejected(tmp_path,budget):
    with pytest.raises(ValueError,match='token budget'):
        call_json([],{'type':'object'},tmp_path/'response.json',max_output_tokens=budget)
