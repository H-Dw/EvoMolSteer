import json
import httpx
import pytest
from agenticgen.adaptive_schedule.io import read
from agenticgen.adaptive_schedule.group_protocol import configuration
from agenticgen.providers.deepseek_api import call_json


@pytest.mark.parametrize('finish',['stop','length'])
def test_64k_is_sent_and_truncated_answer_is_never_accepted(tmp_path,monkeypatch,finish):
    monkeypatch.setenv('DEEPSEEK_API_KEY','synthetic-test-only')
    def handle(request):
        body=json.loads(request.content)
        assert body['max_tokens']==65536
        return httpx.Response(200,json={'id':'synthetic','model':'deepseek-flash',
            'choices':[{'finish_reason':finish,'message':{'content':'{"ok":true}' if finish=='stop' else '{"ok":'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':20}})
    output=tmp_path/'response.json'
    kwargs=dict(max_output_tokens=configuration()['llm']['max_output_tokens'],transport=httpx.MockTransport(handle))
    if finish=='stop':
        assert call_json([{'role':'user','content':'synthetic'}],{'type':'object'},output,**kwargs)=={'ok':True}
        assert read(output.with_suffix('.receipt.json'))['max_output_tokens']==65536
    else:
        with pytest.raises(ValueError,match='Incomplete JSON'):
            call_json([{'role':'user','content':'synthetic'}],{'type':'object'},output,**kwargs)
        assert not output.exists()
        receipt=read(next(tmp_path.glob('*.incomplete_*.receipt.json')))
        assert receipt['finish_reason']=='length' and receipt['usage']['completion_tokens']==20
