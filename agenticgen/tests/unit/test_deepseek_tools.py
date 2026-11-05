import json
from pathlib import Path
import httpx
import pytest
from agenticgen.providers.deepseek_tools import call_with_tools,RoundBudget,SessionStopped
from agenticgen.uniguide.io import read,write


class Registry:
    definitions=[dict(type="function",function=dict(name="measure",description="Test evidence",parameters={"type":"object","properties":{},"additionalProperties":False}))]
    binding={"test":"fixture"}
    receipt_ids=set()
    def execute(self,name,args):
        self.receipt_ids.add("ev_mock")
        return dict(status="complete",evidence_id="ev_mock",value=4)


SCHEMA=dict(type="object",properties={"ok":{"type":"boolean"}},required=["ok"],additionalProperties=False)
MESSAGES=[dict(role="user",content="Test only")]


def wire(content='{"ok":true}',calls=None,finish=None):
    return dict(model="deepseek-flash",id="mock",usage=dict(completion_tokens=20),choices=[dict(
        finish_reason=finish or ("tool_calls" if calls else "stop"),message=dict(role="assistant",content=content,
        reasoning_content="opaque-protocol-field",**({"tool_calls":calls} if calls else {})))])


def tool(identifier="t",name="measure",arguments="{}"):
    return dict(id=identifier,type="function",function=dict(name=name,arguments=arguments))


def run(tmp_path,monkeypatch,responses,**kwargs):
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real")
    seen=[]
    def handle(request):
        seen.append(json.loads(request.content))
        value=responses.pop(0)
        return httpx.Response(value if isinstance(value,int) else 200,json={} if isinstance(value,int) else value)
    result=call_with_tools(MESSAGES,SCHEMA,tmp_path/"answer.json",registry=Registry(),budget=RoundBudget(tmp_path/"budget"),
        transport=httpx.MockTransport(handle),sleep=lambda _:None,**kwargs)
    return result,seen


def test_null_multiple_tools_full_reasoning_passback_and_cached_replay(tmp_path,monkeypatch):
    answer,seen=run(tmp_path,monkeypatch,[wire(None,[tool("a"),tool("b")]),wire()])
    assert answer=={"ok":True} and len(seen)==2
    assistant=seen[1]["messages"][-3]
    assert assistant["content"] is None and assistant["reasoning_content"]=="opaque-protocol-field"
    assert [m["tool_call_id"] for m in seen[1]["messages"][-2:]]==["a","b"]
    assert len(RoundBudget(tmp_path/"budget").rows())==2
    cached,again=run(tmp_path,monkeypatch,[])
    assert cached==answer and again==[]


def test_dependency_and_unknown_tool_return_error_not_fake_measurement(tmp_path,monkeypatch):
    result,seen=run(tmp_path,monkeypatch,[wire(None,[tool(name="not_registered")]),wire(None,[tool()]),wire()])
    assert len(seen)==3
    assert json.loads(seen[1]["messages"][-1]["content"])["status"]=="tool_error"
    assert result["ok"]


def test_final_format_repair_keeps_reasoning_and_counts_http(tmp_path,monkeypatch):
    _,seen=run(tmp_path,monkeypatch,[wire('{"ok":true,"extra":0}'),wire()])
    assert len(seen)==2 and seen[1]["messages"][-2]["reasoning_content"]=="opaque-protocol-field"
    assert read(tmp_path/"answer.receipt.json")["interface_repairs"]==1


def test_repair_reports_schema_and_citation_errors_together(tmp_path,monkeypatch):
    calls=[]
    def validate(result):
        calls.append(result)
        if "extra" in result:
            raise ValueError("Evidence must copy full receipt identity")
    _,seen=run(tmp_path,monkeypatch,[wire('{"ok":true,"extra":0}'),wire()],response_validator=validate)
    repair=seen[1]["messages"][-1]["content"]
    assert "additionalProperties" in repair and "full receipt identity" in repair
    assert len(calls)==2


@pytest.mark.parametrize("response",[wire(None,finish="length"),wire("not-json"),wire(None)])
def test_failed_session_is_preserved_not_automatically_retried(tmp_path,monkeypatch,response):
    with pytest.raises(SessionStopped):run(tmp_path,monkeypatch,[response],max_format_repairs=0)
    assert (tmp_path/"answer.session/FAILED.json").exists()
    with pytest.raises(SessionStopped,match="Archived"):run(tmp_path,monkeypatch,[],max_format_repairs=0)
    assert len(RoundBudget(tmp_path/"budget").rows())==1


def test_thirty_call_limit_and_last_request_disables_tools(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real");seen=[]
    def handle(request):
        seen.append(json.loads(request.content));return httpx.Response(200,json=wire(None,[tool(str(len(seen)))]))
    with pytest.raises(SessionStopped,match="budget"):
        call_with_tools(MESSAGES,SCHEMA,tmp_path/"answer.json",registry=Registry(),budget=RoundBudget(tmp_path/"budget"),
            transport=httpx.MockTransport(handle))
    assert len(seen)==30 and seen[-1]["tool_choice"]=="none"
    assert not (tmp_path/"answer.json").exists()


def test_network_retries_consume_same_round_budget(tmp_path,monkeypatch):
    _,seen=run(tmp_path,monkeypatch,[429,wire()])
    assert len(seen)==2 and len(RoundBudget(tmp_path/"budget").rows())==2


def test_shared_budget_does_not_reset_between_roles(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real");budget=RoundBudget(tmp_path/"budget",1)
    transport=httpx.MockTransport(lambda _:httpx.Response(200,json=wire()))
    call_with_tools(MESSAGES,SCHEMA,tmp_path/"one.json",registry=Registry(),budget=budget,transport=transport)
    with pytest.raises(SessionStopped,match="budget"):
        call_with_tools(MESSAGES,SCHEMA,tmp_path/"two.json",registry=Registry(),budget=budget,transport=transport)
    assert len(budget.rows())==1


def test_tool_step_crash_resume_uses_durable_response(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real");count=[]
    class CrashOnce(Registry):
        def execute(self,name,args):raise KeyboardInterrupt()
    def handle(_):count.append(1);return httpx.Response(200,json=wire(None,[tool()]))
    with pytest.raises(KeyboardInterrupt):
        call_with_tools(MESSAGES,SCHEMA,tmp_path/"answer.json",registry=CrashOnce(),budget=RoundBudget(tmp_path/"budget"),transport=httpx.MockTransport(handle))
    result,seen=run(tmp_path,monkeypatch,[wire()])
    assert result["ok"] and len(count)==1 and len(seen)==1


def test_ambiguous_request_not_reissued(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","mock-not-real")
    def stop(_):raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        call_with_tools(MESSAGES,SCHEMA,tmp_path/"answer.json",registry=Registry(),budget=RoundBudget(tmp_path/"budget"),transport=httpx.MockTransport(stop))
    with pytest.raises(SessionStopped,match="Ambiguous"):run(tmp_path,monkeypatch,[])


@pytest.mark.parametrize('count',[13,15,32])
def test_large_readonly_batches_execute_completely_in_order(tmp_path,monkeypatch,count):
    monkeypatch.setenv('DEEPSEEK_API_KEY','mock-not-real')
    class OrderedRegistry(Registry):
        definitions=[dict(type='function',function=dict(name='measure',description='Read existing evidence',
            parameters=dict(type='object',properties=dict(index=dict(type='integer')),required=['index'],additionalProperties=False)))]
        def __init__(self):self.executed=[];self.receipt_ids=set()
        def execute(self,name,args):
            self.executed.append(args['index']);identity='ev_%020x'%args['index'];self.receipt_ids.add(identity)
            return dict(status='complete',evidence_id=identity)
    registry=OrderedRegistry();seen=[]
    responses=[wire(None,[tool(str(i),arguments=json.dumps(dict(index=i))) for i in range(count)]),wire()]
    def handle(request):
        seen.append(json.loads(request.content));return httpx.Response(200,json=responses.pop(0))
    result=call_with_tools(MESSAGES,SCHEMA,tmp_path/'answer.json',registry=registry,budget=RoundBudget(tmp_path/'budget'),
        transport=httpx.MockTransport(handle))
    assert result=={'ok':True} and registry.executed==list(range(count))
    assert [m['tool_call_id'] for m in seen[1]['messages'][-count:]]==[str(i) for i in range(count)]
    assert len(list((tmp_path/'answer.session/step_0001').glob('TOOL_*.json')))==count
    assert len(RoundBudget(tmp_path/'budget').rows())==2
    assert read(tmp_path/'answer.session/BINDING.json')['max_tool_calls_per_batch']==32
    assert read(tmp_path/'answer.receipt.json')['max_tool_calls_per_batch']==32


def test_thirty_three_calls_fail_before_any_tool_execution(tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','mock-not-real');executed=[];requests=[]
    class CountedRegistry(Registry):
        def execute(self,name,args):executed.append(name);return super().execute(name,args)
    def handle(request):
        requests.append(request);return httpx.Response(200,json=wire(None,[tool(str(i)) for i in range(33)]))
    with pytest.raises(SessionStopped,match='oversized tool-call batch'):
        call_with_tools(MESSAGES,SCHEMA,tmp_path/'answer.json',registry=CountedRegistry(),budget=RoundBudget(tmp_path/'budget'),
            transport=httpx.MockTransport(handle))
    assert not executed and len(requests)==1 and not (tmp_path/'answer.json').exists()
    assert not list((tmp_path/'answer.session/step_0001').glob('TOOL_*.json'))
    assert read(tmp_path/'answer.session/FAILED.json')['automatic_restart'] is False
