"""Citation transport contracts, not LLM strategy-effect tests."""
import copy
import json
import pytest
from agenticgen.adaptive_schedule.evidence_aliases import EvidenceAliases
from agenticgen.adaptive_schedule.io import value_sha, read, defaults
from agenticgen.adaptive_schedule.group_agent import plan_population
from test_plip_group_agent import population, hypotheses


def test_labels_are_exact_session_local_and_lossless():
    original = {'evidence_id':'tool:'+'a'*64,'result':{'score':1.234567890123,'bad':None}}
    a=EvidenceAliases(); projected=a.project([original])[0]
    assert projected['evidence_id']=='E001' and projected['result']==original['result']
    assert projected['source_record_sha256']==value_sha(original)
    assert a.project([original])[0]==projected
    assert a.project([{'evidence_id':'tool:'+'b'*64,'result':2}])[0]['evidence_id']=='E002'
    response={'hypotheses':[{'ticks':[3,5,5,7],'hypothesis':'do not rewrite E001 prose','evidence_ids':['E001']}],
              'learning_update':[{'evidence_ids':['E002']} ]}
    frozen=copy.deepcopy(response); decoded=a.restore_citations(response)
    assert response==frozen and decoded['hypotheses'][0]['ticks']==[3,5,5,7]
    assert decoded['hypotheses'][0]['hypothesis']=='do not rewrite E001 prose'
    assert decoded['hypotheses'][0]['evidence_ids']==[original['evidence_id']]
    assert decoded['learning_update'][0]['evidence_ids']==['tool:'+'b'*64]
    b=EvidenceAliases(); b.project([{'evidence_id':'tool:'+'c'*64,'result':3}])
    assert b.restore_citations({'evidence_ids':['E001']})['evidence_ids']==['tool:'+'c'*64]
    assert 'tool:'+'a'*64 not in str(b.manifest())
    with pytest.raises(ValueError): a.project([{**original,'result':{'changed':True}}])


@pytest.mark.parametrize('invalid',['E01','E999','tool:'+'a'*64,0,{}])
def test_no_fuzzy_reference_correction(invalid):
    a=EvidenceAliases();a.project([{'evidence_id':'tool:'+'a'*64,'result':1}])
    with pytest.raises(ValueError):a.restore_citations({'evidence_ids':[invalid]})


def test_planner_keeps_wire_labels_and_canonical_downstream_sources(tmp_path):
    observed=[]
    def caller(messages,schema,output,**kwargs):
        payload=json.loads(messages[-1]['content']); observed.append(payload)
        if len(observed)==1:
            response={'kind':'tools','reason':'get existing facts','calls':[
                {'name':'geometry_diagnostics','arguments':{'selection':'latest','step':90,'state':'endpoint'}}]}
        else:
            label=payload['tool_results'][0]['evidence_id']
            assert label=='E003'
            response=hypotheses(4,label)
        kwargs['response_validator'](response)
        return response
    result=plan_population(population(),tmp_path,defaults(),caller=caller)
    mapping=read(tmp_path/'evidence_aliases.json')['aliases']
    canonical=result['jobs'][0]['evidence_ids'][0]
    assert canonical==mapping['E003']['source_id']
    record=read(tmp_path/'tools'/(canonical.removeprefix('tool:')+'.json'))
    assert value_sha(record)==mapping['E003']['source_record_sha256']
    assert read(tmp_path/'turn_01.resolved.json')['hypotheses'][0]['evidence_ids']==[canonical]
