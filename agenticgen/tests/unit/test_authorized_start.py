"""The explicit initial transition cannot waive later reviews or invent coverage."""
from pathlib import Path
import pytest
from agenticgen.adaptive_schedule import group_review as gate
from agenticgen.adaptive_schedule import version_gate
from agenticgen.adaptive_schedule.io import read, write, sha
from agenticgen.adaptive_schedule.group_protocol import configuration


@pytest.fixture
def transition(tmp_path, monkeypatch):
    document = Path(__file__).parents[2]/'docs/authorizations/20261023_v6_start.json'
    local = tmp_path/'instruction.json'; write(local, read(document))
    monkeypatch.setattr(gate, 'committed_file', lambda path,digest,revision: local)
    calls=[]
    monkeypatch.setattr(version_gate, 'check_control_gate', lambda *args: calls.append(args))
    approval={'mode':'user_authorized_protocol_start', 'review_completed':False, 'coverage':[],
              'authorization_document':'agenticgen/docs/authorizations/20261023_v6_start.json',
              'authorization_document_sha256':sha(local)}
    return tmp_path, local, approval, calls


def test_initial_user_transition_reports_zero_reviewed_sessions(transition):
    root,_,approval,calls=transition
    result=gate.user_authorized_protocol_start(root/'new','v6',root/'old',configuration(),approval)
    assert result['approved'] and not result['review_completed'] and result['sessions']==0
    assert len(calls)==1  # Existing range/condition policy gate still runs.


def test_user_transition_cannot_waive_later_versions(transition):
    root,_,approval,_=transition
    with pytest.raises(ValueError,match='limited'):
        gate.user_authorized_protocol_start(root/'new','v7',root/'old',configuration(),approval)
    with pytest.raises(ValueError,match='limited'):
        gate.user_authorized_protocol_start(root,'v6',root,configuration(),approval)


def test_user_transition_cannot_claim_complete_review_or_weaken_scope(transition):
    root,doc,approval,_=transition
    approval['review_completed']=True
    with pytest.raises(ValueError,match='represented'):
        gate.user_authorized_protocol_start(root/'new','v6',root/'old',configuration(),approval)
    approval['review_completed']=False
    value=read(doc);value['subsequent_full_review_required']=False;write(doc,value)
    with pytest.raises(ValueError,match='scope'):
        gate.user_authorized_protocol_start(root/'new','v6',root/'old',configuration(),approval)
