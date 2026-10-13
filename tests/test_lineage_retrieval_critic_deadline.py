import asyncio
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from evomolsteer.lineage_retrieval import runner, science_audit
from evomolsteer.lineage_retrieval.orchestrate import SCIENTIST_SCHEMA
from test_lineage_retrieval_bridge import scientist_source, _episode_actions, _FakeServer


def test_critic_elapsed_time_is_excluded_from_scientist_deadline(monkeypatch, tmp_path):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now))
    source = scientist_source()
    first = _episode_actions(source, 1, 1)
    # Reject once, then retry the same unlocked window and pass. A request for
    # the next interval in that same turn must remain denied after the commit.
    submit = next(action for action in first if action[0] == "submit_scientist_interval")
    first.append(submit)
    first.append(("get_scientist_interval", {"interval": 2}))
    continuations = [_episode_actions(source, interval, interval) for interval in range(2, 11)]
    server = _FakeServer(first, continuations)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)

    audits = []
    def fake_critic(job_dir, critic_source, interval, descriptions, hypothesis, prompt):
        clock.now += 0.10  # simulated audit elapsed time exceeds the 0.05 s turn budget
        audits.append(interval)
        if len(audits) == 1:
            return {"passes": False, "evidence_truth": False, "natural_language": True,
                "clause_assessment": {"respected": True, "assessment": "Correct a current-window claim."},
                "unsupported_claims": ["One claim conflicts with source."], "correction_needed": True}
        return {"passes": True, "evidence_truth": True, "natural_language": True,
            "clause_assessment": {"respected": True, "assessment": "Grounded in the current window."},
            "unsupported_claims": [], "correction_needed": False}
    monkeypatch.setattr(science_audit, "audit_scientist_submission", fake_critic)

    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
    events, calls = [], []
    result = asyncio.run(runner._run_call(tmp_path, "scientist", "test prompt", source,
        SCIENTIST_SCHEMA, "gpt-6-luna", "medium", 0.05, None, events, calls,
        "fake-cwd", "fake-home", tmp_path / "attempts" / "attempt-1", journal, source))

    assert len(result["hypotheses"]) == 10
    assert [row["interval"] for row in journal["hypotheses"]] == list(range(1, 11))
    assert audits[0:2] == [1, 1]
    audit_reject = next(event for event in events if event["kind"] == "scientist_semantic_audit_rejected")
    assert audit_reject["interval"] == 1
    rejected_reply = next(response for tool, args, response in server.tool_responses
                          if tool == "submit_scientist_interval" and response.get("semantic_audit"))
    assert rejected_reply["accepted"] is False and rejected_reply["next_allowed_interval"] == 1
    assert rejected_reply["current_window_facts"]["scope"]["allowed_steps"] == list(range(6))
    # Rejection neither commits nor opens interval 2; successful retry then
    # commits interval 1, with interval 2 available only in a new turn.
    denied_future = next(resp for tool, args, resp in server.tool_responses
                         if tool == "get_scientist_interval" and args.get("interval") == 2)
    assert denied_future["accepted"] is False
    assert "No data returned" in denied_future["error"]
    assert len(server.turn_starts) == 10
