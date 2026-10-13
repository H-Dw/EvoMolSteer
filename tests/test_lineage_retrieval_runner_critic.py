"""Fake-critic integration for the Scientist precommit gate."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from evomolsteer.lineage_retrieval import runner, science_audit
from evomolsteer.lineage_retrieval.orchestrate import SCIENTIST_SCHEMA
from test_lineage_retrieval_bridge import scientist_source, descriptions, hypothesis


class _Stdin:
    def __init__(self, server):
        self.server = server
        self.responses = []

    def write(self, raw):
        message = json.loads(raw)
        request = self.server.by_id.get(message.get("id"))
        if request is not None and "result" in message:
            content = message["result"]["contentItems"][0]["text"]
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
            row = (request["params"]["tool"], request["params"].get("arguments", {}),
                   content, message["result"].get("success"))
            self.responses.append(row)

    async def drain(self):
        return None


class _FakeServer:
    def __init__(self, actions_by_turn):
        self.actions_by_turn = actions_by_turn
        self.turn_count = 0
        self.requests = asyncio.Queue()
        self.events = asyncio.Queue()
        self.by_id = {}
        self.responses = self
        self.stdin = _Stdin(self)
        self.proc = type("Proc", (), {"stdin": self.stdin, "returncode": 0})()
        self.thread_args = None
        self.turn_starts = []
        self.reader_task = None

    async def start(self):
        return None

    async def request(self, method, params):
        if method == "thread/start":
            self.thread_args = params
            return {"thread": {"id": "critic-fake-thread", "model": params["model"],
                "modelProvider": "fake", "reasoningEffort": params["config"]["model_reasoning_effort"]}}
        if method == "turn/start":
            self.turn_count += 1
            self.turn_starts.append(params)
            actions = self.actions_by_turn[self.turn_count - 1] if self.turn_count <= len(self.actions_by_turn) else []
            for i, (name, args) in enumerate(actions, 1):
                request_id = self.turn_count * 1000 + i
                message = {"id": request_id, "method": "item/tool/call",
                    "params": {"tool": name, "arguments": args}}
                self.by_id[request_id] = message
                self.requests.put_nowait(message)
            interval = min(10, max(1, self.turn_count - 1))
            ack = json.dumps({"submitted": True, "committed_interval": interval,
                "applied_clauses": [{"clause_id": "S.C04", "evidence": "Grounded comparison in the current six states."}]})
            self.events.put_nowait({"method": "item/agentMessage/delta", "params": {"delta": ack}})
            self.events.put_nowait({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
            return {"turn": {"id": f"critic-fake-turn-{self.turn_count}"}}
        raise AssertionError(method)

    async def close(self):
        return None


def _window_actions(source):
    rows = [("get_scientist_interval", {"interval": 1})]
    donor_ids = [f"T{i + 1:02d}" for i in range(30)]
    for start in range(0, 30, 5):
        rows.append(("get_scientist_donors", {"interval": 1, "donor_ids": donor_ids[start:start + 5]}))
    rows.append(("submit_scientist_interval", {"interval": 1,
        "descriptions": descriptions(1, source), "hypothesis": hypothesis(source, 1)}))
    return rows


def test_source_only_critic_rejection_keeps_window_locked_then_pass_commits(monkeypatch, tmp_path):
    source = scientist_source()
    first = _window_actions(source)
    first.append(("get_scientist_interval", {"interval": 2}))
    actions = [first, _window_actions(source)]
    server = _FakeServer(actions)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    critic_calls = []

    def fake_critic(job_dir, critic_source, interval, descs, hypo, prompt):
        payload, _ = science_audit._window_source(critic_source, interval, descs, hypo)
        payload["scientist_prompt_and_active_clause"] = prompt
        assert payload["scope"]["allowed_steps"] == list(range(6))
        assert payload["scope"]["source_future_states_included"] is False
        assert all(all(0 <= state["step"] <= 5 for state in donor["six_current_observation_states"])
                   for donor in payload["donor_descriptions"])
        assert "ACTIVE CLAUSE S.C04" in payload["scientist_prompt_and_active_clause"]
        assert "原始 feature key 和 pattern code 只放在结构化 metadata" in prompt
        critic_calls.append({"job_dir": job_dir, "interval": interval, "payload": payload, "prompt": prompt})
        if len(critic_calls) == 1:
            return {"passes": False, "evidence_truth": False, "natural_language": True,
                "clause_assessment": {"respected": True, "assessment": "One unsupported statement needs correction."},
                "unsupported_claims": ["T03 paragraph overstates a shape change."], "correction_needed": True}
        return {"passes": True, "evidence_truth": True, "natural_language": True,
            "clause_assessment": {"respected": True, "assessment": "Compared actual current and forecast changes."},
            "unsupported_claims": [], "correction_needed": False}

    monkeypatch.setattr(science_audit, "audit_scientist_submission", fake_critic)
    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
    prompt = "Scientist role instructions.\nACTIVE CLAUSE S.C04: Compare intermediate geometry with same-time population evidence."
    with pytest.raises(ValueError, match="twenty bounded continuations"):
        asyncio.run(runner._run_call(tmp_path, "scientist", prompt, source, SCIENTIST_SCHEMA,
            "gpt-6-luna", "medium", 30, None, [], [], "empty", "home",
            tmp_path / "attempts" / "attempt-1", journal, source))

    assert len(critic_calls) == 2
    assert all(call["interval"] == 1 for call in critic_calls)
    assert "ACTIVE CLAUSE S.C04" in critic_calls[0]["prompt"]
    assert len(journal["hypotheses"]) == 1 and journal["hypotheses"][0]["interval"] == 1
    assert len(journal["interval_descriptions"]) == 30
    proof = list((tmp_path / "attempts" / "attempt-1").glob("scientist_audit_proofs.jsonl"))
    assert len(proof) == 1
    proof_row = json.loads(proof[0].read_text(encoding="utf-8").splitlines()[0])
    assert proof_row["model"] == "gpt-6.1-sol" and proof_row["effort"] == "xhigh"
    assert proof_row["result"]["passes"] is True

    replies = server.stdin.responses
    rejected = next(row[2] for row in replies if row[0] == "submit_scientist_interval" and row[2].get("semantic_audit"))
    assert rejected["accepted"] is False and rejected["next_allowed_interval"] == 1
    assert rejected["corrections_remaining"] == 3
    assert rejected["current_window_facts"]["scope"]["allowed_steps"] == list(range(6))
    future = next(row[2] for row in replies if row[0] == "get_scientist_interval" and row[1]["interval"] == 2)
    assert future["accepted"] is False and "No data returned" in future["error"]
    assert len(server.turn_starts) >= 3
    assert all(turn["threadId"] == "critic-fake-thread" for turn in server.turn_starts)


def test_semantic_correction_limit_fails_closed_without_committing(monkeypatch, tmp_path):
    source = scientist_source()
    server = _FakeServer([_window_actions(source) for _ in range(4)])
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    critic_calls = []

    def always_reject(job_dir, critic_source, interval, descs, hypo, prompt):
        critic_calls.append(interval)
        return {"passes": False, "evidence_truth": False, "natural_language": True,
            "clause_assessment": {"respected": True, "assessment": "One unsupported measurement claim."},
            "unsupported_claims": ["A numeric statement does not match the source."],
            "correction_needed": True}

    monkeypatch.setattr(science_audit, "audit_scientist_submission", always_reject)
    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
    prompt = "Scientist instructions. ACTIVE CLAUSE S.C04: Compare the current interval with background."
    with pytest.raises(ValueError, match=r"correction limit \(4\) was reached"):
        asyncio.run(runner._run_call(tmp_path, "scientist", prompt, source, SCIENTIST_SCHEMA,
            "gpt-6-luna", "medium", 30, None, [], [], "empty", "home",
            tmp_path / "attempts" / "attempt-1", journal, source))

    assert critic_calls == [1, 1, 1, 1]
    assert journal["hypotheses"] == [] and journal["interval_descriptions"] == []
    state = json.loads((tmp_path / "scientist_semantic_corrections.json").read_text(encoding="utf-8"))
    assert state["limit_per_interval"] == 4
    assert state["counts"]["1"] == 4
