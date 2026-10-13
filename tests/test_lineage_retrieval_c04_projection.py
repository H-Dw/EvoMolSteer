"""Fake app-server projection tests for the proposed C04 runner patch."""
import asyncio
import json
import queue
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evomolsteer.lineage_retrieval import runner


class _Stdin:
    def __init__(self, server):
        self.server = server
        self.responses = []

    def write(self, raw):
        message = json.loads(raw)
        request = self.server.by_id.get(message.get("id"))
        if request:
            body = message["result"]["contentItems"][0]["text"]
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass
            self.responses.append((request["params"]["tool"], body, message["result"].get("success")))

    async def drain(self):
        return None


class _Server:
    def __init__(self, actions, final_text='{"ok":true}'):
        self.requests = asyncio.Queue()
        self.events = asyncio.Queue()
        self.by_id = {}
        self.proc = type("Proc", (), {})()
        self.proc.returncode = 0
        self.stdin = _Stdin(self)
        self.proc.stdin = self.stdin
        self.actions = actions
        self.final_text = final_text
        self.turn_count = 0
        self.reader_task = None
        self.thread_args = None

    async def start(self):
        return None

    async def request(self, method, params):
        if method == "thread/start":
            self.thread_args = params
            return {"thread": {"id": "fake-thread", "model": params["model"],
                "modelProvider": "fake", "reasoningEffort": params["config"]["model_reasoning_effort"]}}
        if method == "turn/start":
            self.turn_count += 1
            if self.turn_count == 1:
                for i, (name, args) in enumerate(self.actions, 1):
                    request = {"id": i, "method": "item/tool/call", "params": {"tool": name, "arguments": args}}
                    self.by_id[i] = request
                    self.requests.put_nowait(request)
            self.events.put_nowait({"method": "item/agentMessage/delta", "params": {"delta": self.final_text}})
            self.events.put_nowait({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
            return {"turn": {"id": "fake-turn"}}
        raise AssertionError(method)

    async def close(self):
        return None


def _selector_packet():
    return {"role": "Selector", "step": 3, "affinity_available": False, "cards": [{
        "candidate_id": "anon_1", "now": {"current_radius_gyration_A": 2.0},
        "history": [
            {"step": 1, "time": 0.1, "current_radius_gyration_A": 1.0,
             "forecast_atom_count": 12, "unselected_scalar": 88},
            {"step": 2, "time": 0.2, "current_radius_gyration_A": 1.5,
             "forecast_atom_count": 15, "unselected_scalar": 91},
        ]} ]}


def test_get_history_feature_projection_uses_fake_broker_and_returns_bounded_errors(monkeypatch, tmp_path):
    packet = _selector_packet()
    actions = [
        ("get_history", {"candidate_id": "anon_1", "start_step": 1, "end_step": 2,
                          "features": ["current_radius_gyration_A", "forecast_atom_count"]}),
        ("get_history", {"candidate_id": "anon_1", "start_step": 1, "end_step": 2,
                          "features": ["not_a_feature"]}),
        ("get_history", {"candidate_id": "anon_1", "start_step": 1, "end_step": 2,
                          "features": ["online_pic50"]}),
        ("get_history", {"candidate_id": "anon_1", "start_step": 1, "end_step": 2}),
    ]
    server = _Server(actions)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}

    result = asyncio.run(runner._run_call(Path(tmp_path), "Selector", "test", packet, schema,
        "gpt-6-luna", "medium", 20, lambda name, args, p: p["cards"][0]["history"], [], [],
        "empty", "home", Path(tmp_path) / "attempt", journal))

    assert result == {"ok": True}
    assert server.thread_args["dynamicTools"]
    assert server.thread_args["config"]["features"]["shell_tool"] is False
    replies = server.stdin.responses
    projected = replies[0][1]
    assert projected == [
        {"step": 1, "time": 0.1, "current_radius_gyration_A": 1.0, "forecast_atom_count": 12},
        {"step": 2, "time": 0.2, "current_radius_gyration_A": 1.5, "forecast_atom_count": 15},
    ]
    for _, error, success in replies[1:3]:
        assert success is False and error["accepted"] is False and error["data_returned"] is False
        assert error["available_features"] and "unselected_scalar" not in error["available_features"]
    assert "online_pic50" not in replies[2][1]["available_features"]
    assert replies[3][1][0]["unselected_scalar"] == 88  # No features preserves legacy full output.
    assert replies[3][1][0]["forecast_atom_count"] == 12


def _scientist_source():
    donors = []
    features = ("radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A",
        "contact_pairs_4A", "pocket_contacts_4A", "planarity_fraction", "asphericity", "cov_xx_A2")
    for i in range(30):
        observations = []
        for step in range(51):
            row = {"step": step, "time": step / 10}
            for prefix in ("current", "forecast"):
                for j, feature in enumerate(features):
                    row[f"{prefix}_{feature}"] = i + step / 10 + j
                for axis_index, axis in enumerate("xyz"):
                    row[f"{prefix}_centroid_{axis}_A"] = float(i + step + axis_index)
                row[f"{prefix}_residue_contact_footprint"] = ["A:RES:1"]
            observations.append(row)
        donors.append({"donor_id": f"T{i + 1:02d}", "observations": observations})
    background = []
    for step in range(51):
        background.append({"step": step, "quantiles_P25_P50_P75": {
            f"{prefix}_{feature}": [step / 10, 15 + step / 10, 29 + step / 10]
            for prefix in ("current", "forecast") for feature in features}})
    return {"role": "scientist", "teacher": {"donors": donors}, "background": background}


def test_scientist_feature_summary_is_interval_gated_and_returns_json_errors(monkeypatch, tmp_path):
    source = _scientist_source()
    actions = [
        ("get_scientist_interval", {"interval": 1}),
        ("get_scientist_feature_summary", {"interval": 1, "features": ["current_cov_xx_A2"]}),
        ("get_scientist_feature_summary", {"interval": 1, "features": ["current_missing_feature"]}),
        ("get_scientist_feature_summary", {"interval": 2, "features": ["current_cov_xx_A2"]}),
    ]
    ack = json.dumps({"submitted": True, "committed_interval": 1,
        "applied_clauses": [{"clause_id": "S.BASE.3", "evidence": "same-time median comparison"}]})
    server = _Server(actions, ack)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    schema = {"type": "object", "properties": {
        "interval_descriptions": {"type": "array", "items": {"type": "object", "properties": {}}},
        "hypotheses": {"type": "array", "items": {"type": "object", "properties": {}}},
        "applied_clauses": {"type": "array", "items": {"type": "object", "properties": {}}}},
        "required": ["interval_descriptions", "hypotheses", "applied_clauses"], "additionalProperties": False}
    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
    with pytest.raises(ValueError, match="twenty bounded continuations"):
        asyncio.run(runner._run_call(Path(tmp_path), "scientist", "test", source, schema,
            "gpt-6-luna", "medium", 30, None, [], [], "empty", "home",
            Path(tmp_path) / "attempt", journal, source))
    tools = {tool["name"] for tool in server.thread_args["dynamicTools"]}
    assert "get_scientist_feature_summary" in tools
    valid = next(body for name, body, ok in server.stdin.responses
                 if name == "get_scientist_feature_summary" and ok is True)
    assert valid["accepted"] is True and valid["start_step"] == 0 and valid["end_step"] == 5
    assert len(valid["features"][0]["donors"]) == 30
    unknown = next(body for name, body, ok in server.stdin.responses
                   if name == "get_scientist_feature_summary" and ok is False and "available_features" in body)
    assert unknown["data_returned"] is False and "current_cov_xx_A2" in unknown["available_features"]
    future = next(body for name, body, ok in server.stdin.responses
                  if name == "get_scientist_feature_summary" and ok is False and "next_allowed_interval" in body)
    assert future["next_allowed_interval"] == 1 and "data_returned" not in future
