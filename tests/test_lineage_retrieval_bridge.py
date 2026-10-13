"""Fake app-server integration for the progressive Scientist broker boundary."""
import asyncio
import json
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evomolsteer.lineage_retrieval import runner
from evomolsteer.lineage_retrieval.orchestrate import SCIENTIST_SCHEMA


def scientist_source():
    donors = []
    features = (
        "radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A",
        "contact_pairs_4A", "pocket_contacts_4A", "planarity_fraction", "asphericity",
    )
    for i in range(30):
        observations = []
        for step in range(51):
            row = {"step": step, "time": step / 100}
            for rep in ("current", "forecast"):
                for j, feature in enumerate(features):
                    # Complete numeric profiles make the real summary builder
                    # and median validation execute without external inputs.
                    row[f"{rep}_{feature}"] = 1.0 + i / 10 + step / 100 + j / 20 + (0.5 if rep == "forecast" else 0)
                for axis_index, axis in enumerate("xyz"):
                    row[f"{rep}_centroid_{axis}_A"] = float(i + step + axis_index)
                row[f"{rep}_residue_contact_footprint"] = ["A:RES:1", "A:RES:2"]
            observations.append(row)
        donors.append({"donor_id": f"T{i + 1:02d}", "observations": observations})
    background = []
    for step in range(51):
        quantiles = {}
        for rep in ("current", "forecast"):
            for feature in features:
                key = f"{rep}_{feature}"
                values = [d["observations"][step][key] for d in donors]
                quantiles[key] = [min(values), statistics.median(values), max(values)]
        background.append({"step": step, "quantiles_P25_P50_P75": quantiles})
    return {"role": "scientist", "teacher": {"donors": donors}, "background": background,
            "clock": {"steps": list(range(51))}}


def descriptions(interval, source):
    low = 5 * (interval - 1)
    mid = low + 2
    high = interval * 5
    anchors = [
        (low, "current_radius_gyration_A", "Measured current shape at window start."),
        (mid, "current_nearest_pocket_A", "Measured current pocket distance at the middle state."),
        (high, "current_pocket_contacts_4A", "Measured current pocket contacts at window end."),
        (low, "forecast_radius_gyration_A", "Measured forecast shape at window start."),
        (mid, "forecast_nearest_pocket_A", "Measured forecast pocket distance at the middle state."),
        (high, "forecast_pocket_contacts_4A", "Measured forecast pocket contacts at window end."),
    ]
    return [{
        "donor_id": f"T{i + 1:02d}", "interval": interval,
        "text": f"Observed computational profile {i + 1} in this window.",
        "evidence": [{"step": step, "feature": feature, "claim": claim}
                     for step, feature, claim in anchors],
        "measured_changes": [{
            "feature": feature,
            "start_value": source["teacher"]["donors"][i]["observations"][low][feature],
            "mid_value": source["teacher"]["donors"][i]["observations"][mid][feature],
            "end_value": source["teacher"]["donors"][i]["observations"][high][feature],
            "pattern": "increase",
        } for feature in ("current_radius_gyration_A", "forecast_radius_gyration_A",
                          "current_nearest_pocket_A", "forecast_pocket_contacts_4A")],
    } for i in range(30)]


def hypothesis(source, interval, invalid_public_number=False):
    low, high = 5 * (interval - 1), 5 * interval
    current = [d["observations"][low]["current_radius_gyration_A"] for d in source["teacher"]["donors"]]
    forecast = [d["observations"][high]["forecast_radius_gyration_A"] for d in source["teacher"]["donors"]]
    p50_current = source["background"][low]["quantiles_P25_P50_P75"]["current_radius_gyration_A"][1]
    p50_forecast = source["background"][high]["quantiles_P25_P50_P75"]["forecast_radius_gyration_A"][1]
    word = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")[interval - 1]
    text = f"The {word} window shows measured current shape development while the forecast remains a separate representation."
    if invalid_public_number:
        text += " The public claim includes 1.234."
    return {
        "interval": interval, "start_step": low, "end_step": high,
        "hypothesis": text,
        "support_conditions": [f"The {word} window's measured current feature differs from its forecast."],
        "counter_conditions": [f"A reversed measured shape trend would weaken the {word} window claim."],
        "observable_cues": [f"Compare current shape at step {low}.", f"Compare forecast shape at step {high} in window {word} only."],
        "minority_modes": f"Some paths vary within the {word} window.",
        "background_evidence": [
            {"feature": "current_radius_gyration_A", "step": low,
             "teacher_median": statistics.median(current), "background_median": p50_current,
             "evidence": "Window-start cohort median and same-time population P50."},
            {"feature": "forecast_radius_gyration_A", "step": high,
             "teacher_median": statistics.median(forecast), "background_median": p50_forecast,
             "evidence": "Window-end cohort median and same-time population P50."},
        ],
    }


def _tool_request(index, name, args):
    return {"id": index, "method": "item/tool/call", "params": {"tool": name, "arguments": args}}


class _FakeStdin:
    def __init__(self, server):
        self.server = server
        self.writes = []

    def write(self, payload):
        message = json.loads(payload)
        self.writes.append(message)
        call = self.server.tool_requests.get(message.get("id"))
        if not call or "result" not in message:
            return
        body = message["result"].get("contentItems", [{}])[0].get("text", "")
        try:
            response = json.loads(body)
        except json.JSONDecodeError:
            response = body
        self.server.tool_responses.append((call["params"]["tool"], call["params"].get("arguments", {}), response))
        if (call["params"]["tool"] == "submit_scientist_interval"
                and response.get("accepted") is True):
            interval = call["params"]["arguments"]["interval"]
            ack = json.dumps({"submitted": True, "committed_interval": interval,
                              "applied_clauses": [{"clause_id": "S.BASE.3", "evidence": "Compared actual current and forecast states."}]})
            self.server.events.put_nowait({"method": "item/agentMessage/delta", "params": {"delta": ack}})
            self.server.events.put_nowait({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})

    async def drain(self):
        return None


class _FakeServer:
    def __init__(self, first_actions, continuation_actions=None, premature_ack_first=False):
        self.requests = asyncio.Queue()
        self.events = asyncio.Queue()
        self.tool_requests = {}
        self.tool_responses = []
        self.stdin = _FakeStdin(self)
        self.proc = type("Proc", (), {"stdin": self.stdin, "returncode": 0})()
        self.turn_actions = [list(first_actions)] + [list(actions) for actions in (continuation_actions or [])]
        self.premature_ack_first = premature_ack_first
        self.turn_starts = []
        self.thread_start_params = None
        self.reader_task = None

    async def start(self):
        return None

    async def request(self, method, params):
        if method == "thread/start":
            self.thread_start_params = params
            return {"thread": {"id": "fake-thread-1", "model": params["model"],
                               "modelProvider": "fake", "reasoningEffort": params["config"]["model_reasoning_effort"]}}
        if method == "turn/start":
            self.turn_starts.append(params)
            index = len(self.turn_starts) - 1
            if index < len(self.turn_actions):
                self._enqueue(self.turn_actions[index])
            if index == 0 and self.premature_ack_first:
                ack = json.dumps({"submitted": True, "committed_interval": 1,
                    "applied_clauses": [{"clause_id": "S.BASE.3", "evidence": "Acknowledgement without a journal commit."}]})
                self.events.put_nowait({"method": "item/agentMessage/delta", "params": {"delta": ack}})
                self.events.put_nowait({"method": "turn/completed", "params": {"turn": {"status": "completed"}}})
            return {"turn": {"id": f"fake-turn-{len(self.turn_starts)}"}}
        raise AssertionError(f"unexpected app-server request: {method}")

    async def _unused_request(self, method, params):
        return {}

    def _enqueue(self, actions):
        for i, (name, args) in enumerate(actions, 1):
            message = _tool_request(1000 * len(self.turn_starts) + i, name, args)
            self.tool_requests[message["id"]] = message
            self.requests.put_nowait(message)

    async def close(self):
        return None


def _episode_actions(source, start_interval, end_interval, invalid_first=False, include_future_probe=False):
    actions = []
    index = 1
    for interval in range(start_interval, end_interval + 1):
        actions.append(("get_scientist_interval", {"interval": interval}))
        index += 1
        if include_future_probe and interval == 1:
            actions.append(("get_scientist_interval", {"interval": 2}))
            index += 1
        donor_ids = [f"T{i + 1:02d}" for i in range(30)]
        for chunk in range(0, 30, 5):
            actions.append(("get_scientist_donors", {"interval": interval, "donor_ids": donor_ids[chunk:chunk + 5]}))
            index += 1
        if invalid_first and interval == 1:
            actions.append(("submit_scientist_interval", {"interval": interval,
                "descriptions": descriptions(interval, source), "hypothesis": hypothesis(source, interval, True)}))
            index += 1
        actions.append(("submit_scientist_interval", {"interval": interval,
            "descriptions": descriptions(interval, source), "hypothesis": hypothesis(source, interval)}))
        index += 1
    return actions


def _run_fake(monkeypatch, source, first, continuation=None, tmp_path=None, premature_ack_first=False):
    server = _FakeServer(first, continuation, premature_ack_first)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    packet = {"role": "scientist", "available_interval": 1, "max_allowed_step": 5}
    journal = {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}
    events, calls = [], []
    result = asyncio.run(runner._run_call(
        tmp_path or Path("fake-worker"), "scientist", "scientist protocol test", packet,
        SCIENTIST_SCHEMA, "gpt-6-luna", "medium", 30, None, events, calls,
        "fake-cwd", "fake-home", (tmp_path or Path("fake-worker")) / "attempts" / "attempt-1",
        journal, source,
    ))
    return result, journal, events, calls, server


def test_fake_app_server_enforces_progressive_scientist_gate_and_continues_same_thread(monkeypatch, tmp_path):
    source = scientist_source()
    # Shared numeric offsets with identical directions are legitimately the
    # same qualitative evolution. Give one path an actual footprint difference
    # so this test exercises unsupported template reuse, not cosmetic wording.
    source['teacher']['donors'][1]['observations'][0]['forecast_residue_contact_footprint'] = ['A:OTHER:9']
    first = _episode_actions(source, 1, 1, invalid_first=True, include_future_probe=True)
    template = descriptions(1, source)
    for row in template:
        row["text"] = "The donor follows a computational trajectory in the window."
    # Even exact, source-bound anchors cannot make one copied paragraph valid
    # for thirty distinct profiles.
    first.insert(8, ("submit_scientist_interval", {"interval": 1,
        "descriptions": template, "hypothesis": hypothesis(source, 1)}))
    # Even after a successful commit, a same-turn request for the next window
    # must be denied; only a controller-issued turn may unlock it.
    first.append(("get_scientist_interval", {"interval": 2}))
    continuation = [_episode_actions(source, interval, interval) for interval in range(2, 11)]
    result, journal, events, calls, server = _run_fake(monkeypatch, source, first, continuation, tmp_path)

    assert len(journal["interval_descriptions"]) == 300
    assert [row["interval"] for row in journal["hypotheses"]] == list(range(1, 11))
    assert len(result["hypotheses"]) == 10 and len(result["interval_descriptions"]) == 300
    assert len(result['applied_clauses']) == 10
    assert all(c['evidence'].startswith(f'Interval {i}:') for i, c in enumerate(result['applied_clauses'], 1))
    acknowledgements = next(tmp_path.rglob('scientist_acknowledgements.jsonl'))
    assert len(acknowledgements.read_text(encoding='utf-8').splitlines()) == 10
    assert len(server.turn_starts) == 10
    assert {turn["threadId"] for turn in server.turn_starts} == {"fake-thread-1"}
    assert all(turn["model"] == "gpt-6-luna" and turn["effort"] == "medium" for turn in server.turn_starts)
    assert all(f"interval {i} only" in turn["input"][0]["text"] for i, turn in enumerate(server.turn_starts[1:], 2))
    for interval, turn in enumerate(server.turn_starts[1:], 2):
        visible_packet = json.loads(turn["input"][0]["text"].split("SANITIZED MEMORY PACKET (JSON):\n", 1)[1])
        assert visible_packet["available_interval"] == interval
        assert visible_packet["max_allowed_step"] == interval * 5
        assert visible_packet["committed_interval_count"] == interval - 1
    config = server.thread_start_params["config"]
    assert config["features"] == {"shell_tool": False, "multi_agent": False, "apps": False,
        "plugins": False, "browser_use": False, "computer_use": False, "skill_search": False}
    assert {tool["name"] for tool in server.thread_start_params["dynamicTools"]} == {
        "get_scientist_interval", "get_scientist_donors", "get_scientist_details", "submit_scientist_interval"}

    responses = server.tool_responses
    invalid_public = next(resp for name, args, resp in responses
                          if name == "submit_scientist_interval" and args["interval"] == 1
                          and "exact continuous measurements" in resp.get("error", ""))
    assert invalid_public["next_allowed_interval"] == 1
    assert "exact continuous measurements" in invalid_public["error"]
    copied_template = next(resp for name, args, resp in responses
                           if name == "submit_scientist_interval" and args["interval"] == 1
                           and "identical paragraph reused" in resp.get("error", ""))
    assert copied_template["next_allowed_interval"] == 1
    future_probe = next(resp for name, args, resp in responses
                        if name == "get_scientist_interval" and args["interval"] == 2)
    assert future_probe["accepted"] is False
    assert "donor_ids" not in future_probe and "cohort_background" not in future_probe
    assert "No data returned" in future_probe["error"]
    same_turn_future = [resp for name, args, resp in responses
                        if name == "get_scientist_interval" and args["interval"] == 2]
    assert any("controller opens interval 2 only in a new turn" in resp["error"] for resp in same_turn_future)

    accepted = [(args["interval"], resp) for name, args, resp in responses
                if name == "submit_scientist_interval" and resp.get("accepted") is True]
    assert [interval for interval, _ in accepted] == list(range(1, 11))
    profile_calls = [(args["interval"], args["donor_ids"]) for name, args, _ in responses
                     if name == "get_scientist_donors"]
    assert all(1 <= len(ids) <= 5 for _, ids in profile_calls)
    for interval in range(1, 11):
        ids = [donor_id for step, chunk in profile_calls if step == interval for donor_id in chunk]
        assert len(ids) == 30 and len(set(ids)) == 30
    committed = [event["interval"] for event in events if event["kind"] == "scientist_interval_committed"]
    assert committed == list(range(1, 11))
    assert [event["interval"] for event in events if event["kind"] == "scientist_window_turn_completed"] == list(range(1, 10))


def test_fake_app_server_requires_all_thirty_profiles_before_submission(monkeypatch, tmp_path):
    source = scientist_source()
    actions = [
        ("get_scientist_interval", {"interval": 1}),
        ("get_scientist_donors", {"interval": 1, "donor_ids": [f"T{i + 1:02d}" for i in range(5)]}),
        ("submit_scientist_interval", {"interval": 1, "descriptions": descriptions(1, source), "hypothesis": hypothesis(source, 1)}),
    ]
    server = _FakeServer(actions)
    monkeypatch.setattr(runner, "_AppServer", lambda *args: server)
    packet = {"role": "scientist", "available_interval": 1, "max_allowed_step": 5}
    with pytest.raises(ValueError, match="Read all thirty donor profiles"):
        asyncio.run(runner._run_call(Path("fake"), "scientist", "prompt", packet,
            SCIENTIST_SCHEMA, "gpt-6-luna", "medium", 30, None, [], [], "cwd", "home",
            tmp_path / "attempt-1", {"interval_descriptions": [], "hypotheses": [], "applied_clauses": []}, source))
    submit_responses = [row for row in server.tool_responses if row[0] == "submit_scientist_interval"]
    assert submit_responses == []


def test_fake_app_server_retries_premature_ack_in_same_window_turn(monkeypatch, tmp_path):
    source = scientist_source()
    actions = [_episode_actions(source, interval, interval) for interval in range(1, 11)]
    result, journal, events, _, server = _run_fake(monkeypatch, source, [], actions, tmp_path,
                                                    premature_ack_first=True)
    assert len(result["hypotheses"]) == 10
    assert len(journal["interval_descriptions"]) == 300
    assert any(event["kind"] == "scientist_premature_ack_rejected" and event["next_allowed_interval"] == 1
               for event in events)
    assert len(server.turn_starts) == 11
    assert server.turn_starts[1]["threadId"] == server.turn_starts[0]["threadId"]
    assert "interval 1 only" in server.turn_starts[1]["input"][0]["text"]
