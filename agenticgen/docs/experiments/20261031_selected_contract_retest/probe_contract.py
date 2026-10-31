"""Bounded real transport fixtures, never a molecular measurement or Agent round."""

import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from agenticgen.providers.deepseek_tools import RoundBudget, call_with_tools
from agenticgen.uniguide.io import frozen, read, sha


class ProbeRegistry:
    def __init__(self, definitions, source):
        allowed = {"summarize_population", "compare_paired_runs", "inspect_interval_evidence", "audit_action_batch"}
        if {d["function"]["name"] for d in definitions} != allowed:
            raise ValueError("Probe requires the actual four read-only evidence schemas")
        self.definitions = definitions
        self.binding = dict(engineering_fixture_only=True, source_binding_sha256=sha(source))
        self.receipt_ids = set()

    def execute(self, name, arguments):
        raise ValueError("Interface fixture: no evidence execution is available; submit the specified final object")


def probe(old_plan, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Preserve the previous probe; do not automatically restart its claims")
    output.mkdir(parents=True)
    plan = read(old_plan)
    jobs = {j["id"]: j for j in plan["jobs"]}
    budget = RoundBudget(output / "model_budget", max_calls=4, max_output_tokens=65536)
    records = []
    for job_id, iteration, decision in [
        ("03_prefix_joint_6pyy", 2, "propose"),
        ("06_five_windows_3w54", 1, "hold"),
    ]:
        job = jobs[job_id]
        config = read(job["config"])
        source = (Path(config["root"]) / "versions/GD01" / job["teacher"] /
                  f"advice_{iteration:03d}/synthesis.session/BINDING.json")
        original = read(source)
        schema = copy.deepcopy(original["response_schema"])
        schema["properties"] = {k: schema["properties"][k] for k in ("decision", "selected", "reason")}
        schema["required"] = list(schema["properties"])
        schema["properties"]["decision"]["enum"] = [decision]
        candidate = schema["properties"]["selected"]["items"]["enum"][0]
        selected = [candidate] if decision == "propose" else []
        expected = dict(decision=decision, selected=selected,
                        reason="Interface fixture only; no scientific measurement or optimization decision.")

        def validate(value, expected=expected):
            if value["decision"] != expected["decision"] or value["selected"] != expected["selected"]:
                raise ValueError("Return the specified interface fixture decision and explicit selected array")

        result = call_with_tools(
            [dict(role="system", content=(
                "This is an engineering transport fixture, not molecular guidance. "
                "Do not call evidence tools or make optimization choices. "
                "Call submit_final_response alone with exactly the object in the user message.")),
             dict(role="user", content=json.dumps(expected, ensure_ascii=True))],
            schema, output / (job_id + ".json"),
            registry=ProbeRegistry(original["tools"], source), budget=budget,
            response_validator=validate, final_response_mode="strict_tool_v1",
            model="deepseek-flash", thinking=True, reasoning_effort="high",
            max_output_tokens=16384, max_format_repairs=1, attempts=3,
        )
        receipt = read(output / (job_id + ".receipt.json"))
        records.append(dict(job_id=job_id, source=str(source), source_sha256=sha(source),
            returned=result, receipt=receipt, fixture_only=True,
            no_sampling_or_evidence_execution=True, not_a_guidance_round=True))
    claims = budget.rows()
    complete = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        mode="strict_tool_v1", endpoint="https://api.deepseek.com/beta/chat/completions",
        model="deepseek-flash", thinking=True, reasoning_effort="high",
        requests=len(claims), actual_responses=len(list(output.glob("*.session/step_*/RESPONSE.json"))),
        observed_completion_tokens=sum(r.get("completion_tokens", 0) for r in claims),
        scientific_attempts=0, guidance_rounds=0, schema_probe_cases=records,
        maximum_http_attempts=4, maximum_output_tokens=65536)
    frozen(output / "ENGINEERING_PROBE_COMPLETE.json", complete)
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    probe(args.source_plan, args.output)
