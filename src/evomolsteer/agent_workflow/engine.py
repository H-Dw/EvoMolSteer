"""Public action/observation loop; deterministic extraction, no hidden reasoning log."""
import copy
import hashlib
from pathlib import Path

import jsonschema

from evomolsteer.deepseek_api import call_json
from evomolsteer.io import digest, read_json, write_json
from .contracts import POLICY, action_schema, canonical, format_contract, strict_json, validate


class WorkflowBudgetError(ValueError):
    pass


def _persist(path, value):
    """Resume may reuse identical artifacts, never overwrite different source runs."""
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Workflow artifact belongs to different inputs: " + path.name)
    else:
        write_json(path, value)


def extract_result(role, result, tools, choices=None):
    """Project wire fields into the existing downstream contracts explicitly."""
    if role == "Analyst":
        legacy = {"role": role, "findings": [], "hypotheses": [], "limitations": result["limitations"]}
        for i, item in enumerate(result["findings"]):
            value = {"id": f"F{i + 1:03d}", "claim": item["claim"], "evidence_ids": tools.resolve(item["evidence_refs"])}
            if "priority" in item:
                value["priority"] = item["priority"]
            legacy["findings"].append(value)
        for i, item in enumerate(result["hypotheses"]):
            legacy["hypotheses"].append({"id": f"H{i + 1:03d}", "statement": item["statement"],
                "evidence_ids": tools.resolve(item["evidence_refs"]),
                **{k: item[k] for k in ["observable", "direction", "scope", "falsifier", "curve_features"] if k in item}})
        return legacy
    edit = copy.deepcopy(choices[result["edit_choice"]]) if result["decision"] == "edit" else None
    return {"role": role, "decision": result["decision"], "edit": edit,
            "evidence_ids": tools.resolve(result["evidence_refs"]), "rationale": result["rationale"],
            "failure_modes": result["failure_modes"],
            "parameter_origins": {"hypothesis_ids": result["hypothesis_refs"], "registered_choice": result["edit_choice"]}}


def run_agent(role, tools, instructions, task, output_dir, *, choices=None, handoff=None,
              max_steps=10, max_output_tokens=4096, max_context_bytes=280000,
              caller=call_json, transport=None, finish_validator=None, finish_policy=None,
              schema_enricher=None, evidence_availability=None, interface_policy=POLICY, max_format_repairs=1):
    """Use JSON actions on stable Chat Completions, not fabricated native tool calls.

    The role can issue a single whitelisted operation per turn. Host execution
    produces an observation; only observed handles are legal final citations.
    Scientific research instructions come solely from `instructions`.
    """
    if role not in {"Analyst", "Designer"} or not 1 <= max_steps <= 30:
        raise ValueError("Registered role and bounded step budget required")
    if (finish_validator is None) != (finish_policy is None):
        raise ValueError("A scientific finish validator requires a versioned policy binding")
    if schema_enricher is not None and finish_policy is None:
        raise ValueError("A scientific schema requires a versioned finish policy")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    before = tools.provenance()
    binding = {"interface_version": interface_policy, "role": role, "model": "deepseek-flash",
               "instruction_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
               "sources": tools.files, "task": task, "choices": choices, "handoff": handoff,
               "initial_references": before, "max_steps": max_steps,
               "max_output_tokens": max_output_tokens, "max_context_bytes": max_context_bytes}
    if finish_policy is not None:
        binding["finish_policy"] = finish_policy
    if schema_enricher is not None:
        binding.update(scientific_schema=True, max_format_repairs=max_format_repairs)
    _persist(folder / "binding.json", binding)
    instructions_path = folder / "instructions.md"
    if instructions_path.exists() and instructions_path.read_text(encoding="utf-8") != instructions:
        raise ValueError("Changed instructions in an existing workflow")
    instructions_path.write_text(instructions, encoding="utf-8")
    initial = {"task": task, "catalogue": tools.catalogue(), "registered_choices": choices,
               "handoff": handoff, "initial_references": before["references"],
               "clock_observation": tools.execute("read_metadata", {"section": "clocks"})}
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": canonical(initial)}]
    hypothesis_ids = [v["id"] for v in (handoff or {}).get("hypotheses", [])]
    signatures = set()
    last_gaps, repeated_gaps, diagnostics = None, 0, []
    for step in range(max_steps):
        if len(canonical(messages).encode()) > max_context_bytes:
            raise WorkflowBudgetError("Context budget exhausted; no truncated evidence or synthetic success")
        schema = action_schema(role, tools.known, tools.schemas, choices, hypothesis_ids)
        if schema_enricher is not None:
            schema = schema_enricher(schema, tools, stalled=repeated_gaps >= 2, diagnostics=diagnostics)
        current_messages = messages
        if evidence_availability is not None:
            current_messages = messages + [{"role": "user", "content": canonical({
                "current_evidence_availability": evidence_availability(tools),
                "repeated_unresolved_finishes": repeated_gaps,
                "recovery_constraint": "After repeated unchanged evidence gaps, finish hypotheses must use "
                    "observed features and matching handles. Read missing curves or omit unsupported hypotheses."})}]

        def semantic(action):
            if action["action"] == "tool":
                tools.validate_call(action["tool"], action["arguments"])

        kwargs = dict(input_store=folder / "inputs", max_output_tokens=max_output_tokens,
                      response_validator=semantic, response_parser=strict_json,
                      format_policy=interface_policy,
                      format_builder=lambda s, d=None: format_contract(s, d, policy=interface_policy))
        if max_format_repairs != 1:
            kwargs['max_format_repairs'] = max_format_repairs
        if transport is not None:
            kwargs["transport"] = transport
        action = caller(current_messages, schema, folder / f"step_{step:02d}.json", **kwargs)
        # Even injected clients and cached responses must pass identical checks.
        validate(action, schema)
        semantic(action)
        if action["action"] == "finish":
            tools.verify_sources()
            diagnostics = finish_validator(action["result"], tools) if finish_validator else []
            if diagnostics:
                gap_key = canonical({'diagnostics': diagnostics, 'known_refs': sorted(tools.known)})
                repeated_gaps = repeated_gaps + 1 if gap_key == last_gaps else 1
                last_gaps = gap_key
                feedback = {"accepted": False, "policy": finish_policy, "diagnostics": diagnostics,
                            "remaining_turns": max_steps - step - 1,
                            "instruction": "The answer is valid JSON but scientific source delivery is incomplete. "
                            "Use a registered tool to read missing evidence, then cite its actual handle in the "
                            "relevant hypothesis/decision. Do not invent sources or claim a summary is a curve. "
                            "If evidence does not support an edit, omit unsupported hypotheses or retain/defer."}
                if schema_enricher is not None:
                    feedback['consecutive_unchanged_gaps'] = repeated_gaps
                _persist(folder / f"step_{step:02d}.finish_rejected.json", feedback)
                messages.extend([{"role": "assistant", "content": canonical(action)},
                                 {"role": "user", "content": canonical({"finish_feedback": feedback})}])
                continue
            legacy = extract_result(role, action["result"], tools, choices)
            _persist(folder / "wire_result.json", action["result"])
            _persist(folder / f"{role}.json", legacy)
            _persist(folder / "references.json", tools.provenance())
            receipts = [read_json(folder / f"step_{i:02d}.receipt.json") for i in range(step + 1)
                        if (folder / f"step_{i:02d}.receipt.json").exists()]
            backend = "real_api" if len(receipts) == step + 1 and all(r.get("backend") == "real_api" for r in receipts) else "injected_client"
            _persist(folder / "completed.json", {"interface_version": interface_policy, "role": role, "steps": step + 1,
                "binding_sha256": digest(folder / "binding.json"), "response_sha256": digest(folder / f"{role}.json"),
                "references_sha256": digest(folder / "references.json"), "backend": backend})
            return legacy
        signature = canonical({"tool": action["tool"], "arguments": action["arguments"]})
        if signature in signatures:
            raise WorkflowBudgetError("Repeated identical tool request; no progress")
        signatures.add(signature)
        observation = tools.execute(action["tool"], action["arguments"])
        _persist(folder / f"step_{step:02d}.observation.json", observation)
        messages.extend([{"role": "assistant", "content": canonical(action)},
                         {"role": "user", "content": canonical({"tool_observation": observation,
                          "remaining_turns": max_steps - step - 1,
                          "note": "Finish when sufficient. The last turn must be action=finish."})}])
    raise WorkflowBudgetError("Action budget exhausted; no incumbent masquerading as a successful analysis")
