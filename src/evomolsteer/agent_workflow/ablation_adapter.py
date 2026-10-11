"""Opt-in integration with the existing scientific role and reward compiler APIs."""
import copy
import re
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from .contracts import POLICY
from .engine import run_agent
from .sources import EvidenceTools


def registered_choices(allowed, tools, analyst):
    """The host enumerates legal edits, including source-bound regional indices."""
    regions = set()
    for hypothesis in analyst.get("hypotheses", []) + analyst.get("findings", []):
        for identifier in hypothesis.get("evidence_ids", []):
            record = tools.records.get(identifier, {})
            region = tools.index["regions"].get(identifier)
            match = re.match(r"landmark_(\d+)_", str(record.get("feature", "")))
            if region is not None:
                regions.add(int(region))
            if match:
                regions.add(int(match.group(1)))
    choices = {}
    for parameter, values in allowed.items():
        if parameter in {"region_index", "temporal_region_index"}:
            values = sorted(regions)
        if not isinstance(values, (list, tuple)):
            raise ValueError("Unsupported compiler parameter rule")
        for value in values:
            choices[f"C{len(choices) + 1:03d}"] = {"parameter": parameter, "value": value}
    return choices


def call_role_tools(repo, study, case, chain, role, active, *, alternative=None,
                    analyst=None, incumbent=None, lane="G", delivery=None, caller=None,
                    max_steps=10, max_output_tokens=4096):
    from evomolsteer.ablation.agents import role_prompt, operative_incumbent, validate_evidence_citations
    from evomolsteer.ablation.protocol import validate_activation
    from evomolsteer.deepseek_api import call_json
    repo, study = Path(repo), Path(study)
    activation = validate_activation(repo, study)
    protocol = activation["protocol"]
    if protocol.get("comparison_policy", {}).get("treatments") == ["full", "drop"] and alternative is not None:
        raise ValueError("Replacement treatments are prohibited")
    folder = study / "calls" / case / f"chain_{chain:02d}"
    tools = EvidenceTools(study / "evidence")
    if role == "Designer":
        if analyst is None:
            raise ValueError("Designer requires the actual Analyst response")
        analysis_path = folder / "Analyst.tools_v1" / "Analyst.json"
        if read_json(analysis_path) != analyst:
            raise ValueError("Analyst handoff differs from its extracted source response")
        completion = read_json(analysis_path.parent / "completed.json")
        if (completion["response_sha256"] != digest(analysis_path) or
            completion["references_sha256"] != digest(analysis_path.parent / "references.json")):
            raise ValueError("Analyst response/reference receipt checksum mismatch")
        tools.import_references(read_json(analysis_path.parent / "references.json"))
    prompt = role_prompt(repo, study, role, active, alternative)
    interface_path = repo / "skills/agent-json-interface/SKILL.md"
    prompt += "\n\n" + interface_path.read_text(encoding="utf-8")
    prompt += "\n\n" + (interface_path.parent / "references/wire_examples.md").read_text(encoding="utf-8")
    task = {"role": role, "case": case, "chain": chain, "lane": lane,
            "maximum_edits": 1, "objective": tools.packet["objective"],
            "incumbent": operative_incumbent(incumbent),
            "fixed_parameters": protocol["temperature_policy"]["parameters"],
            "delivery_diagnostics": delivery,
            "formula": "R=tau*logsumexp(log_pi-rho_delta(q)/tau), q=detached-Hungarian endpoint mean squared distance; "
                       "rho_delta=delta^2*(sqrt(1+q/delta^2)-1). Actual pointcloud_delta_A defaults to 1 A. "
                       "Native endpoint VJP, no affinity-head derivative, no extra model forward. "
                       "teacher_neighbors is a count; temporal_recipe changes only a registered scalar/attention, not velocity. "
                       "No temperature edits. Inert legacy metadata is not a reward term."}
    choices = registered_choices(protocol["lanes"][lane]["allowed"], tools, analyst) if role == "Designer" else None
    result = run_agent(role, tools, prompt, task, folder / (role + ".tools_v1"), choices=choices,
                       handoff=analyst, max_steps=max_steps, max_output_tokens=max_output_tokens,
                       caller=caller or call_json)
    known = set(tools.index["ids"]) | set(tools.packet) | {"observable_definitions"}
    validate_evidence_citations(result, known)
    path = folder / (role + ".json")
    if path.exists() and read_json(path) != result:
        raise ValueError("A previous interface response exists; use a new case ID")
    write_json(path, result)
    (folder / (role + ".instructions.md")).write_text(prompt, encoding="utf-8")
    write_json(folder / (role + ".delivery.json"), {"active_modules": active, "alternative": alternative,
        "interface_version": POLICY, "instruction_sha256": digest(folder / (role + ".instructions.md")),
        "evidence_sha256": activation["packet_sha256"], "role": role,
        "model": "deepseek-flash", "backend": "real_api", "schema_forces_module_function": False,
        "workflow_relative_path": role + ".tools_v1"})
    return result


def review_tools(study, case, role, focal_ids, *, caller=None):
    from evomolsteer.deepseek_api import call_json
    from evomolsteer.ablation.agents import validate_review
    from .reviewer import run_review
    study = Path(study)
    modules = [m for m in read_json(study / "module_registry.json")["modules"]
               if m["role"] == role and m["id"] in focal_ids]
    calls = [read_json(study / "calls" / case / f"chain_{i:02d}" / (role + ".json")) for i in range(6)]
    tools = EvidenceTools(study / "evidence")
    identifiers = sorted({identifier for response in calls
        for value in response.get("findings", []) + response.get("hypotheses", [])
        for identifier in value.get("evidence_ids", [])} |
        {identifier for response in calls for identifier in response.get("evidence_ids", [])})
    context = {"clocks": tools.packet["clocks"], "definitions": tools.definitions,
               "records": [{k: tools._preview(v) for k, v in tools.records[i].items()}
                           for i in identifiers if i in tools.records],
               "limitations": tools.packet["limitations"]}
    output = study / "reviews" / (case + "_" + role + ".tools_v1.json")
    rubrics = [{"module": m["id"], "goal": m["label_zh"], "checks": m["functional_rubric"]} for m in modules]
    result = run_review(calls, rubrics, output, input_store=study / "inputs", evidence_context=context,
                        caller=caller or call_json)
    validate_review(result, calls, modules)
    path = study / "reviews" / (case + "_" + role + ".json")
    if path.exists() and read_json(path) != result:
        raise ValueError("Existing review belongs to a different interface")
    write_json(path, result)
    return result
