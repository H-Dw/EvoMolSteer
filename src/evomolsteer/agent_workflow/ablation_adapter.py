"""Opt-in integration with the existing scientific role and reward compiler APIs."""
import copy
import re
from pathlib import Path

from evomolsteer.io import digest, read_json, write_json
from .contracts import POLICY
from .engine import run_agent
from .sources import EvidenceTools


def analyst_source(study, case, chain, response):
    """Resolve a same-chain handoff or explicitly registered frozen pool source.

    A Designer deletion must reuse the actual selected Analyst, not require a
    fictitious new Analyst in its own condition directory.
    """
    study = Path(study).resolve()
    outer = study / "calls" / case / f"chain_{chain:02d}" / "Analyst.json"
    if not outer.exists():
        pool = read_json(study / "selected_Analyst_pool.json")
        rows = pool["outputs"]
        if len(rows) != 6 or not 0 <= chain < 6:
            raise ValueError("Six chain-indexed frozen Analyst sources required")
        # Registered paths are repository-relative; bind them to this study.
        row = rows[chain]
        parts = Path(row["path"]).parts
        relative = study.name
        position = parts.index(relative) if relative in parts else -1
        if position < 0:
            raise ValueError("Frozen Analyst path is outside this study")
        outer = study.joinpath(*parts[position + 1:]).resolve()
        if not outer.is_relative_to(study / "calls") or digest(outer) != row["sha256"]:
            raise ValueError("Frozen Analyst source changed or escaped this study")
        if outer.parent.name != f"chain_{chain:02d}":
            raise ValueError("Frozen handoff must preserve the actual chain index")
    inner = outer.parent / "Analyst.tools_v1" / "Analyst.json"
    completion = read_json(inner.parent / "completed.json")
    if read_json(outer) != response or read_json(inner) != response:
        raise ValueError("Analyst handoff differs from its extracted source response")
    if (completion["backend"] != "real_api" or
        completion["response_sha256"] != digest(inner) or
        completion["binding_sha256"] != digest(inner.parent / "binding.json") or
        completion["references_sha256"] != digest(inner.parent / "references.json")):
        raise ValueError("Analyst source is not a verified actual API workflow")
    return inner


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
                    max_steps=10, max_output_tokens=8192):
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
        analysis_path = analyst_source(study, case, chain, analyst)
        tools.import_references(read_json(analysis_path.parent / "references.json"))
    prompt = role_prompt(repo, study, role, active, alternative)
    interface_path = repo / "skills/agent-json-interface/SKILL.md"
    prompt += "\n\n" + interface_path.read_text(encoding="utf-8")
    prompt += "\n\n" + (interface_path.parent / "references/wire_examples.md").read_text(encoding="utf-8")
    task = {"role": role, "case": case, "chain": chain, "lane": lane,
            "task_wording": ("Interpret the supplied evidence and answer the coordinate-design task." if chain < 3 else
                "Use these observations to provide your response to the supplied molecular coordinate task."),
            "maximum_edits": 1, "objective": tools.packet["objective"],
            "incumbent": operative_incumbent(incumbent),
            "fixed_parameters": protocol["temperature_policy"]["parameters"],
            "delivery_diagnostics": delivery,
            "formula": "R=tau*logsumexp(log_pi-rho_delta(q)/tau), q=detached-Hungarian endpoint mean squared distance; "
                       "rho_delta=delta^2*(sqrt(1+q/delta^2)-1). Actual pointcloud_delta_A defaults to 1 A. "
                       "Native endpoint VJP, no affinity-head derivative, no extra model forward. "
                       "teacher_neighbors is a count; temporal_recipe changes only a registered scalar/attention, not velocity. "
                       "No temperature edits. Inert legacy metadata is not a reward term.",
            "formula_registry": {
                "prior": "Detached nearest-k teachers; log_pi=-q_anchor/prior_temperature; scores beta=0; k is a count.",
                "regional": "Detached background plus Gaussian landmark attention; positive surrounding weight.",
                "temporal_region": "Same static teachers; validated absolute contrast changes regional attention at observed score nodes.",
                "temporal_teacher": "(1-blend)*R_static + blend*R_observed_teacher(time); same shared FLOWR endpoint, no velocity target.",
                "dose": "Endpoint-VJP force normalized and RMS-scaled against native predictive update; eta/ramp only in U lane.",
                "diagnostic_limit": "Moment effects, pair kernels and turnover are diagnostic evidence. Executing pointcloud scalar has no pair-kernel, moment, joint-motif or geometry-block term; legacy robust_delta is inert."}}
    choices = registered_choices(protocol["lanes"][lane]["allowed"], tools, analyst) if role == "Designer" else None
    budget = protocol.get("agent_workflow", {})
    result = run_agent(role, tools, prompt, task, folder / (role + ".tools_v1"), choices=choices,
                       handoff=analyst, max_steps=budget.get("max_steps", max_steps),
                       max_output_tokens=budget.get("max_output_tokens", max_output_tokens),
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
        "workflow_relative_path": role + ".tools_v1",
        "Analyst_source": analysis_path.relative_to(study).as_posix() if role == "Designer" else None})
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
