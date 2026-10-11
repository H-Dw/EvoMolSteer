"""Quarantined scientific diagnostics through the production JSON/tool loop."""
import copy
from pathlib import Path

import jsonschema

from evomolsteer.deepseek_api import call_json
from evomolsteer.io import digest, read_json, write_json
from .contracts import POLICY, canonical, obj, strict_json, format_contract
from .engine import run_agent
from .sources import EvidenceTools


def fixture_sources(study, destination, *, contrast=None, unit=None):
    """Bind transformed copies to new hashes; never change production sources."""
    from evomolsteer.ablation.diagnostics import contrast_fixture, geometry_fixture
    study, destination = Path(study), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    packet = read_json(study / "evidence/packet.json")
    index = read_json(study / "evidence/index.json")
    curves = read_json(study / "evidence/continuous_functions.json")
    if contrast is not None:
        packet = contrast_fixture(packet, contrast)
        curves["functions"] = {}
    elif unit is not None:
        fixture = geometry_fixture(packet, unit)
        packet = copy.deepcopy(packet)
        packet.update(fixture)
        packet["tables"] = {}
        packet["teacher_modes"]["geometry_unit"] = unit
        curves["functions"] = {}
    else:
        raise ValueError("Explicit isolated diagnostic transformation required")
    ids = [row[table["columns"].index("evidence_id")]
           for table in packet["tables"].values() for row in table["rows"]]
    index["ids"] = ids
    index["regions"] = {key: value for key, value in index["regions"].items() if key in ids}
    for name, value in [("packet.json", packet), ("index.json", index),
                        ("continuous_functions.json", curves),
                        ("observable_definitions.json", read_json(study / "evidence/observable_definitions.json"))]:
        path = destination / name
        if path.exists() and read_json(path) != value:
            raise ValueError("Diagnostic fixture changed after its first request")
        write_json(path, value)
    return destination


def run_tool_probes(repo, study, *, caller=call_json):
    from evomolsteer.ablation.agents import role_prompt, operative_incumbent
    from evomolsteer.ablation.protocol import validate_activation
    from evomolsteer.continuous.m2_skill_api import BASE
    repo, study = Path(repo), Path(study)
    protocol = validate_activation(repo, study)["protocol"]
    output = study / "diagnostics/tools_v1"
    interface = repo / "skills/agent-json-interface"
    suffix = "\n\n" + (interface / "SKILL.md").read_text(encoding="utf-8") + "\n" + (
        interface / "references/wire_examples.md").read_text(encoding="utf-8")
    responses = []
    for mode in ["observed", "sign_changed", "missing"]:
        source = fixture_sources(study, output / (mode + ".sources"), contrast=mode)
        tools = EvidenceTools(source)
        result = run_agent("Analyst", tools, role_prompt(repo, study, "Analyst", ["A1.1"]) + suffix,
            {"diagnostic_only": True, "no_compilation": True, "objective": tools.packet["objective"],
             "task": "Analyze only this isolated supplied evidence with the role interface. No generation follows.",
             "limitations": tools.packet["limitations"]}, output / ("Analyst_" + mode),
            max_steps=10, max_output_tokens=8192, caller=caller)
        responses.append({"id": mode, "role": "Analyst", "input": tools.packet, "response": result,
                          "sha256": digest(output / ("Analyst_" + mode) / "Analyst.json")})
    for unit in ["A", "nm"]:
        source = fixture_sources(study, output / (unit + ".sources"), unit=unit)
        tools = EvidenceTools(source)
        choices = {}
        for parameter, values in {"teacher_neighbors": [2, 4, 8], "pointcloud_delta_A": [.5, 1., 2.]}.items():
            for value in values:
                choices[f"C{len(choices) + 1:03d}"] = {"parameter": parameter, "value": value}
        result = run_agent("Designer", tools, role_prompt(repo, study, "Designer", ["D2.1"]) + suffix,
            {"diagnostic_only": True, "no_compilation": True, "parameter_unit": "A",
             "incumbent": operative_incumbent(read_json(repo / BASE)),
             "fixed_parameters": protocol["temperature_policy"]["parameters"],
             "task": "Interpret the supplied isolated teacher dispersion in physical units; answer the Designer interface. No generation follows."},
            output / ("Designer_unit_" + unit), choices=choices, max_steps=10,
            max_output_tokens=8192, caller=caller)
        responses.append({"id": unit, "role": "Designer", "input": tools.packet["teacher_modes"],
                          "response": result,
                          "sha256": digest(output / ("Designer_unit_" + unit) / "Designer.json")})
    names = ["signed_contrast_response", "missing_claim_withheld", "physical_unit_consistency"]
    schema = obj({"checks": {"type": "array", "items": obj({
        "id": {"enum": names}, "passed": {"type": "boolean"},
        "explanation": {"type": "string", "maxLength": 600}})}})
    def validator(value):
        if len(value["checks"]) != 3 or {x["id"] for x in value["checks"]} != set(names):
            raise jsonschema.ValidationError("Every diagnostic check must appear exactly once")
    messages = [{"role": "system", "content": "You are an interface Reviewer. Return JSON only. "
                 "Use actual inputs and outputs, not performance. Do not require an edit or a predetermined ranking."},
                {"role": "user", "content": canonical({"cases": responses, "checks": {
                    names[0]: "Observed/sign-changed contrasts must use their actual signs; interpretation may change.",
                    names[1]: "Missing supplied contrasts must cause unsupported enrichment claims to be withheld.",
                    names[2]: "Equivalent Angstrom/nm scales must be interpreted consistently in physical units. A tenfold unit error fails; exploratory choices may differ."}})}]
    result = caller(messages, schema, output / "functional_preflight.json", input_store=output / "inputs",
        max_output_tokens=4096, response_parser=strict_json, response_validator=validator,
        format_policy=POLICY, format_builder=format_contract)
    validator(result)
    write_json(output / "quarantine.json", {"no_compilation": True, "no_inference": True,
        "interface": "evidence-tools-v1", "production_evidence_sha256": digest(study / "evidence/packet.json"),
        "responses": {x["id"] + x["role"]: x["sha256"] for x in responses},
        "review_sha256": digest(output / "functional_preflight.json"),
        "passed": all(x["passed"] for x in result["checks"]), "not_module_performance_evidence": True})
    if not all(x["passed"] for x in result["checks"]):
        raise ValueError("Tool workflow did not respond correctly to isolated functional probes")
    return result
