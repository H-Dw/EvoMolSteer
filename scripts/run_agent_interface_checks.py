"""Small real-API regression suite on prior interface failure classes.

No FLOWR job, no alternative ablation condition, no outcome feedback to roles.
Scientific output correctness beyond provenance is not a format-test conclusion.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

from evomolsteer.ablation.agents import role_prompt, validate_evidence_citations, design_schema
from evomolsteer.ablation.compiler import compile_program
from evomolsteer.agent_workflow import EvidenceTools, run_agent
from evomolsteer.agent_workflow.ablation_adapter import registered_choices
from evomolsteer.agent_workflow.contracts import canonical, validate
from evomolsteer.agent_workflow.reviewer import run_review
from evomolsteer.continuous.m2_skill_api import BASE, effective_signature
from evomolsteer.io import digest, read_json, write_json


def suite(repo, study, output, cases, max_steps):
    repo, study, output = repo.resolve(), study.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = read_json(study / "module_registry.json")["modules"]
    protocol = read_json(study / "activation.json")["protocol"]
    base = read_json(repo / BASE)
    skill = repo / "skills/agent-json-interface/SKILL.md"
    interface = skill.read_text(encoding="utf-8") + "\n" + (skill.parent / "references/wire_examples.md").read_text(encoding="utf-8")
    source = study / "evidence"
    report = {"model": "deepseek-flash", "backend": "real_api", "temperature": 0,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "interface_skill_sha256": digest(skill), "study_source": str(study),
              "suite_type": "interface_regression_not_generation_ablation", "cases": []}
    for case in cases:
        tools = EvidenceTools(source)
        directory = output / case
        active = [m["id"] for m in registry if m["role"] == "Analyst" and
                  not (case == "drop" and m["id"] == "A1.1")]
        if case == "findings":
            records = tools.execute("get_evidence", {"source_ids": [f"regional_effects:{i}" for i in range(14)]})
            task = {"test_type": "output-format-only", "records": records,
                    "request": "Return exactly 14 separate findings, one concise factual description for each supplied row, "
                               "with its actual handle. No hypotheses are needed. This is a formatting regression, not scientific selection."}
            instructions = "You are Analyst in an output-format regression test. Use the supplied rows only.\n" + interface
        else:
            task = {"objective": tools.packet["objective"], "request": "Interpret supplied training evidence for coordinate reward design. "
                    "Query the registered tools for necessary evidence and definitions; finish within the stated turn budget. "
                    "Keep substantive unsupported questions in limitations. No current outcome data are available.",
                    "tool_turn_budget": max_steps - 1}
            instructions = role_prompt(repo, study, "Analyst", active) + "\n" + interface
        print("START", case, flush=True)
        result = run_agent("Analyst", tools, instructions, task, directory / "Analyst",
                           max_steps=max_steps, max_output_tokens=6144)
        validate_evidence_citations(result, set(tools.index["ids"]) | set(tools.packet) | {"observable_definitions"})
        if case == "findings" and len(result["findings"]) != 14:
            raise ValueError("Fourteen-row format probe did not preserve all requested rows")
        wire = read_json(directory / "Analyst/wire_result.json")
        count = sum(len(v["evidence_refs"]) for v in wire["findings"] + wire["hypotheses"])
        if case != "findings" and count == 0:
            raise ValueError("An empty answer does not establish source-bound analysis success")
        row = {"case": case, "status": "validated", "active_Analyst_modules": active if case != "findings" else [],
               "steps": read_json(directory / "Analyst/completed.json")["steps"],
               "findings": len(result["findings"]), "hypotheses": len(result["hypotheses"]), "valid_citations": count,
               "instruction_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
               "functions_not_forced_by_interface": True}
        if case == "citation":
            designer_tools = EvidenceTools(source)
            designer_tools.import_references(read_json(directory / "Analyst/references.json"))
            choices = registered_choices(protocol["lanes"]["G"]["allowed"], designer_tools, result)
            design_task = {"objective": tools.packet["objective"], "incumbent": base,
                "fixed_parameters": protocol["temperature_policy"]["parameters"], "maximum_edits": 1,
                "formula": "R=tau*logsumexp(log_pi-rho_delta(q)/tau); q=detached Hungarian endpoint mean square distance, "
                           "rho_delta=delta^2*(sqrt(1+q/delta^2)-1), operative pointcloud_delta_A=1 A by default. "
                           "Real FLOWR endpoint VJP; no affinity-head gradient or extra production forwards. "
                           "teacher_neighbors is a count. Temperatures are frozen. "
                           "Only registered edits are executable; legacy robust_delta is inert."}
            dactive = [m["id"] for m in registry if m["role"] == "Designer"]
            dinstructions = role_prompt(repo, study, "Designer", dactive) + "\n" + interface
            design = run_agent("Designer", designer_tools, dinstructions, design_task, directory / "Designer",
                               choices=choices, handoff=result, max_steps=max_steps, max_output_tokens=4096)
            validate(design, design_schema(protocol["lanes"]["G"]["allowed"], len(tools.packet["teacher_modes"]["landmarks_A"])))
            validate_evidence_citations(design, set(tools.index["ids"]) | set(tools.packet) | {"observable_definitions"})
            # Compiler writes programs under its study. Supply an isolated copied
            # manifest/evidence root, avoiding the active ablation's program queue.
            import shutil
            compiler_study = output / "compiler_study"
            compiler_study.mkdir(exist_ok=True)
            shutil.copy2(study / "activation.json", compiler_study / "activation.json")
            if not (compiler_study / "evidence").exists():
                shutil.copytree(source, compiler_study / "evidence")
            _, _, signature, audit = compile_program(repo, compiler_study, design, base)
            row.update(designer_decision=design["decision"], designer_edit=design["edit"],
                       compiled_effective_sha256=signature, frozen_M2_sha256=effective_signature(base),
                       compiler_unchanged_temperatures=audit["unchanged_temperatures"])
        report["cases"].append(row)
        write_json(output / "summary.json", report)
        print("VALIDATED", case, "steps", row["steps"], flush=True)
    if "citation" in cases and "drop" in cases:
        calls = [read_json(output / c / "Analyst/Analyst.json") for c in ["citation", "drop"]]
        module = next(m for m in registry if m["id"] == "A1.1")
        rubrics = [{"module": "A1.1", "goal": module["label_zh"], "checks": module["functional_rubric"]}]
        all_ids = sorted({i for response in calls for v in response["findings"] + response["hypotheses"] for i in v["evidence_ids"]})
        context_tools = EvidenceTools(source)
        records = [{k: context_tools._preview(v) for k, v in context_tools.records[i].items()}
                   for i in all_ids if i in context_tools.records]
        review = run_review(calls, rubrics, output / "Reviewer.json", input_store=output / "review_inputs",
                            evidence_context={"records": records, "clocks": context_tools.packet["clocks"],
                                              "definitions": context_tools.definitions})
        report["reviewer"] = {"status": "validated", "assessments": len(review["assessments"]),
                              "verbatim_quotes_extracted_by_host": True, "generation_effect_inference": False}
    receipts = [read_json(p) for p in output.rglob("step_*.receipt.json") if "invalid" not in p.name]
    if (output / "Reviewer.receipt.json").exists():
        receipts.append(read_json(output / "Reviewer.receipt.json"))
    report["API"] = {"successful_calls": len(receipts), "format_repairs": sum(len(r.get("interface_repairs", [])) for r in receipts),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in receipts),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in receipts),
        "all_requested_and_returned_models_exact": all(r["requested_model"] == r["returned_model"] == "deepseek-flash" for r in receipts),
        "actual_action_counts_are_not_independent_molecular_experiments": True}
    report["status"] = "passed"
    write_json(output / "summary.json", report)
    print(canonical({"status": report["status"], "cases": cases, "API": report["API"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("docs/experiments/m2_full_drop_ablation_20261012"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", default="citation,findings,drop")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()
    cases = args.cases.split(",")
    if not cases or any(c not in {"citation", "findings", "drop"} for c in cases) or len(set(cases)) != len(cases):
        parser.error("Choose unique prior failure classes: citation,findings,drop")
    suite(Path(__file__).resolve().parents[1], args.study, args.output_dir, cases, args.max_steps)


if __name__ == "__main__":
    main()
