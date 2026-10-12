"""Prepare opt-in prompts, call real roles or launch one selected temporal policy.

Default is local request preparation. No remote connector is invoked here.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from evomolsteer.io import digest, read_json, write_json
from evomolsteer.temporal_guidance.policy import program_signature
from evomolsteer.temporal_guidance.workflow import compile_recipe, inside, prepare, require_execution, run_roles, sources, register_profile_knots


def infer(repo, config, condition, runtime, execute, recipe=None):
    require_execution(config, execute)
    if condition not in config.get("enabled_conditions", []):
        raise ValueError("Condition is not activated")
    if recipe:
        declared = next(c for c in config["conditions"] if c["id"] == condition)
        if recipe not in declared["recipes"]:
            raise ValueError("Numerical control is not declared in this condition")
        selection = compile_control(repo, config, recipe)
        condition += "__" + recipe
    else:
        selection = read_json(inside(repo, config["output"]) / "selections" / (condition + ".json"))
    bank = read_json(inside(repo, config["control_manifest"]))
    if selection["reuse_M2"]:
        result = {"reused": True, "control_manifest": config["control_manifest"],
                  "control_manifest_sha256": digest(inside(repo, config["control_manifest"])),
                  "notice": "本次 Agent 输出未改变执行函数", "new_FLOWR_runs": 0}
        write_json(inside(repo, config["output"]) / "inference" / (condition + ".json"), result)
        return result
    program = inside(repo, selection["selected"]["program"])
    if program_signature(read_json(program)) != selection["selected"]["signature"]:
        raise ValueError("Selected program signature changed")
    artifacts = sources(repo, config)
    if digest(runtime["checkpoint"]) != bank["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen controls")
    inputs = Path(runtime["input_dataset"])
    inputs = inputs / "inputs" if (inputs / "inputs").is_dir() else inputs
    for name, sha in artifacts["static_reference"]["required_input_sha256"].items():
        if digest(inputs / name) != sha:
            raise ValueError("Runtime target/frame differs from the frozen controls")
    batches = config["confirmation_batches"] if runtime["cohort"] == "confirmation" else config["screen_batches"]
    if runtime["cohort"] not in {"screen", "confirmation"}:
        raise ValueError("Use a predeclared matched cohort")
    generation_files = {**bank["generator_sources"], **{
        "src/evomolsteer/generation/" + n: digest(repo / "src/evomolsteer/generation" / n)
        for n in ["scheduled_endpoint_reward.py", "scheduled_dependency_controller.py"]}}
    for name, sha in bank["generator_sources"].items():
        if digest(inside(repo, name)) != sha:
            raise ValueError("Frozen generator source changed: " + name)
    from evomolsteer.temporal_guidance.policy import canonical_hash
    cache_key = canonical_hash({"program_signature": selection["selected"]["signature"],
                                "checkpoint": bank["checkpoint_sha256"], "generator_sources": generation_files,
                                "prior_seed_mode": config["prior_seed_mode"], "master_seed": config["master_seed"],
                                "steps": config["steps"], "batch_size": config["batch_size"]})
    cache_path = inside(repo, config["output"]) / "inference_cache" / (cache_key + ".json")
    cache = read_json(cache_path) if cache_path.exists() else {"cache_key": cache_key, "batches": {}}
    for entry in cache["batches"].values():
        p = Path(entry["COMPLETE"])
        if not p.is_file() or digest(p) != entry["COMPLETE_sha256"]:
            raise ValueError("Cached campaign was retired or changed; restore its verified archive before reuse")
        if digest(entry["guidance_trace"]) != entry["guidance_trace_sha256"]:
            raise ValueError("Cached control trace changed")
    missing = [b for b in batches if str(b) not in cache["batches"]]
    receipt_path = inside(repo, config["output"]) / "inference" / (condition + ".command.json")
    if not missing:
        result = {"reused": True, "cache_key": cache_key, "batches": {str(b): cache["batches"][str(b)] for b in batches},
                  "new_FLOWR_runs": 0, "controls_generated_again": False}
        write_json(receipt_path, result)
        return result
    campaign = "TS_" + cache_key[:16] + "_" + "-".join(map(str, missing))
    command = [sys.executable, str(repo / "scripts/generate_scheduled_dependency_flowr.py"),
               "--allow-execution", "--flowr-root", runtime["flowr_root"],
               "--root", runtime["output_root"], "--checkpoint", runtime["checkpoint"],
               "--input-dataset", str(inputs), "--program", str(program),
               "--reference", str(inside(repo, selection["selected"]["reference"])),
               "--arms", "gradient", "--campaign", campaign,
               "--n", str(len(missing) * config["batch_size"]), "--batch", str(config["batch_size"]),
               "--batch-indices", ",".join(map(str, missing)), "--steps", str(config["steps"]),
               "--seed", str(config["master_seed"]), "--prior-seed-mode", config["prior_seed_mode"], "--export-terminal"]
    receipt = {"command": command, "program_sha256": digest(program),
               "condition": condition, "requested_batches": batches, "new_batches": missing,
               "cached_batches": [b for b in batches if b not in missing], "controls_generated_again": False}
    write_json(receipt_path, receipt)
    subprocess.run(command, cwd=repo, check=True)
    for b in missing:
        complete = Path(runtime["output_root"]) / "results" / campaign / "gradient" / f"batch_{b:03d}" / "COMPLETE.json"
        if read_json(complete)["resampling_steps"]:
            raise ValueError("A supposedly unselected trajectory resampled particles")
        from evomolsteer.storage.evaluation_view import execution_arrays, state_signature
        arrays = execution_arrays(complete.with_name("trajectory.h5"))
        expected_indices = np.tile(np.arange(config["batch_size"]), (config["steps"], 1))
        if arrays["resampled"].any() or not np.array_equal(arrays["selected_indices"], expected_indices):
            raise ValueError("Particle ancestry is not an identity mapping")
        initial_signature = state_signature(arrays)
        if initial_signature != bank["result"]["initial_state_signatures"][f"gradient/batch_{b:03d}"]:
            raise ValueError("Actual candidate initial state differs from the fixed M2 bank")
        trace = complete.with_name("guidance_trace.jsonl")
        cache["batches"][str(b)] = {"COMPLETE": str(complete.resolve()), "COMPLETE_sha256": digest(complete),
                                    "guidance_trace": str(trace.resolve()), "guidance_trace_sha256": digest(trace),
                                    "initial_state_signature": initial_signature, "identity_ancestry_verified": True}
    write_json(cache_path, cache)
    receipt.update(cache_key=cache_key, batches={str(b): cache["batches"][str(b)] for b in batches})
    write_json(receipt_path, receipt)
    return receipt


def compile_control(repo, config, recipe):
    """Registered mathematical controls are not mislabeled as Agent decisions."""
    program, reference = compile_recipe(repo, config, recipe)
    target = inside(repo, config["output"]) / "compiled_controls" / recipe / "program.json"
    write_json(target, program)
    baseline = program_signature(sources(repo, config)["base_program"])
    signature = program_signature(program)
    result = {"selected": {"program": target.relative_to(repo).as_posix(), "reference": reference, "signature": signature},
              "reuse_M2": signature == baseline, "origin": "predeclared deterministic numerical control; no Agent call"}
    write_json(target.parent / "receipt.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", choices=["prepare", "compile-control", "register-profile", "agents", "infer"], default="prepare")
    p.add_argument("--condition")
    p.add_argument("--runtime-config")
    p.add_argument("--recipe", help="Predeclared numerical control; not an Agent-produced decision")
    p.add_argument("--profile")
    p.add_argument("--feature")
    p.add_argument("--new-recipe")
    p.add_argument("--new-config")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = read_json(inside(repo, args.config))
    if args.mode == "prepare":
        prepare(repo, config)
    elif args.mode == "compile-control":
        if not args.recipe or args.recipe not in config["recipes"]:
            p.error("Provide a registered --recipe")
        compile_control(repo, config, args.recipe)
    elif args.mode == "register-profile":
        if not all([args.profile, args.feature, args.recipe, args.new_recipe, args.new_config]):
            p.error("Registering profile knots needs --profile --feature --recipe --new-recipe --new-config")
        new = register_profile_knots(repo, config, args.profile, args.feature, args.recipe, args.new_recipe)
        new["output"] = config["output"] + "__" + args.new_recipe
        for c in new["conditions"]:
            if args.recipe in c["recipes"]:
                c["recipes"].append(args.new_recipe)
        write_json(inside(repo, args.new_config), new)
    else:
        if not args.condition or args.condition not in {c["id"] for c in config["conditions"]}:
            p.error("Provide a declared --condition")
        if args.mode == "agents":
            run_roles(repo, config, args.condition, execute=args.execute)
        else:
            if not args.runtime_config:
                p.error("Inference needs --runtime-config")
            infer(repo, config, args.condition, read_json(args.runtime_config), args.execute, args.recipe)


if __name__ == "__main__":
    main()
