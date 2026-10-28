"""Offline illustrative decisions, never a molecular or model experiment.

Use the exact pure legality functions from the committed internal-distance
protocol, not concurrent edits or any sampling entrypoint. All metric values,
domain limits and the multiplier below are explicitly hypothetical fixtures.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

COMMIT = "da1b535ad0f71fb9e37e37773d6d0c58234dff4e"
ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).with_name("simulation.json")


def source(path):
    return subprocess.run(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT,
                          check=True, capture_output=True).stdout.decode("utf-8")


def selected_definitions(text, names, namespace):
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            nodes.append(node)
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    found.update(target.id for node in nodes if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name))
    if found != names:
        raise ValueError(f"Missing pinned definitions: {names - found}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-pure-definitions", "exec"), namespace)


def main():
    protocol = source("agenticgen/src/agenticgen/uniguide/geometry_protocol.py")
    ordinary = source("agenticgen/src/agenticgen/uniguide/protocol.py")
    io_source = source("agenticgen/src/agenticgen/uniguide/io.py")
    namespace = {"math": math, "json": json, "hashlib": hashlib}
    selected_definitions(io_source, {"value_sha"}, namespace)
    selected_definitions(ordinary, {"BRANCHES"}, namespace)
    selected_definitions(protocol, {"POLICY", "NORMALIZED", "EFFECTIVE", "MULTIPLIER", "ticks", "effective",
                                   "validate_domain", "validate_action"}, namespace)
    key = namespace["NORMALIZED"]
    body = dict(policy=namespace["POLICY"], teacher_id="SIMULATED_TEACHER",
                condition_sha256="HYPOTHETICAL_NO_MOLECULE", calibrated_guidance_multiplier=1.,
                sampling_protocol_sha256="HYPOTHETICAL_NO_SAMPLING",
                normalized_minimum=-2., normalized_maximum=2.)
    body["calibration_key"] = namespace["value_sha"](
        [body["teacher_id"], body["condition_sha256"], body["sampling_protocol_sha256"]])
    domain = dict(body, domain_sha256=namespace["value_sha"](body))
    history = [
        dict(id="simulated/baseline", **{key: [0., 0., 0., 0.]},
             controlled_atom_pair_distance_root_mean_square_deviation=.34,
             controlled_to_neighborhood_distance_root_mean_square_deviation=.14),
        dict(id="simulated/stronger_second_interval", **{key: [0., 2., 0., 0.]},
             controlled_atom_pair_distance_root_mean_square_deviation=.30,
             controlled_to_neighborhood_distance_root_mean_square_deviation=.18),
        dict(id="simulated/weaker_second_interval", **{key: [0., -2., 0., 0.]},
             controlled_atom_pair_distance_root_mean_square_deviation=.38,
             controlled_to_neighborhood_distance_root_mean_square_deviation=.10),
    ]
    anchors = {row["id"]: row for row in history}
    seen = [row[key] for row in history]
    specifications = [
        ("increase", [1, 0, 0, 0]), ("increase", [0, 0, 1, 0]),
        ("decrease", [0, -1, 0, 0]), ("decrease", [0, 0, 0, -1]),
        ("redistribute", [0, -1, 1, 0]), ("redistribute", [0, -1, 0, 1]),
    ]
    candidates = []
    resolved = []
    for index, (branch, schedule) in enumerate(specifications):
        action = dict(id=f"simulated/candidate-{index}", branch=branch,
                      anchor_id="simulated/baseline", updates=[
                          dict(window_id=window, step=step)
                          for window, step in enumerate(schedule) if step])
        result = namespace["validate_action"](action, anchors, domain, seen + resolved)
        assert result == schedule
        candidates.append(action)
        resolved.append(result)
    bad_actions = {
        "renamed_duplicate": dict(id="new-name", branch="increase",
                                  anchor_id="simulated/baseline", updates=[dict(window_id=1, step=2)]),
        "reference_label_as_anchor": dict(candidates[0], anchor_id="r80"),
        "wrong_direction": dict(candidates[0], branch="decrease"),
        "outside_domain": dict(candidates[0], anchor_id="simulated/stronger_second_interval",
                               updates=[dict(window_id=1, step=1)]),
    }
    rejections = {}
    for name, action in bad_actions.items():
        try:
            namespace["validate_action"](action, anchors, domain, seen)
        except ValueError as error:
            rejections[name] = str(error)
        else:
            raise AssertionError(f"Invalid scenario accepted: {name}")
    record = dict(
        evidence_status="HYPOTHETICAL: no model response or molecular measurement",
        protocol_commit=COMMIT,
        protocol_source_sha256=hashlib.sha256(protocol.encode("utf-8")).hexdigest(),
        domain=domain, history=history, legal_candidates=candidates,
        executor_resolved_schedules=resolved, rejected_examples=rejections,
        human_preference_status="pending; no recommendation accepted by default",
        selected_portfolio=None,
        illustrative_future_observation=dict(
            evidence_status="HYPOTHETICAL, NOT OBSERVED", candidate_id=candidates[2]["id"],
            controlled_atom_pair_distance_root_mean_square_deviation=.36,
            controlled_to_neighborhood_distance_root_mean_square_deviation=.11,
            interpretation="Direction matches the local tradeoff hypothesis in this fixture; uncertainty and generalization remain unassessed"),
        new_model_calls=0, new_candidates_generated=0, new_sampling_attempts=0,
        note="Integer updates are pinned to the stated commit; complete schedules are calculated by its validator, not returned by the Agent. Multiplier and bounds are illustrative only.")
    OUTPUT.write_text(json.dumps(record, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(OUTPUT), legal_proposals=len(candidates),
                          rejected_invalid_examples=len(rejections), experimental_calls=0)))


if __name__ == "__main__":
    main()
