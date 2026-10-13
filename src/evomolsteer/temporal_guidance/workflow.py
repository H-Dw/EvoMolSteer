"""Separate opt-in prompt workflow and compiler for temporal M2 experiments.

prepare() renders requests without API calls. run_roles() is explicitly gated.
The existing compact/G-lane workflow and its frozen controller are not changed.
"""
import copy
import gzip
from collections import Counter
from pathlib import Path

import numpy as np

from ..agent_workflow.contracts import obj
from ..agent_workflow.engine import run_agent
from ..agent_workflow.sources import EvidenceTools
from ..io import digest, read_json, write_json
from .policy import canonical_hash, make_policy, policy_is_identity, program_signature
from .evidence import FINISH_POLICY, analyst_finish, designer_finish, temporal_sources
from .evidence_protocol import PROTOCOL, INTERFACE, TemporalEvidenceProtocol

EVIDENCE_FILES = ("packet.json", "index.json", "continuous_functions.json", "observable_definitions.json")


def inside(repo, name):
    p = (repo / name).resolve()
    if not p.is_relative_to(repo.resolve()):
        raise ValueError("Study artifacts must stay inside the specified repository")
    return p


def load_reference(path):
    import json
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def sources(repo, config):
    result = {}
    for key in ("base_program", "static_reference", "temporal_reference"):
        p = inside(repo, config[key])
        if digest(p) != config["source_sha256"][key]:
            raise ValueError("Frozen scientific artifact mismatch: " + key)
        result[key] = read_json(p) if key == "base_program" else load_reference(p)
    p, s, d = [result[k] for k in ("base_program", "static_reference", "temporal_reference")]
    if p["reference_sha256"] != config["source_sha256"]["static_reference"] or p["window"] != s["window"]:
        raise ValueError("Base M2/reference mismatch")
    if p["reward_view"] != "endpoint_pointcloud" or p.get("time_ramp_power", 0) != 0 or p.get("flow_control"):
        raise ValueError("This study requires the un-ramped, unprojected M2 incumbent")
    if s["window"] != d["window"] or s["times"] != d["times"] or s["required_input_sha256"] != d["required_input_sha256"]:
        raise ValueError("Temporal/static provenance mismatch")
    for a, b in zip(s["frames"], d["frames"], strict=True):
        if not np.array_equal(a["teacher_endpoint_A"], b["teacher_endpoint_A"]):
            raise ValueError("Temporal recipe changes frozen static teachers")
    evidence = inside(repo, config["evidence"])
    if read_json(evidence / "packet.json")["clocks"]["learning_window"] != s["window"]:
        raise ValueError("Analysis and guidance learning windows differ")
    bank = read_json(inside(repo, config["control_manifest"]))
    if config["steps"] != bank["steps"] or config["master_seed"] != bank["master_seed"] or config["prior_seed_mode"] != bank["prior_seed_mode"]:
        raise ValueError("Frozen comparison RNG/integration policy changed")
    batches = config["confirmation_batches"]
    if config["batch_size"] != bank["batch_size"] or len(set(batches)) != len(batches) or any(b not in bank["batches"] for b in batches):
        raise ValueError("Use the indexed frozen comparison batches")
    if not set(config["screen_batches"]).issubset(batches) or len(set(config["screen_batches"])) != len(config["screen_batches"]):
        raise ValueError("Screening must use an unrepeated subset of the confirmation RNG bank")
    return result


class TemporalEvidenceTools(EvidenceTools):
    """Optional compact spline profiles; source-bound retrieval, no arbitrary code."""
    def __init__(self, evidence_dir, profiles=None, *, bulk_curves=False):
        super().__init__(evidence_dir)
        if bulk_curves:
            self.schemas["get_curves"] = obj({"features": {"type": "array", "minItems": 1,
                "maxItems": 8, "uniqueItems": True, "items": {"enum": sorted(self.curves["functions"])}}})
        self.profile_path = Path(profiles).resolve() if profiles else None
        if self.profile_path:
            self.profiles = read_json(self.profile_path)
            if self.profiles["window"] != self.curves["window"] or self.profiles["score_times"] != self.curves["times"]:
                raise ValueError("Profile clocks differ from the frozen selection evidence")
            if self.profiles["input_sha256"]["packet"] != self.files["packet.json"]:
                raise ValueError("Profiles were mined for another evidence packet")
            self.files["stage_profiles.json"] = digest(self.profile_path)
            self.schemas["get_stage_profile"] = obj({"feature": {"enum": sorted(self.profiles["functions"])}})

    def catalogue(self):
        data = super().catalogue()
        data["temporal_profiles_available"] = self.profile_path is not None
        data["stage_profiles_available"] = self.profile_path is not None
        data["curves_available"] = bool(self.curves["functions"])
        data["profile_warning"] = "Stage profiles are optional. Existing continuous curves are independently available " \
                                  "through get_curve; a missing stage profile does not mean curves are absent."
        return data

    def _get_curves(self, features):
        return {"records": [self._get_curve(feature) for feature in features]}

    def _get_stage_profile(self, feature):
        pointer = "/functions/" + feature.replace("~", "~0").replace("/", "~1")
        return self._deliver("temporal_profile/" + feature, "stage_profiles.json", pointer,
                             self.profiles["functions"][feature], kind="stage_profile")

    def _source_value(self, identity):
        if identity["kind"] == "stage_profile":
            return self.profiles["functions"][identity["source_id"][len("temporal_profile/"):]]
        return super()._source_value(identity)

    def verify_sources(self):
        for name, sha in self.files.items():
            path = self.profile_path if name == "stage_profiles.json" else self.root / name
            if digest(path) != sha:
                raise ValueError("Changed immutable temporal evidence")


def render_instructions(repo, config, condition, role):
    paths = [config["base_skills"][role], *condition["extensions"][role],
             config["contract"], "skills/agent-json-interface/SKILL.md"]
    paths.extend(config.get("delivery_skills", []))
    return "\n\n".join(inside(repo, p).read_text(encoding="utf-8") for p in paths), {
        p: digest(inside(repo, p)) for p in paths}


def compile_recipe(repo, config, recipe_id):
    artifacts = sources(repo, config)
    recipe = config["recipes"][recipe_id]
    p = copy.deepcopy(artifacts["base_program"])
    s = artifacts["static_reference"]
    policy = make_policy(s["window"], s["times"], config["steps"], recipe["route"],
                         knots=recipe["knots"], dose=recipe["dose"], blend=recipe["blend"], delta=recipe["delta"])
    if recipe.get("profile_source"):
        profile = read_json(inside(repo, recipe["profile_source"]["path"]))
        if digest(inside(repo, recipe["profile_source"]["path"])) != recipe["profile_source"]["sha256"]:
            raise ValueError("Changed stage-boundary profile")
        if not profile["functions"][recipe["profile_source"]["feature"]]["accepted_for_stage_hypothesis"]:
            raise ValueError("Unstable data-derived stage knots are not eligible")
        a, b = s["times"][0], s["times"][-1]
        expected = [0., *[(t - a) / (b - a) for t in profile["functions"][recipe["profile_source"]["feature"]]["knot_times"]], 1.]
        if recipe["knots"] != expected:
            raise ValueError("Claimed data-derived knots differ from the cited profile")
        policy["knots_origin"] = "exploratory_nested_batch_CV_profile: " + recipe["profile_source"]["feature"]
    if policy_is_identity(policy):
        return p, config["static_reference"]
    reference = config["temporal_reference"] if any(v > 0 for v in policy["teacher_blend"]) else config["static_reference"]
    p["reference_sha256"] = digest(inside(repo, reference))
    p["temporal_policy"] = policy
    p["temporal_policy_sha256"] = canonical_hash(policy)
    return p, reference


def _catalogue(repo, config, condition):
    result = {}
    for name in condition["recipes"]:
        recipe = config["recipes"][name]
        if recipe["route"] != condition["route"] and not (condition["route"] == "combined"):
            raise ValueError("Conditions may not silently combine dose and reward changes")
        p, ref = compile_recipe(repo, config, name)
        result[name] = {"parameter": "temporal_recipe", "value": name,
                        "route": recipe["route"], "description": recipe["description"],
                        "numerical_signature": program_signature(p), "reference": ref,
                        "policy": p.get("temporal_policy"), "is_M2": "temporal_policy" not in p}
    return result


def _prompt_choices(config, catalogue):
    """Compact registered intervention levels; full numeric tables stay host-side."""
    return {name: {k: item[k] for k in
            ["parameter", "value", "route", "description", "numerical_signature", "is_M2"]} |
            {"schedule": {k: config["recipes"][name][k] for k in ["knots", "dose", "blend", "delta"]},
             "normalization": "dose is normalized over actual score nodes to a mean of one",
             "interpolation": "smoothstep; phase is normalized over observed score nodes; no extrapolation"}
            for name, item in catalogue.items()}


def prepare(repo, config):
    """Pure local request preparation; never calls a role, SSH or FLOWR."""
    repo = Path(repo).resolve()
    protocol = config.get("evidence_protocol", FINISH_POLICY)
    if protocol not in {FINISH_POLICY, PROTOCOL}:
        raise ValueError("Unknown temporal evidence protocol")
    artifacts = sources(repo, config)
    out = inside(repo, config["output"])
    evidence = inside(repo, config["evidence"])
    binding = {"schema_version": "temporal-study-binding-1.0", "source_sha256": config["source_sha256"],
               "config_sha256": canonical_hash(config), "evidence": {n: digest(evidence / n) for n in EVIDENCE_FILES},
               "control_manifest_sha256": digest(inside(repo, config["control_manifest"])),
               "code_sha256": {"workflow": digest(__file__), "policy": digest(Path(__file__).with_name("policy.py"))},
               "execution_enabled": config["execution_enabled"], "conditions": {}}
    binding["code_sha256"]["evidence_delivery"] = digest(Path(__file__).with_name("evidence.py"))
    binding["code_sha256"]["agent_engine"] = digest(Path(__file__).parents[1] / "agent_workflow/engine.py")
    binding["finish_policy"] = protocol
    if protocol == PROTOCOL:
        binding["code_sha256"].update(
            evidence_protocol=digest(Path(__file__).with_name("evidence_protocol.py")),
            json_syntax=digest(Path(__file__).parents[1] / "json_syntax.py"),
            API=digest(Path(__file__).parents[1] / "deepseek_api.py"),
            contracts=digest(Path(__file__).parents[1] / "agent_workflow/contracts.py"))
    profiles = inside(repo, config["stage_profiles"]) if config.get("stage_profiles") else None
    if profiles:
        TemporalEvidenceTools(evidence, profiles)
        binding["stage_profiles_sha256"] = digest(profiles)
    planned, files_to_write = [], []
    for condition in config["conditions"]:
        folder = out / "prompts" / condition["id"]
        catalogue = _catalogue(repo, config, condition)
        shared = {"objective": read_json(evidence / "packet.json")["objective"],
                  "learning_support": artifacts["static_reference"]["window"],
                  "observed_score_times": artifacts["static_reference"]["times"],
                  "route": condition["route"], "registered_choices": _prompt_choices(config, catalogue),
                  "fixed_parameters": {k: artifacts["base_program"][k] for k in
                                       ["mixture_temperature", "teacher_endpoint_temperature_A2", "teacher_neighbors",
                                        "teacher_score_beta", "native_rms_ratio", "time_ramp_power"]},
                  "delta_policy": "Only registered curvature conditions may vary delta_A; all temperatures are frozen",
                  "baseline_signature": program_signature(artifacts["base_program"]),
                  "incumbent": {k: artifacts["base_program"].get(k) for k in
                                ["reward_view", "derivative_path", "coordinate_representation", "pointcloud_delta_A", "dose_reference"]},
                  "temporal_profiles_available": profiles is not None,
                  "stage_profiles_available": profiles is not None,
                  "curves_available": bool(read_json(evidence / "continuous_functions.json")["functions"]),
                  "evidence_delivery": {"policy": protocol,
                      "Analyst": "Read an actual get_curve/get_stage_profile observation before finish. "
                                 "Each temporal hypothesis cites its relevant curve handle; summaries alone are insufficient.",
                      "Designer": "For edit, cite an actual Analyst hypothesis and its inherited curve/profile handle. "
                                  "If the handoff does not justify a change, retain/defer."},
                  "task_wordings": ["Compare temporal coordinate evidence across the complete observed selection window.",
                                    "Assess time-dependent coordinate contrasts on the full supplied support."],
                  "hypotheses_are_not_observed_guidance_effects": True,
                  "no_current_outcomes_in_role_input": True}
        if protocol == PROTOCOL:
            shared["evidence_delivery"].update(
                Analyst="Declare curve_features for every hypothesis; read get_curve/get_curves, then cite "
                        "a matching delivered handle for every declared feature. Summaries are not curves. "
                        "Keep unsupported observations in findings/limitations, not actionable hypotheses.",
                Designer="For each adopted hypothesis and each of its features, cite the actual inherited "
                         "curve/profile handle. Only registered choices; retaining M2 is allowed.")
        skills = {}
        for role in ["Analyst", "Designer"]:
            text, receipts = render_instructions(repo, config, condition, role)
            files_to_write.append((folder / (role + ".instructions.md"), text))
            skills[role] = receipts
        files_to_write.extend([(folder / "task.json", shared), (folder / "catalogue.json", catalogue)])
        binding["conditions"][condition["id"]] = {"instruction_sources": skills,
                                                     "task_sha256": canonical_hash(shared)}
        planned.append({"id": condition["id"], "route": condition["route"], "registered_recipes": list(catalogue),
                        "chains": config["chains"], "phase": condition["phase"], "status": "planned_not_called"})
    target = out / "binding.json"
    if target.exists() and read_json(target) != binding:
        raise ValueError("Changed versioned plan inputs; use a new output folder")
    for path, contents in files_to_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, str):
            path.write_text(contents, encoding="utf-8")
        else:
            write_json(path, contents)
    write_json(target, binding)
    write_json(out / "plan_manifest.json", {"conditions": planned, "preparation_API_calls": 0, "preparation_FLOWR_runs": 0,
                                           "preparation_remote_tasks": 0, "purpose": "preparation catalogue, not cumulative runtime status"})
    return binding


def require_execution(config, execute):
    if not execute or config.get("execution_enabled") is not True:
        raise ValueError("Planning configuration: set execution_enabled=true in a new run config AND provide --execute")


def run_roles(repo, config, condition_id, *, execute=False):
    require_execution(config, execute)
    if condition_id not in config.get("enabled_conditions", []):
        raise ValueError("Condition not activated after the preceding phase's review")
    repo = Path(repo).resolve()
    prepare(repo, config)
    condition = next(c for c in config["conditions"] if c["id"] == condition_id)
    choices = _prompt_choices(config, _catalogue(repo, config, condition))
    out = inside(repo, config["output"])
    task = read_json(out / "prompts" / condition_id / "task.json")
    profile = inside(repo, config["stage_profiles"]) if config.get("stage_profiles") else None
    rows = []
    protocol = config.get("evidence_protocol", FINISH_POLICY)
    for chain in range(config["chains"]):
        folder = out / "calls" / condition_id / f"chain_{chain:02d}"
        handoff, references, analyst_curves, hypothesis_sources = None, None, [], {}
        for role in ["Analyst", "Designer"]:
            tools = TemporalEvidenceTools(inside(repo, config["evidence"]), profile, bulk_curves=protocol == PROTOCOL)
            if references is not None:
                tools.import_references(references)
            text, receipt = render_instructions(repo, config, condition, role)
            role_task = {**task, "role": role, "task_wording": task["task_wordings"][int(chain >= config["chains"] // 2)]}
            if role == "Designer":
                role_task["hypothesis_temporal_sources"] = hypothesis_sources
            protocol_kwargs = {}
            if protocol == PROTOCOL:
                delivery = TemporalEvidenceProtocol(role, hypothesis_sources)
                validator = delivery.validate
                protocol_kwargs = dict(schema_enricher=delivery.enrich_schema,
                    evidence_availability=delivery.availability, interface_policy=INTERFACE, max_format_repairs=2)
            else:
                validator = analyst_finish if role == "Analyst" else \
                            lambda result, tools: designer_finish(result, tools, hypothesis_sources)
            response = run_agent(role, tools, text, role_task, folder / role,
                                 choices=choices if role == "Designer" else None, handoff=handoff,
                                 max_steps=config["max_steps"], max_output_tokens=config["max_output_tokens"],
                                 finish_validator=validator, finish_policy=protocol, **protocol_kwargs)
            if read_json(folder / role / "completed.json")["backend"] != "real_api":
                raise ValueError("Temporal study requires real API receipts")
            write_json(folder / (role + ".json"), response)
            write_json(folder / (role + ".skills.json"), receipt)
            if role == "Analyst":
                handoff, references = response, tools.provenance()
                analyst_curves = sorted(temporal_sources(tools, tools.known))
                wire = read_json(folder / role / "wire_result.json")
                hypothesis_sources = {h["id"]: sorted(temporal_sources(tools, w["evidence_refs"]))
                                      for h, w in zip(handoff["hypotheses"], wire["hypotheses"], strict=True)}
        edit = response.get("edit")
        if response["decision"] == "edit":
            name = edit["value"]
            if edit["parameter"] != "temporal_recipe" or name not in choices:
                raise ValueError("Designer edit is outside the condition catalogue")
            ids = response["parameter_origins"]["hypothesis_ids"]
            hypotheses = [h for h in handoff.get("hypotheses", []) if h["id"] in ids]
            if not hypotheses or not response["evidence_ids"] or not analyst_curves:
                raise ValueError("Temporal edit needs an actual Analyst hypothesis and observed temporal evidence")
            cited = set(response["evidence_ids"]) | {i for h in hypotheses for i in h["evidence_ids"]}
            if "curve_support" not in cited and not any(i.startswith("temporal_profile/") for i in cited):
                raise ValueError("Temporal edit does not cite the curves/profile actually read")
            if validator(read_json(folder / "Designer/wire_result.json"), tools):
                raise ValueError("Compiled temporal edit lost its feature-specific hypothesis curve")
            program, reference = compile_recipe(repo, config, name)
        else:
            name = "retain_M2"
            program = sources(repo, config)["base_program"]
            reference = config["static_reference"]
        signature = program_signature(program)
        program_path = folder / "reward_program.json"
        write_json(program_path, program)
        write_json(folder / "function_audit.json", {
            "analyst_temporal_sources_read": analyst_curves, "analyst_hypotheses": len(handoff.get("hypotheses", [])),
            "hypothesis_temporal_sources": hypothesis_sources,
            "designer_temporal_sources_cited": sorted(temporal_sources(tools,
                read_json(folder / "Designer/wire_result.json")["evidence_refs"])),
            "finish_policy": protocol,
            "effective_recipe": name, "numerical_signature": signature,
            "output_changed_execution": signature != task["baseline_signature"],
            "gate": "Source delivery and compiler behavior only; scientific adequacy needs review and generated trace/quality checks",
            "unchanged_notice": "本次 Agent 输出未改变执行函数" if signature == task["baseline_signature"] else None})
        rows.append({"chain": chain, "decision": response["decision"], "recipe": name,
                     "program": program_path.relative_to(repo).as_posix(), "reference": reference, "signature": signature})
    counts = Counter(r["signature"] for r in rows)
    tied = sorted(s for s, n in counts.items() if n == max(counts.values()))
    winner = task["baseline_signature"] if task["baseline_signature"] in tied else tied[0]
    selected = next(r for r in rows if r["signature"] == winner)
    result = {"rows": rows, "selected": selected, "counts": dict(counts),
              "selection": "modal numerical program; M2 tie-break; no outcome selection",
              "reuse_M2": winner == task["baseline_signature"]}
    write_json(out / "selections" / (condition_id + ".json"), result)
    return result


def register_profile_knots(repo, config, profile_path, feature, prototype, name):
    """Make a NEW config using stable observed knots, preserving chosen values.

    The preset parameter levels remain interventions, not inferred optimal doses.
    No role or generation is launched. Caller explicitly selects the feature.
    """
    path = inside(repo, profile_path)
    profile = read_json(path)
    TemporalEvidenceTools(inside(repo, config["evidence"]), path)
    function = profile["functions"][feature]
    if not function["accepted_for_stage_hypothesis"]:
        raise ValueError("No reproducible knot for this feature; keep a smooth policy")
    a, b = profile["score_times"][0], profile["score_times"][-1]
    knots = [0., *[(t - a) / (b - a) for t in function["knot_times"]], 1.]
    recipe = copy.deepcopy(config["recipes"][prototype])
    original = np.asarray(recipe["knots"], float)
    for key in ["dose", "blend", "delta"]:
        recipe[key] = np.interp(np.linspace(0, 1, len(knots)), original, recipe[key]).tolist()
    recipe["knots"] = knots
    recipe["description"] += "; exploratory observed knots for " + feature
    recipe["profile_source"] = {"path": profile_path, "sha256": digest(path), "feature": feature}
    result = copy.deepcopy(config)
    if name in result["recipes"]:
        raise ValueError("New recipe name required")
    result["recipes"][name] = recipe
    result["stage_profiles"] = profile_path
    result["execution_enabled"], result["enabled_conditions"] = False, []
    return result
