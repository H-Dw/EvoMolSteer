"""Two compact, evidence-driven roles; direction labels have no execution quota."""

import json
from pathlib import Path

import jsonschema

from . import linear_scale as scale
from .geometry_council import object_schema, text_schema, parse_output
from .geometry_protocol import choose
from .guidance_windows import windows
from .io import read, frozen, value_sha


def history_view(row):
    from .absolute_coordinate import METRICS
    from .coordinate_statistics import RANKING, SUMMARY

    names = [
        "id",
        scale.EFFECTIVE,
        scale.TICKS,
        "attempts",
        "built",
        "connected",
        "unique_smiles",
        RANKING,
        *METRICS,
        "maximum_unconstrained_single_atom_guidance_increment_angstrom",
        "maximum_single_atom_guidance_increment_angstrom",
        "stability_status",
        "assignment_diagnostics_not_used_for_ranking",
    ]
    result = {name: row[name] for name in names if name in row}
    stats = row.get(SUMMARY, {})
    result[SUMMARY] = {
        k: stats[k]
        for k in (
            "arithmetic_mean",
            "upper_tail_filtered_mean",
            "retained_count",
            "excluded_high_count",
            "threshold_angstrom",
            "ranking_eligible",
        )
        if k in stats
    }
    proposal = row.get("proposal")
    if proposal:
        result["original_selection"] = {
            k: proposal[k]
            for k in (
                "anchor_id",
                "updates",
                "exploration_tier",
                "improvement_reason",
                "uncertainty",
            )
            if k in proposal
        }
    return result


def advise(
    config, history, domain, folder, skills, failures=(), final=False, provider=None
):
    from .analysis_tools import EvidenceTools
    from agenticgen.providers.deepseek_tools import RoundBudget, call_with_tools

    folder, skills = Path(folder), Path(skills)
    folder.mkdir(parents=True, exist_ok=True)
    anchors = {r["id"]: r for r in history}
    if not anchors or len(anchors) != len(history):
        raise ValueError("Unique completed linear history required")
    learning = config.get("guidance_checkpoint_context")
    if learning and learning["rebase_from_best_checkpoint"]:
        anchors = {
            learning["best_checkpoint_id"]: anchors[learning["best_checkpoint_id"]]
        }
    condition_path = (
        Path(config["root"]) / "conditions" / (domain["teacher_id"] + ".json")
    )
    condition = read(condition_path) if condition_path.exists() else None
    registry = EvidenceTools(
        config["root"], history, domain, folder / "calculations", condition=condition
    )
    registry.precompute()
    settings = config["agent_tools"]
    budget = RoundBudget(
        folder / "model_budget",
        settings["max_model_calls_per_round"],
        settings["max_output_tokens_per_round"],
    )
    provider = provider or call_with_tools
    incumbent = choose(history)
    from .linear_runtime import range_allowance, formal_allowance

    common = dict(
        teacher_id=domain["teacher_id"],
        condition_sha256=domain["condition_sha256"],
        history=[history_view(row) for row in history],
        failures=list(failures),
        incumbent_id=incumbent["id"],
        no_future_execution=final,
        scale_reference={k: domain[k] for k in (scale.REFERENCE, scale.UNIT)},
        search_domain={
            k: domain["control_settings"][k]
            for k in ("search_minimum", "search_maximum")
        },
        confirmed_constant_levels=domain["confirmed_effective_levels"],
        profile_scoped_evidence=domain.get("confirmed_profiles", []),
        working_neighbourhoods=scale.boundary_status(domain, incumbent),
        execution_contract=dict(
            maximum_selected=domain["control_settings"]["maximum_selected"],
            maximum_window_tick_change=domain["control_settings"][
                "maximum_window_tick_change"
            ],
            maximum_total_tick_change=domain["control_settings"][
                "maximum_total_tick_change"
            ],
            windows=[
                dict(window_id=i, step_bounds=list(w))
                for i, w in enumerate(windows(domain))
            ],
        ),
        evidence_receipts=sorted(registry.receipt_ids),
        range_evidence_allowance=range_allowance(config, domain["teacher_id"]),
        remaining_formal_attempts=formal_allowance(config),
    )
    if learning:
        common["checkpoint_learning"] = learning
    common["allowed_anchor_ids"] = list(anchors)
    common["conditioning_protocol"] = dict(
        prefix_inpainting=config.get("prefix_inpainting", {"enabled": False}),
        atom_assignment_policy=config.get("atom_assignment_policy"),
    )
    if domain.get("exploration_policy"):
        from .exploration import describe

        common["exploration_hypotheses"] = describe(
            domain["exploration_policy"], domain[scale.UNIT]
        )
        common["search_objective"] = (
            "Find potentially better complete configurations through distinct jump intensities. Do not design ablations or prove inference mechanisms."
        )
        common["working_neighbourhood_is_a_preference_not_an_admission_limit"] = True
    text = text_schema
    decisions = ["hold"] if final else ["propose", "hold", "request_domain_evidence"]
    proposal_schema = object_schema(
        dict(
            decision=dict(enum=decisions),
            reason=text(scale.EXPLANATION_MAX_CHARACTERS),
            proposals=dict(
                type="array",
                maxItems=(
                    0 if final else domain["control_settings"]["maximum_proposals"]
                ),
                items=scale.action_schema(anchors, domain),
            ),
        )
    )
    if config["llm"].get("final_response_mode") == "strict_tool_v1":
        fields = proposal_schema["properties"]
        proposal_schema = object_schema({key: fields[key] for key in ("decision", "proposals", "reason")})

    def check_proposals(result):
        jsonschema.validate(result, proposal_schema)
        proposed = result["proposals"]
        if (result["decision"] == "hold") != (not proposed):
            raise jsonschema.ValidationError(
                "hold has no actions; other decisions require proposals"
            )
        seen = list(history)
        for action in proposed:
            allowed = set(anchors) | registry.receipt_ids
            if not action["evidence_ids"] or set(action["evidence_ids"]) - allowed:
                raise jsonschema.ValidationError(
                    "Use original measured IDs or observed calculation receipts"
                )
            try:
                resolved = scale.resolve_action(action, anchors, domain, seen, failures)
            except ValueError as error:
                raise jsonschema.ValidationError(str(error)) from error
            seen.append(resolved)
            if (
                result["decision"] == "request_domain_evidence"
                and resolved["task_type"] != "range_evidence"
            ):
                raise jsonschema.ValidationError(
                    "Range request contains an ordinary confirmed-level action"
                )
        if len({p["id"] for p in proposed}) != len(proposed):
            raise jsonschema.ValidationError("Repeated proposal identity")

    def invoke(role, context, schema, validator):
        context = dict(
            context, existing_calculation_receipts=registry.observed_calculations()
        )
        if config["llm"].get("final_response_mode") == "strict_tool_v1":
            from agenticgen.providers.strict_tools import FINAL_TOOL
            payload_field = "selected" if role == "synthesis" else "proposals"
            context["final_submission_contract"] = dict(
                tool=FINAL_TOOL, required_fields=schema["required"],
                payload_field=payload_field,
                explicit_empty_array_required=True, infer_selection_from_reason=False,
                decision_cases={"hold": payload_field + " must be []",
                    "propose_or_request_domain_evidence": payload_field + " must contain an explicitly chosen nonempty payload"},
                no_future_execution=final,
            )
            if role == "synthesis":
                context["final_submission_contract"]["candidate_ids"] = list(context["candidates"])
        modules = ["analysis.md", role + ".md", "computation.md"]
        messages = [
            dict(
                role="system",
                content="\n\n".join(
                    (skills / name).read_text(encoding="utf-8") for name in modules
                ),
            ),
            dict(
                role="user",
                content=json.dumps(
                    context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            ),
        ]
        frozen(
            folder / (role + ".prompt_source.json"),
            dict(messages=messages, response_schema=schema),
        )
        frozen(
            folder / (role + ".execution_claim.json"),
            dict(
                role=role,
                scale_policy=scale.POLICY,
                domain_sha256=domain["domain_sha256"],
                registry_sha256=value_sha(registry.binding),
            ),
        )
        return provider(
            messages,
            schema,
            folder / (role + ".json"),
            response_validator=validator,
            response_parser=parse_output,
            registry=registry,
            budget=budget,
            **config["llm"],
        )

    proposed = invoke("proposal", common, proposal_schema, check_proposals)
    candidates = {p["id"]: p for p in proposed["proposals"]}
    synthesis_schema = object_schema(
        dict(
            decision=dict(enum=decisions),
            reason=text(scale.EXPLANATION_MAX_CHARACTERS),
            selected=dict(
                type="array",
                uniqueItems=True,
                maxItems=0 if final else domain["control_settings"]["maximum_selected"],
                items=(
                    dict(type="string", enum=list(candidates))
                    if candidates
                    else dict(type="string")
                ),
            ),
        )
    )
    if config["llm"].get("final_response_mode") == "strict_tool_v1":
        fields = synthesis_schema["properties"]
        synthesis_schema = object_schema({key: fields[key] for key in ("decision", "selected", "reason")})

    def check_selection(result):
        jsonschema.validate(result, synthesis_schema)
        if (result["decision"] == "hold") != (not result["selected"]):
            raise jsonschema.ValidationError(
                "hold selects no actions; other decisions select existing proposals"
            )
        if any(identifier not in candidates for identifier in result["selected"]):
            raise jsonschema.ValidationError("Unknown selected proposal")
        if result["decision"] == "request_domain_evidence" and any(
            scale.resolve_action(candidates[i], anchors, domain, history, failures)[
                "task_type"
            ]
            != "range_evidence"
            for i in result["selected"]
        ):
            raise jsonschema.ValidationError(
                "Range evidence request must select unconfirmed frontier profiles"
            )

    selected = invoke(
        "synthesis",
        dict(common, proposal_decision=proposed, candidates=candidates),
        synthesis_schema,
        check_selection,
    )
    result = dict(selected, actions=[candidates[i] for i in selected["selected"]])
    frozen(folder / "DECISION.json", result)
    return result
