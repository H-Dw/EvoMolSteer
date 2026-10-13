"""Source delivery gates for temporal roles; no inferred or fabricated citations."""

FINISH_POLICY = "temporal-observed-curves-v2"


def temporal_sources(tools, refs):
    """Keep feature-specific identities, including pages of an observed curve."""
    result = set()
    for ref in refs:
        identity = tools.delivered[ref]
        while identity["kind"] == "field":
            identity = identity["base"]
        if identity["kind"] in {"curve", "stage_profile"}:
            result.add(identity["source_id"])
    return result


def analyst_finish(result, tools):
    errors = []
    if not temporal_sources(tools, tools.known):
        errors.append({"code": "observed_curve_not_read", "pointer": "/hypotheses",
                       "reason": "No get_curve/get_stage_profile observation was delivered. "
                       "Existing continuous curves remain available when stage profiles are absent.",
                       "next_action": "Choose a relevant feature from curve_feature_names and call get_curve."})
    for index, hypothesis in enumerate(result["hypotheses"]):
        if not temporal_sources(tools, hypothesis["evidence_refs"]):
            errors.append({"code": "hypothesis_missing_curve", "pointer": f"/hypotheses/{index}/evidence_refs",
                           "reason": "A temporal hypothesis must cite the observed curve/profile for its observable. "
                           "Metadata, curve_support audit rows and continuous_trends summaries are insufficient.",
                           "next_action": "Read the relevant curve and cite its handle, or remove the unsupported hypothesis."})
    return errors


def designer_finish(result, tools, hypothesis_sources):
    if result["decision"] != "edit":
        return []
    ids = result["hypothesis_refs"]
    supporting = set().union(*(set(hypothesis_sources.get(i, [])) for i in ids))
    cited = temporal_sources(tools, result["evidence_refs"])
    if not ids or not supporting or not cited.intersection(supporting):
        return [{"code": "edit_missing_hypothesis_curve", "pointer": "/evidence_refs",
                 "reason": "A temporal edit must cite an actual Analyst hypothesis and at least one of "
                 "that hypothesis's feature-specific curve/profile sources; reading an unrelated curve is insufficient.",
                 "next_action": "Use the supplied hypothesis_temporal_sources and inherited handles, or retain/defer."}]
    return []
