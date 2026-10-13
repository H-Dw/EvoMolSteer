import pytest

from evomolsteer.lineage_retrieval.donor_validation import validate_donor_descriptions


MEASURED_FEATURES = (
    "current_radius_gyration_A", "current_contact_pairs_4A", "current_nearest_pocket_A",
    "forecast_radius_gyration_A", "forecast_contact_pairs_4A", "forecast_nearest_pocket_A",
)


def make_source():
    donors = []
    for di in range(30):
        observations = []
        for step in range(51):
            row = {"step": step, "online_pic50": 5.0 + step / 100}
            for fi, key in enumerate(MEASURED_FEATURES):
                row[key] = float(di * 10 + step + fi)
            for rep in ("current", "forecast"):
                row[f"{rep}_centroid_x_A"] = float(step)
                row[f"{rep}_centroid_y_A"] = float(di)
                row[f"{rep}_centroid_z_A"] = float(-step)
                row[f"{rep}_residue_contact_footprint"] = [f"R{step}"]
            observations.append(row)
        donors.append({"donor_id": f"T{di:02d}", "observations": observations})
    return {"teacher": {"donors": donors}}


def make_description(donor, interval=2, text=None):
    lo, hi = 5 * (interval - 1), 5 * interval
    evidence = []
    for feature in MEASURED_FEATURES:
        rep = "current" if feature.startswith("current_") else "forecast"
        evidence.extend([
            {"feature": feature, "step": lo, "claim": "start", "observed_value": donor["observations"][lo][feature]},
            {"feature": feature, "step": hi, "claim": "end", "observed_value": donor["observations"][hi][feature]},
        ])
    mid = lo + 2
    changes = []
    for feature in MEASURED_FEATURES:
        vals = [donor["observations"][s][feature] for s in range(lo, hi + 1)]
        pattern = "increase" if all(b > a for a, b in zip(vals, vals[1:])) else "mixed"
        changes.append({"feature": feature, "start_value": vals[0], "mid_value": vals[2],
                        "end_value": vals[-1], "pattern": pattern})
    return {"donor_id": donor["donor_id"], "interval": interval,
            "text": text or f"Observed trajectory group {int(donor['donor_id'][1:])} at step {lo}.",
            "evidence": evidence, "measured_changes": changes}


def valid_descriptions(source, interval=2):
    return [make_description(donor, interval) for donor in source["teacher"]["donors"]]


def test_rejects_reused_paragraph_when_qualitative_window_evolution_differs():
    source = make_source()
    source["teacher"]["donors"][1]["observations"][7]["forecast_residue_contact_footprint"] = ["OTHER"]
    descriptions = valid_descriptions(source)
    descriptions[1]["text"] = descriptions[0]["text"]
    with pytest.raises(ValueError, match="identical paragraph reused.*qualitative evolution differs"):
        validate_donor_descriptions(descriptions, source, 2)


def test_allows_same_text_for_different_numeric_offsets_with_same_trends_and_footprints():
    source = make_source()
    descriptions = valid_descriptions(source)
    descriptions[0]["text"] = "For donor T00 at step 5: cloned observation."
    descriptions[1]["text"] = "For donor T01 at step 10: cloned observation."
    result = validate_donor_descriptions(descriptions, source, 2)
    assert result["donors_validated"] == 30
    assert result["identical_paragraph_pairs_same_qualitative_evolution"] == 1
    assert result["semantic_claims_interpreted"] is False


def test_evidence_citations_are_literal_and_window_bounded():
    source = make_source()
    descriptions = valid_descriptions(source)
    descriptions[0]["evidence"][0]["feature"] = "current_nonexistent_A"
    with pytest.raises(ValueError, match="absent from source observation"):
        validate_donor_descriptions(descriptions, source, 2)

    descriptions = valid_descriptions(source)
    descriptions[0]["evidence"][0]["step"] = 11
    with pytest.raises(ValueError, match="outside 5..10"):
        validate_donor_descriptions(descriptions, source, 2)


def test_requires_measured_changes_with_source_values_and_six_state_pattern():
    source = make_source()
    descriptions = valid_descriptions(source)
    first = source["teacher"]["donors"][0]
    key = "current_radius_gyration_A"
    vals = [first["observations"][s][key] for s in range(5, 11)]
    first["observations"][7][key] = vals[2] + 10
    descriptions = valid_descriptions(source)
    record = next(x for x in descriptions[0]["measured_changes"] if x["feature"] == key)
    record["mid_value"] = first["observations"][7][key]
    record["pattern"] = "mixed"
    validate_donor_descriptions(descriptions, source, 2)

    record["pattern"] = "increase"
    with pytest.raises(ValueError, match="pattern must be 'mixed'"):
        validate_donor_descriptions(descriptions, source, 2)


def test_measured_change_anchors_must_match_source_and_include_both_structures_and_pocket():
    source = make_source()
    descriptions = valid_descriptions(source)
    descriptions[0]["measured_changes"][0]["end_value"] += 0.1
    with pytest.raises(ValueError, match="end_value disagrees with source"):
        validate_donor_descriptions(descriptions, source, 2)

    descriptions = valid_descriptions(source)
    descriptions[0]["measured_changes"] = [x for x in descriptions[0]["measured_changes"]
                                            if "forecast_" not in x["feature"]]
    with pytest.raises(ValueError, match="at least 4 scalar features"):
        validate_donor_descriptions(descriptions, source, 2)

