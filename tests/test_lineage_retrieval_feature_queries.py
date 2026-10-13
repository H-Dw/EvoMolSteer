import copy

import pytest

from evomolsteer.lineage_retrieval.feature_queries import scientist_feature_summary


def make_source():
    donors = []
    for i in range(30):
        observations = []
        for step in range(51):
            observations.append({
                "step": step,
                "current_cov_xx_A2": i * 0.1 + step * 0.01,
                "forecast_atom_count": 20 + i + step,
                "delta_current_contact_pairs_4A": float((step % 3) - 1),
                "current_online_pic50": 6.0 + i / 10,
                "batch": 999,
                "slot": 888,
                "current_covariance_vector": [float(step), float(i)],
            })
        donors.append({"donor_id": f"T{i+1:02d}", "observations": observations})
    background = []
    for step in range(51):
        background.append({"step": step, "quantiles_P25_P50_P75": {
            "current_cov_xx_A2": [step * .01 - 1, step * .01 + .5, step * .01 + 2],
            "forecast_atom_count": [25.0 + step, 40.0 + step, 55.0 + step],
            "delta_current_contact_pairs_4A": [-1.0, 0.0, 1.0],
        }})
    return {"teacher": {"donors": donors}, "background": background,
            "oracle": {"terminal_node_id": "must-not-escape"}, "source_path": "private/raw/path"}


def test_queries_scalar_features_outside_default_evidence_cards():
    source = make_source()
    result = scientist_feature_summary(source, 2, ["current_cov_xx_A2", "forecast_atom_count"])
    assert (result["start_step"], result["mid_step"], result["end_step"]) == (5, 7, 10)
    assert [entry["feature"] for entry in result["features"]] == ["current_cov_xx_A2", "forecast_atom_count"]
    first = result["features"][0]["donors"][0]
    assert first["start"] == pytest.approx(0.05)
    assert first["mid"] == pytest.approx(0.07)
    assert first["end"] == pytest.approx(0.10)
    assert first["pattern"] == "increase"
    row = next(x for x in result["cohort_background"]
               if x["feature"] == "current_cov_xx_A2" and x["step"] == 5)
    assert row["teacher_median"] == pytest.approx(1.50)
    assert row["background_P25"] == pytest.approx(-0.95)
    assert row["background_median"] == pytest.approx(0.55)
    assert row["background_P75"] == pytest.approx(2.05)
    assert row["teacher_minus_background_P50"] == pytest.approx(0.95)
    assert row["teacher_vs_background_P50"] == "above"
    assert row["relative_to_background_exact_direction"] == "above"
    assert row["relative_to_background_near_equal"] is False
    assert row["relative_to_background_tolerance"] == 1e-6
    assert row["relative_to_background_is_statistical_significance"] is False
    assert "oracle" not in result and "source_path" not in result


def test_window_query_is_invariant_to_future_sentinels():
    source = make_source()
    baseline = scientist_feature_summary(source, 1, ["current_cov_xx_A2", "forecast_atom_count"])
    mutated = copy.deepcopy(source)
    for donor in mutated["teacher"]["donors"]:
        for row in donor["observations"][6:]:
            row["current_cov_xx_A2"] = ["future vector sentinel"]
            row["forecast_atom_count"] = float("inf")
    for row in mutated["background"][6:]:
        row["quantiles_P25_P50_P75"] = {"current_cov_xx_A2": [1e90] * 3,
                                       "forecast_atom_count": [1e90] * 3}
    assert scientist_feature_summary(mutated, 1, ["current_cov_xx_A2", "forecast_atom_count"]) == baseline


def test_background_missing_values_remain_null_and_missing_teacher_feature_rejects():
    source = make_source()
    source["background"][5]["quantiles_P25_P50_P75"].pop("current_cov_xx_A2")
    result = scientist_feature_summary(source, 1, ["current_cov_xx_A2"])
    row = next(x for x in result["cohort_background"] if x["step"] == 5)
    assert row["background_median"] is None
    assert row["background_P25"] is None
    assert row["teacher_count_below_background_P25"] is None
    assert row["teacher_minus_background_P50"] is None
    assert row["teacher_vs_background_P50"] is None
    assert row["relative_to_background_exact_direction"] is None
    assert row["relative_to_background_near_equal"] is None


def test_background_p50_equal_and_near_equal_use_numeric_tolerance():
    source = make_source()
    # The step-5 median of this feature is 1.5; a sub-tolerance offset remains numerically equal.
    source["background"][5]["quantiles_P25_P50_P75"]["current_cov_xx_A2"] = [0.0, 1.5 + 5e-7, 3.0]
    row = next(x for x in scientist_feature_summary(source, 1, ["current_cov_xx_A2"])["cohort_background"]
               if x["step"] == 5)
    assert row["teacher_minus_background_P50"] == pytest.approx(-5e-7)
    assert row["teacher_vs_background_P50"] == "equal"
    assert row["relative_to_background_exact_direction"] == "equal"
    assert row["relative_to_background_near_equal"] is True

    source["background"][5]["quantiles_P25_P50_P75"]["current_cov_xx_A2"] = [0.0, 1.5 + 2e-6, 3.0]
    row = next(x for x in scientist_feature_summary(source, 1, ["current_cov_xx_A2"])["cohort_background"]
               if x["step"] == 5)
    assert row["teacher_vs_background_P50"] == "below"
    assert row["relative_to_background_near_equal"] is False

    del source["teacher"]["donors"][0]["observations"][2]["current_cov_xx_A2"]
    with pytest.raises(ValueError, match="step 2: feature .* is missing"):
        scientist_feature_summary(source, 1, ["current_cov_xx_A2"])


@pytest.mark.parametrize("bad_value", [[1.0, 2.0], True, float("nan")])
def test_rejects_non_scalar_bool_or_nonfinite_values_with_source_location(bad_value):
    source = make_source()
    source["teacher"]["donors"][0]["observations"][7]["current_cov_xx_A2"] = bad_value
    with pytest.raises(ValueError, match="donor T01 step 7: feature .* must be finite scalar numeric"):
        scientist_feature_summary(source, 2, ["current_cov_xx_A2"])


def test_rejects_affinity_and_identifier_features():
    source = make_source()
    with pytest.raises(ValueError, match="not available to Scientist"):
        scientist_feature_summary(source, 1, ["current_online_pic50"])
    with pytest.raises(ValueError, match="scalar key"):
        scientist_feature_summary(source, 1, ["batch"])

