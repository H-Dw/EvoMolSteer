import copy

import pytest

from evomolsteer.lineage_retrieval.summaries import summarize_scientist_interval


FEATURES = (
    "radius_gyration_A", "centroid_pocket_centroid_A", "nearest_pocket_A",
    "contact_pairs_4A", "pocket_contacts_4A", "planarity_fraction", "asphericity",
)


def make_source():
    donors = []
    for donor_index in range(30):
        observations = []
        for step in range(51):
            row = {"step": step}
            for rep in ("current", "forecast"):
                for feature_index, feature in enumerate(FEATURES):
                    row[f"{rep}_{feature}"] = float(donor_index + step + feature_index)
                row[f"{rep}_centroid_x_A"] = float(step)
                row[f"{rep}_centroid_y_A"] = float(donor_index)
                row[f"{rep}_centroid_z_A"] = float(-step)
                row[f"{rep}_residue_contact_footprint"] = [f"R{step}"]
            observations.append(row)
        donors.append({"donor_id": f"T{donor_index:02d}", "observations": observations})
    background = []
    for step in range(51):
        features = {}
        for rep in ("current", "forecast"):
            for feature_index, feature in enumerate(FEATURES):
                center = float(step + feature_index + 20)
                features[f"{rep}_{feature}"] = [center - 5, center, center + 5]
        background.append({"step": step, "quantiles_P25_P50_P75": features})
    return {"teacher": {"donors": donors}, "background": background}


def test_window_summary_reads_only_the_six_requested_states():
    source = make_source()
    baseline = summarize_scientist_interval(source, 2)
    changed = copy.deepcopy(source)
    for donor in changed["teacher"]["donors"]:
        for observation in donor["observations"][11:]:
            for key in list(observation):
                if key != "step":
                    observation[key] = 1e99
    for row in changed["background"][11:]:
        for quantiles in row["quantiles_P25_P50_P75"].values():
            quantiles[:] = [1e99, 1e99, 1e99]
    assert summarize_scientist_interval(changed, 2) == baseline
    assert baseline["start_step"] == 5 and baseline["mid_step"] == 7 and baseline["end_step"] == 10
    assert len(baseline["donors"]) == 30


def test_initial_collapse_then_rebound_is_visible_in_adjacent_directions():
    source = make_source()
    values = [10, 8, 6, 7, 9, 12]
    for observation, value in zip(source["teacher"]["donors"][0]["observations"][:6], values):
        observation["current_radius_gyration_A"] = value
    card = summarize_scientist_interval(source, 1)["donors"][0]
    result = card["features"]["current_radius_gyration_A"]
    assert result["start"] == 10
    assert result["mid"] == 6
    assert result["end"] == 12
    assert result["net_delta"] == 2
    assert result["net_direction"] == "up"
    assert result["adjacent_directions"] == ["down", "down", "up", "up", "up"]
    assert result["sign_reversal"] is True
    assert result["measured_pattern_1e6_tolerance"] == "mixed"


def test_summary_and_measured_pattern_ignore_sub_tolerance_jitter():
    source = make_source()
    values = [1, 1 + 1e-8, 1 + 2e-8, 1 + 1e-8, 1, 1 + 1e-8]
    for row, value in zip(source['teacher']['donors'][0]['observations'][:6], values):
        row['current_nearest_pocket_A'] = value
    card = summarize_scientist_interval(source, 1)['donors'][0]['features']['current_nearest_pocket_A']
    assert card['measured_pattern_1e6_tolerance'] == 'flat'
    assert card['sign_reversal'] is False


def test_teacher_comparison_uses_background_bands_and_full_precision():
    source = make_source()
    for donor_index, donor in enumerate(source["teacher"]["donors"]):
        donor["observations"][5]["current_radius_gyration_A"] = 10.0 + donor_index
    row = next(r for r in summarize_scientist_interval(source, 1)["cohort_background"]
               if r["step"] == 5 and r["feature"] == "current_radius_gyration_A")
    assert row["teacher_median"] == 24.5
    assert row["background_P25"] == 20.0
    assert row["background_median"] == 25.0
    assert row["background_P75"] == 30.0
    assert row["teacher_count_below_background_P25"] == 10
    assert row["teacher_count_inside_background_P25_P75"] == 11
    assert row["teacher_count_above_background_P75"] == 9
    assert row["teacher_count_nonnull"] == 30
    card = summarize_scientist_interval(source, 1)["donors"][0]
    assert card["features"]["current_radius_gyration_A"]["end_vs_background"] == "below"
    assert card["centroid_xyz_delta_A"]["current"] == [5.0, 0.0, -5.0]
    assert card["residue_contact_footprints"]["current"] == {
        "start": ["R0"], "mid": ["R2"], "end": ["R5"]
    }


def test_nulls_and_interval_bounds_are_handled():
    source = make_source()
    source["teacher"]["donors"][0]["observations"][50]["forecast_asphericity"] = float("nan")
    result = summarize_scientist_interval(source, 10)["donors"][0]
    feature = result["features"]["forecast_asphericity"]
    assert feature["end"] is None
    assert feature["net_delta"] is None
    assert feature["net_direction"] is None
    assert feature["end_vs_background"] is None
    with pytest.raises(ValueError, match="1 through 10"):
        summarize_scientist_interval(source, 0)

