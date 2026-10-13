import copy
import math
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evomolsteer.lineage_retrieval.validation import (
    validate_background_evidence,
    validate_hypothesis_specificity,
)


def make_source():
    donors = []
    for donor_index in range(30):
        observations = []
        for step in range(51):
            observations.append({
                "step": step,
                "current_radius_gyration_A": 1.0 + donor_index / 10 + step / 100,
                "forecast_radius_gyration_A": 2.0 + donor_index / 20 + step / 50,
                "delta_current_Rg": (donor_index - 15) / 100,
                "online_pic50": 8.0 + donor_index / 100,
                "final_score": 999.0,
            })
        donors.append({"donor_id": f"T{donor_index + 1:02d}", "observations": observations})
    background = []
    for step in range(51):
        background.append({"step": step, "quantiles_P25_P50_P75": {
            "current_radius_gyration_A": [1.0, 1.5 + step / 100, 2.0],
            "forecast_radius_gyration_A": [2.0, 3.0 + step / 100, 4.0],
            "delta_current_Rg": [-0.2, 0.0, 0.2],
        }})
    return {"teacher": {"donors": donors}, "background": background}


def median_at(source, step, feature):
    return statistics.median(d["observations"][step][feature] for d in source["teacher"]["donors"])


def make_hypothesis(source, interval=2):
    low, high = 5 * (interval - 1), 5 * interval
    records = []
    for step, feature in ((low, "current_radius_gyration_A"), (high, "forecast_radius_gyration_A")):
        records.append({
            "step": step,
            "feature": feature,
            "teacher_median": median_at(source, step, feature),
            "background_median": (source["background"][step]["quantiles_P25_P50_P75"].get(feature) or [None, None])[1],
        })
    return {"interval": interval, "start_step": low, "end_step": high,
            "background_evidence": records}


def test_valid_background_evidence_checks_thirty_donor_and_background_medians():
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"][0]["teacher_median"] += 1e-7
    hypothesis["background_evidence"][1]["background_median"] += 1e-6
    assert validate_background_evidence(hypothesis, source) is True


@pytest.mark.parametrize("field,value", [
    ("teacher_median", True),
    ("background_median", False),
    ("teacher_median", math.nan),
    ("background_median", math.inf),
])
def test_background_evidence_rejects_boolean_or_nonfinite_supplied_medians(field, value):
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"][0][field] = value
    with pytest.raises(ValueError, match="allowed-time source-backed feature medians"):
        validate_background_evidence(hypothesis, source)


def test_background_evidence_rejects_wrong_medians_and_returns_exact_repair_values():
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"][0]["teacher_median"] = 12345.0
    with pytest.raises(ValueError) as exc:
        validate_background_evidence(hypothesis, source)
    text = str(exc.value)
    assert '"step":5' in text
    assert '"feature":"current_radius_gyration_A"' in text
    assert f'"teacher_median":{median_at(source, 5, "current_radius_gyration_A")}' in text
    assert '"step":15' not in text
    assert "final_score" not in text


def test_background_evidence_requires_two_distinct_records_at_both_window_ends():
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"] = [copy.deepcopy(hypothesis["background_evidence"][0])] * 2
    with pytest.raises(ValueError, match="distinct feature/step|interval end step"):
        validate_background_evidence(hypothesis, source)


def test_background_evidence_rejects_future_step_and_shows_only_window_options():
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"][1]["step"] = 15
    with pytest.raises(ValueError) as exc:
        validate_background_evidence(hypothesis, source)
    text = str(exc.value)
    assert "within 5..10" in text
    assert '"step":15' not in text
    assert '"step":10' in text
    assert "final_score" not in text


def test_wrong_clock_reports_source_backed_options_only_inside_expected_window():
    source = make_source()
    hypothesis = make_hypothesis(source)
    hypothesis["end_step"] = 15
    with pytest.raises(ValueError) as exc:
        validate_background_evidence(hypothesis, source)
    text = str(exc.value)
    assert "clock must be exactly 5..10" in text
    assert '"step":5' in text and '"step":10' in text
    assert '"step":15' not in text


def test_background_median_null_is_allowed_only_when_source_has_no_p50():
    source = make_source()
    source["background"][5]["quantiles_P25_P50_P75"].pop("current_radius_gyration_A")
    hypothesis = make_hypothesis(source)
    hypothesis["background_evidence"][0]["background_median"] = None
    assert validate_background_evidence(hypothesis, source) is True
    hypothesis["background_evidence"][0]["background_median"] = 1.5
    with pytest.raises(ValueError, match="must be null"):
        validate_background_evidence(hypothesis, source)


def test_hypothesis_specificity_rejects_repeated_public_bundle_but_allows_real_window_change():
    bundle = {
        "interval": 1, "start_step": 0, "end_step": 5,
        "hypothesis": "The current cloud contracts while the forecast remains broader.",
        "support_conditions": ["Current radius falls."],
        "counter_conditions": ["Current radius expands."],
        "observable_cues": ["Compare shape at the window endpoints."],
        "minority_modes": "Some paths remain broader.",
    }
    repeated = [bundle, {**bundle, "interval": 2, "start_step": 5, "end_step": 10}]
    with pytest.raises(ValueError, match="identical public hypothesis bundles.*do not invent distinctions"):
        validate_hypothesis_specificity(repeated)
    with pytest.raises(ValueError, match='identical public hypothesis bundles'):
        validate_hypothesis_specificity([{**h, 'background_evidence': [{'teacher_median': i}]} for i, h in enumerate(repeated)])
    changed = [bundle, {**repeated[1], "observable_cues": ["Contact rises from the start to the end of this window."]}]
    assert validate_hypothesis_specificity(changed) is True


def test_specificity_rejects_clock_only_template_variations():
    def window(interval, low, high):
        return {'interval': interval, 'start_step': low, 'end_step': high,
                'hypothesis': f'第{interval}窗从{low}步到{high}步逐渐改变尺度，实际方向以数据为准。',
                'support_conditions': ['比较实际状态。'], 'counter_conditions': ['不得推断能量。'],
                'observable_cues': [f'查看{low}步及{high}步的接触。'], 'minority_modes': '存在局部差异。'}
    with pytest.raises(ValueError, match='changing only numeric clock'):
        validate_hypothesis_specificity([window(4, 15, 20), window(5, 20, 25)])
    changed = [window(4, 15, 20), window(5, 20, 25)]
    changed[0]['hypothesis'] = '接触由宽散区域集中到VAL53附近。'
    changed[1]['hypothesis'] = '接触由宽散区域集中到VAL66附近。'
    assert validate_hypothesis_specificity(changed) is True
