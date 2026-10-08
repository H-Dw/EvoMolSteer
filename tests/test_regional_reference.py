import copy

import numpy as np
import pandas as pd
import pytest

from evomolsteer.continuous.regional_reference import (
    batch_direction_support, evaluate_xyz, joint_direction, learn_rules,
    synthesize_reference, validate_live_receipt,
)
from evomolsteer.io import digest, write_json


def fit(coefficients=(1.,), start=.2, end=.5):
    return {"degree": len(coefficients)-1, "coefficients": list(coefficients),
            "observed_time_start": start, "observed_time_end": end}


def xyz(x=1., y=0., z=0., start=.2):
    return {axis: fit((value,), start=start) for axis, value in zip("xyz", (x, y, z))}


def rules():
    return {"window": [0., .5], "region_width_A": 1., "absolute_covariance_floor": 1e-12,
            "selected_regions": [{"region": 0, "landmark_A": [0., 0., 0.], "functions_xyz": xyz()}]}


def reference():
    return {"window": [0., .5], "times": [.1, .2], "schema_version": "affinity-endpoint-library-1.0",
            "branch_mutation": {"original_fact": "retain"}, "original_priors": [.3, .7], "frames": [
                {"time": t, "teacher_endpoint_A": [[[0., 0., 0.], [2., 0., 0.]]],
                 "teacher_scores": [8.], "teacher_batches": [0], "high_scaled": [1.],
                 "teacher_contrast_direction_unit": [[[0., 1., 0.], [0., -1., 0.]]],
                 "teacher_contrast_confidence": [.02], "teacher_contrast_atom_weight": [[.8, 1.2]],
                 "teacher_contrast_provenance": [{"raw_direction_RMS_A": .09, "lower_immediate_parent_slots": [2], "local_confidence": .02}]} for t in [.1, .2]]}


def statistics(depths=(2, 3)):
    effects, batches, fitted = [], [], {"depths": {}}
    for depth in depths:
        fitted["depths"][str(depth)] = {"functions": {}}
        for axis, value in zip("xyz", (1., .4, -.2)):
            feature = f"region_00_internal_displacement_{axis}"
            effects.append({"depth": depth, "metric": "adjusted_covariance", "feature": feature,
                            "whole_window_mean": value, "CI_low": value-.05, "CI_high": value+.05,
                            "q": .01 if axis == "x" else .2, "independent_batches_available": 14})
            fitted["depths"][str(depth)]["functions"]["adjusted_covariance/"+feature] = fit((value,))
            for batch in range(14):
                batches.append({"batch": batch, "depth": depth, "metric": "adjusted_covariance",
                                "feature": feature, "whole_window_value": value*(1+batch/100)})
    return pd.DataFrame(effects), pd.DataFrame(batches), fitted, {"landmarks_A": [[0., 0., 0.]], "region_width_A": 1.}


def test_legendre_complete_xyz_and_missing_node_boundary():
    functions = xyz()
    functions["y"] = fit((2., 3.))
    assert evaluate_xyz(functions, .19) is None
    assert evaluate_xyz(functions, .51) is None
    np.testing.assert_allclose(evaluate_xyz(functions, .2), [1., -1., 0.])
    np.testing.assert_allclose(evaluate_xyz(functions, .5), [1., 5., 0.])
    with pytest.raises(ValueError, match="three XYZ"):
        evaluate_xyz({"x": fit()}, .3)
    functions["x"]["coefficients"] = [1., 2.]
    with pytest.raises(ValueError, match="shape"):
        evaluate_xyz(functions, .3)


def test_joint_field_translation_null_and_unit_atom_rms():
    direction, diagnostic = joint_direction([[0., 0., 0.], [2., 0., 0.], [3., 1., 0.]], .3, rules())
    np.testing.assert_allclose(direction.mean(0), np.zeros(3), atol=1e-15)
    assert np.sqrt(np.mean(np.sum(direction**2, axis=1))) == pytest.approx(1.)
    assert diagnostic["active_region_ids"] == [0]
    null, _ = joint_direction([[1., 0., 0.], [1., 0., 0.]], .3, rules())
    assert not null.any()
    boundary, _ = joint_direction([[0., 0., 0.], [2., 0., 0.]], .1, rules())
    assert not boundary.any()


def test_single_depth_rule_preserves_joint_xyz_and_batch_units():
    args = statistics()
    learned = learn_rules(*args, [0., .5])
    selected = learned["selected_regions"]
    assert len(selected) == 1 and selected[0]["depth"] == 2
    assert selected[0]["independent_batches"] == 14
    assert len(selected[0]["loo_cosines"]) == 14
    assert selected[0]["loo_positive_fraction"] == 1.
    assert selected[0]["functions_xyz"]["y"]["coefficients"] == [.4]
    assert selected[0]["functions_xyz"]["z"]["coefficients"] == [-.2]
    assert not learned["selection_uses_final_generated_labels"]


def test_missing_fits_and_covariance_triplet_are_rejected():
    effects, batches, fitted, catalog = statistics((2,))
    fitted["depths"]["2"]["functions"].pop("adjusted_covariance/region_00_internal_displacement_z")
    with pytest.raises(ValueError, match="complete XYZ"):
        learn_rules(effects, batches, fitted, catalog, [0., .5])
    with pytest.raises(ValueError, match="triplet"):
        learn_rules(effects[effects.feature.str.endswith(('x', 'y'))], batches, {}, catalog, [0., .5])
    with pytest.raises(ValueError, match="XYZ"):
        batch_direction_support(np.ones((14, 4)))


def test_directional_support_uses_batches_and_excludes_zero():
    values = np.array([[1., 0., 0.]]*13+[[0., 0., 0.]])
    support = batch_direction_support(values)
    assert support["independent_batches"] == 14
    assert support["loo_observed_batches"] == 13
    assert support["loo_cosines"][-1] is None
    assert support["loo_positive_fraction"] == 1.


def test_teacher_immutability_identity_scope_and_zero_field():
    original = reference()
    snapshot = copy.deepcopy(original)
    transformed, coverage = synthesize_reference(original, rules())
    assert original == snapshot
    for before, after in zip(original["frames"], transformed["frames"]):
        for key, value in before.items():
            if key not in {"teacher_contrast_direction_unit", "teacher_contrast_confidence"}:
                assert after[key] == value
        assert after["teacher_original_branch_direction_unit"] == before["teacher_contrast_direction_unit"]
        assert after["teacher_original_branch_confidence"] == before["teacher_contrast_confidence"]
    assert transformed["branch_mutation"] == original["branch_mutation"]
    assert transformed["original_priors"] == original["original_priors"]
    assert transformed["frames"][0]["teacher_contrast_confidence"] == [0.]
    assert transformed["frames"][1]["teacher_contrast_confidence"] == [.02]
    assert coverage["new_active_teachers"] == 1
    other = rules(); other["window"] = [0., .6]
    with pytest.raises(ValueError, match="windows"):
        synthesize_reference(original, other)


def test_no_original_branch_eligibility_is_manufactured():
    original = reference()
    original["frames"][1]["teacher_contrast_provenance"][0]["lower_immediate_parent_slots"] = []
    transformed, coverage = synthesize_reference(original, rules())
    assert not np.asarray(transformed["frames"][1]["teacher_contrast_direction_unit"]).any()
    assert transformed["frames"][1]["teacher_contrast_confidence"] == [0.]
    assert coverage["new_active_teachers"] == 0


def test_live_receipt_hash_and_actual_script_validation(tmp_path):
    script, inp, out = [tmp_path/name for name in ('script.py', 'input.json', 'output.json')]
    for path in (script, inp, out): path.write_text('original')
    receipt = {"tool_id": "sample", "command": ["python", str(script)], "returncode": 0,
               "inputs_and_code_unchanged": True, "complete_outputs": True, "error": None,
               "code_files": {str(script): digest(script)}, "input_files": {str(inp): digest(inp)},
               "output_files": {str(out): digest(out)}}
    path = tmp_path/'receipt.json'; write_json(path, receipt)
    _, bound = validate_live_receipt(path, "sample")
    assert len(bound) == 4
    out.write_text('changed')
    with pytest.raises(ValueError, match="changed"):
        validate_live_receipt(path, "sample")
