import numpy as np

from evomolsteer.lineage_retrieval.data import (
    _ancestor_at,
    _artifact_manifest,
    _bond_edge_count,
    _charged_atom_count,
    _feature_table,
    _pool_groups,
    _pool_view,
    _rank_top30,
    _sig,
    _visible_source,
    _validate_trajectory,
    _verify_frozen_study,
)


def test_exact_ancestor_uses_only_selections_through_cutoff():
    selected = np.array([
        [0, 1, 2, 3],
        [1, 1, 3, 0],
        [2, 0, 2, 1],
        [3, 2, 1, 0],
        [0, 3, 2, 1],
    ])
    final_slot = 2
    # At cutoff 2: final slot 2 -> step 3 slot 1 -> step 2 slot 0.
    assert _ancestor_at(selected, final_slot, 2) == 0
    changed_future = selected.copy()
    changed_future[4, 0] = 3
    assert _ancestor_at(changed_future, final_slot, 2) == 0


def test_final_readout_parent_uses_step_99_selection():
    selected = np.zeros((100, 4), dtype=int)
    selected[99, 2] = 1
    selected[98, 1] = 3
    assert _ancestor_at(selected, 2, 99) == 1
    assert _ancestor_at(selected, 2, 98) == 3


def test_trajectory_validation_checks_copies_roots_and_clocks():
    slots = np.arange(50)
    arrays = {
        "selected_indices": np.tile(slots, (100, 1)),
        "root_slot": np.tile(slots, (100, 1)),
        "current_coords": np.zeros((100, 50, 1, 3), dtype=np.float32),
        "predicted_coords": np.zeros((100, 50, 1, 3), dtype=np.float32),
        "proposal_coords": np.zeros((100, 50, 1, 3), dtype=np.float32),
        "current_atomics": np.zeros((100, 50, 1), dtype=np.uint8),
        "proposal_atomics": np.zeros((100, 50, 1), dtype=np.uint8),
        "current_bonds": np.zeros((100, 50, 1, 1), dtype=np.uint8),
        "proposal_bonds": np.zeros((100, 50, 1, 1), dtype=np.uint8),
        "current_charges": np.zeros((100, 50, 1), dtype=np.uint8),
        "proposal_charges": np.zeros((100, 50, 1), dtype=np.uint8),
        "score_time": np.repeat((np.arange(100) / 100.)[:, None], 3, axis=1),
        "step_size": np.full(100, 0.01),
        "state_time": np.arange(1, 101) / 100.,
        "offspring_count": np.ones((100, 50), dtype=int),
    }
    _validate_trajectory(arrays, 0, slots)
    arrays["current_coords"][1, 0, 0, 0] = 1.
    import pytest
    with pytest.raises(ValueError, match="coordinate copy mismatch"):
        _validate_trajectory(arrays, 0, slots)


def test_pool_groups_merge_only_identical_recursive_signatures():
    grouped = _pool_groups({0: "same-history", 1: "same-history", 2: "different-history"})
    assert grouped == {"same-history": [0, 1], "different-history": [2]}


def test_feature_table_preserves_fields_from_both_representations():
    table = _feature_table([
        {"representation": "current", "current_centroid_x_A": 1.0},
        {"representation": "forecast", "forecast_centroid_x_A": 2.0},
    ])
    assert "current_centroid_x_A" in table.column_names
    assert "forecast_centroid_x_A" in table.column_names
    assert table.num_rows == 2


def test_edge_and_charge_counts_decode_categories_instead_of_summing_labels():
    bonds = np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]])
    assert _bond_edge_count(bonds) == 2
    labels = np.array([1, 2, 0, 1])
    mask = np.array([True, True, True, False])
    assert _charged_atom_count(labels, mask, {"<PAD>": 0, "0": 1, "1": 2}) == 1


def test_visible_features_drop_oracle_and_selection_channels():
    visible = _visible_source({
        "radius_gyration_A": 2.0,
        "online_pic50": 7.2,
        "selection_probability": 0.5,
        "weight_on": 1.0,
        "rank_best1": 1,
        "pic50_off": 2.4,
        "final_score": 8.0,
        "smiles": "C",
    })
    assert visible == {"radius_gyration_A": 2.0, "online_pic50": 7.2}


def test_pool_view_namespaces_both_structures_and_keeps_only_current_affinity():
    current = {
        "time": 0.2, "step": 20, "online_pic50": 7.1,
        "centroid_x_A": 1.0, "radius_gyration_A": 2.0,
        "delta_current_radius_gyration_A": 0.2,
        "structure_hash": "private", "locator": {"path": "raw.npz"},
    }
    forecast = {
        "time": 0.2, "step": 20, "online_pic50": 7.1,
        "centroid_x_A": 1.5, "radius_gyration_A": 2.3,
        "delta_forecast_radius_gyration_A": -0.1,
        "forecast_revision_rms_A": 0.4,
    }
    view = _pool_view(current, forecast)
    assert view["current_centroid_x_A"] == 1.0
    assert view["forecast_centroid_x_A"] == 1.5
    assert view["delta_current_Rg"] == 0.2
    assert view["delta_forecast_Rg"] == -0.1
    assert view["delta_forecast_revision_rms_A"] == 0.4
    assert view["online_pic50"] == 7.1
    assert not {"batch", "slot", "root_slot", "final_score", "selection_probability", "locator"} & set(view)


def test_structure_signature_includes_on_target_affinity_and_probability_bits():
    arrays = {
        "current_coords": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "predicted_coords": np.ones((1, 1, 2, 3), dtype=np.float32),
        "mask": np.ones((1, 1, 2), dtype=np.uint8),
        "current_atomics": np.ones((1, 1, 2), dtype=np.uint8),
        "current_bonds": np.zeros((1, 1, 2, 2), dtype=np.uint8),
        "current_charges": np.zeros((1, 1, 2), dtype=np.uint8),
        "predicted_atomics": np.ones((1, 1, 2), dtype=np.uint8),
        "predicted_bonds": np.zeros((1, 1, 2, 2), dtype=np.uint8),
        "predicted_charges": np.zeros((1, 1, 2), dtype=np.uint8),
        "predicted_atomics_probs_f16": np.zeros((1, 1, 2, 2), dtype=np.float16),
        "predicted_charges_probs_f16": np.zeros((1, 1, 2, 2), dtype=np.float16),
        "pic50_on": np.array([[7.0]], dtype=np.float32),
    }
    first = _sig(arrays, 0, 0)
    arrays["pic50_on"][0, 0] = 7.1
    assert _sig(arrays, 0, 0) != first


def test_target_ranking_uses_relief_per_heavy_not_absolute_mmff():
    import pandas as pd

    rows = pd.DataFrame({
        "node_id": ["b", "a", "c", "d"],
        "valid_connected": [True, True, True, True],
        "pb_full_pass": [True, True, True, True],
        "energy_status": ["converged"] * 4,
        "mmff_relief_per_heavy": [0.1, 0.2, 0.3, 9.0],
        "strain_MMFF94s_kcal_mol": [99.0, 1.0, 0.0, 0.0],
        "pic50_on_rescore": [8.0, 8.0, 7.0, 9.0],
    })
    selected, p75 = _rank_top30(rows)
    assert np.isclose(p75, 2.475)
    assert selected.node_id.tolist() == ["b", "a", "c"]


def test_frozen_study_is_verified_and_never_silently_rebuilt(tmp_path):
    import json
    import pytest

    study = tmp_path / "study"
    study.mkdir()
    artifact = study / "evidence" / "pools" / "t005.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"step":5,"time":0.05,"cards":[]}', encoding="utf-8")
    frozen = {
        "source_hashes": {"trajectory.npz": "source-hash"},
        "analysis_hashes": {"nodes.csv.gz": "analysis-hash"},
        "preparation_code_sha256": "code-hash",
        "artifact_manifest": _artifact_manifest(study),
    }
    runtime = study / "runtime" / "run_status.json"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("{}", encoding="utf-8")
    (study / "protocol.json").write_text(json.dumps(frozen), encoding="utf-8")
    assert _verify_frozen_study(study, frozen["source_hashes"], frozen["analysis_hashes"], "code-hash") == frozen
    with pytest.raises(RuntimeError, match="refusing to rebuild"):
        _verify_frozen_study(study, frozen["source_hashes"], frozen["analysis_hashes"], "changed-code")
    artifact.write_text('{"step":5,"time":0.05,"cards":[{}]}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifacts changed"):
        _verify_frozen_study(study, frozen["source_hashes"], frozen["analysis_hashes"], "code-hash")


def test_runtime_files_are_allowed_before_and_after_preparation(tmp_path):
    import pytest

    study = tmp_path / "study"
    (study / "diagnostics").mkdir(parents=True)
    (study / "diagnostics" / "notes.json").write_text("{}", encoding="utf-8")
    assert _verify_frozen_study(study, {}, {}, "code") is None
    partial = study / "oracle" / "teacher_map.json"
    partial.parent.mkdir()
    partial.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="partial preparation"):
        _verify_frozen_study(study, {}, {}, "code")
