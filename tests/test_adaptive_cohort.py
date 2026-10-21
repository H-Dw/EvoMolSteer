import copy

import pytest

from evomolsteer.adaptive_schedule.cohort import TEACHERS, counts, progress, select_templates
from evomolsteer.adaptive_schedule.io import DEFAULT_ROOT, defaults, read, write, sha, value_sha
from evomolsteer.adaptive_schedule.runner import prepare, assert_prepared


def templates():
    return [{"teacher_id": t["teacher_id"], "target": t["target"], "rank": t["rank"],
             "core_atoms": [0], "source_record": {"node_id": t["node_id"]}} for t in TEACHERS]


def test_original_ranks_are_kept_and_unselected_empty_templates_do_not_block():
    active = templates()
    archive = active[::-1] + [{"teacher_id": "excluded", "target": "5ost_AEZ_A_915",
                              "rank": 1, "core_atoms": []}]
    assert select_templates(archive, TEACHERS) == active
    assert [t["rank"] for t in active] == [1, 2, 3]
    assert "r123_i4" in str(DEFAULT_ROOT)
    assert counts(defaults(), active) == {"teacher_count": 3, "target_count": 1,
        "initial_batches": 5, "teacher_batches": 15, "chains_per_version": 300,
        "decisions_per_version": 3000, "r0_results": 300, "maximum_results_all_versions": 6300}


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_node", "renumber", "empty"])
def test_selection_cannot_silently_replace_a_teacher(problem):
    active = templates()
    if problem == "missing":
        active.pop()
    elif problem == "duplicate":
        active.append(copy.deepcopy(active[0]))
    elif problem == "wrong_node":
        active[0]["source_record"]["node_id"] = "other"
    elif problem == "renumber":
        active[1]["rank"] = 4
    else:
        active[0]["core_atoms"] = []
    with pytest.raises(ValueError):
        select_templates(active, TEACHERS)


def completed_batch(root, teacher, batch, rounds=10):
    folder = root / "versions/v1" / teacher / f"batch_{batch:03d}"
    chains = [{"slot": i, "best_id": "r00", "history": [{"id": f"r{r:02d}"} for r in range(rounds + 1)]}
              for i in range(20)]
    write(folder / "histories.json", chains)
    write(folder / "COMPLETE.json", {"chains": 20, "rounds": 10})
    write(folder / "round_10/RETENTION_COMPLETE.json", {"history_sha256": value_sha(chains)})
    return folder


def test_fifteen_actual_batches_complete_the_three_teacher_cohort(tmp_path):
    active = templates()
    for t in active:
        for b in range(5):
            completed_batch(tmp_path, t["teacher_id"], b)
    state = progress(tmp_path, "v1", defaults(), active)
    assert state["complete"] and state["completed_batches"] == 15
    folder = tmp_path / "versions/v1" / active[-1]["teacher_id"] / "batch_004"
    (folder / "COMPLETE.json").unlink()
    assert not progress(tmp_path, "v1", defaults(), active)["complete"]
    completed_batch(tmp_path, "5ost_AEZ_A_915__top1", 0)
    state = progress(tmp_path, "v1", defaults(), active)
    assert not state["complete"] and len(state["unexpected"]) == 1
    assert state["completed_batches"] == 14


def test_marker_without_ten_rounds_or_finished_retention_is_not_completion(tmp_path):
    active = templates()
    folder = completed_batch(tmp_path, active[0]["teacher_id"], 0, rounds=9)
    state = progress(tmp_path, "v1", defaults(), active)
    assert state["completed_batches"] == 0 and state["invalid"]
    completed_batch(tmp_path, active[0]["teacher_id"], 0)
    (folder / "round_10/RETENTION_COMPLETE.json").unlink()
    assert progress(tmp_path, "v1", defaults(), active)["invalid"]


def test_prepare_does_not_overwrite_existing_ten_teacher_experiment(tmp_path):
    previous = defaults()
    previous["targets"].append("5ost_AEZ_A_915")
    write(tmp_path / "config.json", previous)
    write(tmp_path / "teachers.json", [{"historical": "keep unchanged"}])
    before = {p.name: sha(p) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="new cohort"):
        prepare(tmp_path)
    assert before == {p.name: sha(p) for p in tmp_path.iterdir()}


def test_prepare_filters_source_catalog_and_hash_binds_selected_scope(tmp_path, monkeypatch):
    from evomolsteer.adaptive_schedule import runner
    from evomolsteer.adaptive_schedule.vendor import common
    history = tmp_path / "source"
    active = templates()
    for t in active:
        sdf = history / (t["teacher_id"] + ".sdf")
        sdf.parent.mkdir(exist_ok=True)
        sdf.write_text("teacher fixture")
        t["sdf"] = str(sdf)
        t["protein"] = str(history / "protein.pdb")
    (history / "protein.pdb").write_text("protein fixture")
    excluded = {"teacher_id": "excluded", "target": "5ost_AEZ_A_915", "rank": 1,
                "core_atoms": [], "source_record": {"node_id": "unused"}}
    write(history / "teachers.json", active + [excluded])
    config = defaults()
    config["history"] = str(history)
    for b, seed in enumerate(config["seeds"]):
        p = history / "initials" / config["targets"][0] / f"batch_{b:03d}.pt.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"historical prior fixture")
        write(p.with_suffix(".json"), {"seed": seed, "state_file_sha256": sha(p)})
    for arm in ["steer", "unguided"]:
        write(history / arm / "final_records.json", [{"slot": i} for i in range(100)])
    monkeypatch.setattr(runner, "defaults", lambda: config)
    monkeypatch.setattr(runner, "commit", lambda: "test-commit")
    monkeypatch.setattr(common, "BASELINE_DATASETS", {a: str(history / a) for a in ["steer", "unguided"]})
    monkeypatch.setattr(common, "campaign", lambda arm, target: history / arm)
    monkeypatch.setattr(common, "protein", lambda target: history / "protein.pdb")
    root = tmp_path / "new"
    result = prepare(root)
    assert result["teacher_count"] == 3 and result["chains_per_version"] == 300
    assert read(root / "teachers.json") == active
    assert len(list((root / "baselines").glob("*.json"))) == 2
    assert read(root / "teacher_selection.json")["excluded"][0]["teacher_id"] == "excluded"
    assert assert_prepared(root) == config
    assert prepare(root)["already_prepared"]
    changed = copy.deepcopy(active)
    changed[0]["rank"] = 2
    write(root / "teachers.json", changed)
    with pytest.raises(ValueError, match="file changed"):
        assert_prepared(root)


def test_reports_preserve_poor_rounds_and_separate_them_from_top1(tmp_path, monkeypatch):
    from evomolsteer.adaptive_schedule import agent, quality
    from evomolsteer.adaptive_schedule.analysis import analyze
    import csv
    t = templates()[0]
    t.update(atom_labels=[1], charge_labels=[0], coords_A=[[0, 0, 0]], interactions=[],
             source_record={"pic50_on_rescore": 7})
    write(tmp_path / "teachers.json", [t])
    write(tmp_path / "config.json", defaults())
    monkeypatch.setattr(agent, "node", lambda *args: {"local_loss": 1, "local_atom_labels": [1], "local_charge_labels": [0]})
    monkeypatch.setattr(quality, "metrics", lambda *args: {"quality_evaluated": False})
    monkeypatch.setattr(quality, "metrics_folder", lambda *args: {"quality_evaluated": False})
    def record(batch, slot, score):
        return {"batch": batch, "slot": slot, "node_id": f"b{batch}_s{slot}", "pic50_on_rescore": score, "smiles": "CO",
                "build_success": True, "connected": True, "failure_reason": ""}
    for arm in ["steer", "unguided"]:
        write(tmp_path / "baselines" / f"{t['target']}__{arm}.json", [record(b, s, 6) for b in range(5) for s in range(20)])
    for b in range(5):
        chains = []
        for slot in range(20):
            histories = [{"id": f"r{r:02d}", "branch": f"b{b}/r{r}", "parent_id": None,
                          "ticks": [5]*4, "record": record(b, slot, 5 if r == 0 else 8 if r == 1 else 4)}
                         for r in range(11)]
            chains.append({"slot": slot, "best_id": "r01", "history": histories})
        write(tmp_path / "versions/v1" / t["teacher_id"] / f"batch_{b:03d}/histories.json", chains)
    analyze(tmp_path, "v1")
    summary = {r["arm"]: r for r in read(tmp_path / "analysis/v1/summary.json") if not r["deduplicated"]}
    assert summary["v1_best_per_root"]["mean"] == 8
    assert summary["v1_fixed_r10"]["mean"] == 4
    with (tmp_path / "analysis/v1/fixed_round_distributions.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 22
    assert all(r["complete_round"] == "True" and r["attempts"] == "100" for r in rows)
    # Removing a final attempt must not fill it using an earlier round.
    path = tmp_path / "versions/v1" / t["teacher_id"] / "batch_004/histories.json"
    chains = read(path)
    chains[0]["history"].pop()
    write(path, chains)
    analyze(tmp_path, "v1")
    assert not any(r["arm"] == "v1_fixed_r10" for r in read(tmp_path / "analysis/v1/summary.json"))
    with (tmp_path / "analysis/v1/fixed_round_distributions.csv").open(encoding="utf-8-sig", newline="") as f:
        last = [r for r in csv.DictReader(f) if r["round"] == "10"]
    assert all(r["complete_round"] == "False" and r["attempts"] == "99" for r in last)
