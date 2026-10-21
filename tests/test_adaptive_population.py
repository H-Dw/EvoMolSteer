import copy

import pytest

from evomolsteer.adaptive_schedule.io import defaults, read, write, sha
from evomolsteer.adaptive_schedule.grid import INTERVALS, CHECKPOINT_STEPS, HEAD_STEPS
from evomolsteer.adaptive_schedule.contracts import expanded, earliest
from evomolsteer.adaptive_schedule.population import round_context, stats


def record(score, valid=True):
    return {"pic50_on_rescore": score, "build_success": valid, "connected": valid, "failure_reason": ""}


def setup(root):
    config = defaults()
    teacher = {"teacher_id": "5nn4_SC2_A_1017__top1", "target": "5nn4_SC2_A_1017",
               "source_record": {"pic50_on_rescore": 7.8}}
    folder = root / "versions/v1" / teacher["teacher_id"]
    for b in range(5):
        write(folder / f"batch_{b:03d}/histories.json", [
            {"slot": s, "best_id": "r01", "history": [
                {"id": "r00", "record": record(4)}, {"id": "r01", "record": record(5)}]}
            for s in range(20)])
    for arm in ["unguided", "steer"]:
        write(root / "baselines" / f"{teacher['target']}__{arm}.json", [record(3) for _ in range(100)])
    return config, teacher, folder


def test_four_intervals_cover_exactly_requested_native_steps():
    values = expanded([1, 2, 3, 4])
    active = [i for i, v in enumerate(values) if v]
    assert active == list(range(10, 91))
    for i, (start, end) in enumerate(INTERVALS):
        assert values[start:end + 1] == [(i + 1) / 10] * (end - start + 1)
        new = [5]*4
        new[i] = 6
        assert earliest([5]*4, new) == start
        assert start in CHECKPOINT_STEPS
    assert all(s in CHECKPOINT_STEPS and s in HEAD_STEPS for s in [25, 26, 50, 51, 75, 76, 90, 91])


def test_every_batch_sees_same_pre_round_population_on_resume(tmp_path):
    config, teacher, folder = setup(tmp_path)
    context = round_context(tmp_path, config, teacher, "v1", 2)
    assert context["priority"] == ["mean_pic50", "best_pic50"]
    assert context["previous_unselected_round"]["mean_pic50"] == 5
    assert context["incumbent_population"]["attempts"] == 100
    path = folder / "round_contexts/round_02.json"
    old_hash = sha(path)
    chains = read(folder / "batch_000/histories.json")
    for c in chains:
        c["history"].append({"id": "r02", "record": record(9)})
        c["best_id"] = "r02"
    write(folder / "batch_000/histories.json", chains)
    assert round_context(tmp_path, config, teacher, "v1", 2) == context
    assert sha(path) == old_hash
    with pytest.raises(ValueError, match="five batches"):
        round_context(tmp_path, config, teacher, "v1", 3)


def test_population_keeps_failed_slots_and_separates_latest_from_incumbents(tmp_path):
    config, teacher, folder = setup(tmp_path)
    for b in range(5):
        path = folder / f"batch_{b:03d}/histories.json"
        chains = read(path)
        for c in chains:
            c["history"].append({"id": "r02", "record": record(2)})
        chains[0]["history"][-1]["record"] = record(None, False)
        write(path, chains)
    context = round_context(tmp_path, config, teacher, "v1", 3)
    assert context["previous_unselected_round"] == {"attempts": 100, "eligible": 95, "failed": 5, "mean_pic50": 2, "best_pic50": 2}
    assert context["incumbent_population"]["mean_pic50"] == 5
    assert stats([record(None, False)])["mean_pic50"] is None


def test_changed_past_evidence_cannot_overwrite_round_context(tmp_path):
    config, teacher, folder = setup(tmp_path)
    round_context(tmp_path, config, teacher, "v1", 2)
    path = folder / "batch_004/histories.json"
    chains = read(path)
    chains[0]["history"][1]["record"] = record(7)
    write(path, chains)
    with pytest.raises(ValueError, match="context changed"):
        round_context(tmp_path, config, teacher, "v1", 2)


def test_runner_advances_full_population_before_next_agent_round(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from evomolsteer.adaptive_schedule import runner
    from evomolsteer.adaptive_schedule.io import value_sha
    config = defaults()
    config["execution"]["scheduler"] = "serial"
    config["rounds"] = 2
    teacher = {"teacher_id": "5nn4_SC2_A_1017__top1", "target": "5nn4_SC2_A_1017",
               "source_record": {"pic50_on_rescore": 7.8}}
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr(runner, "assert_prepared", lambda root: config)
    monkeypatch.setattr(runner, "source_fingerprint", lambda: {})
    monkeypatch.setattr(runner, "bind_version", lambda *args: tmp_path / "skill.md")
    from contextlib import nullcontext
    monkeypatch.setattr(runner, "lock", lambda root: nullcontext())
    monkeypatch.setitem(sys.modules, "evomolsteer.adaptive_schedule.flowr", SimpleNamespace(load_model=lambda *args: object()))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    write(tmp_path / "VALIDATED.json", {"source_fingerprint": {}, "restart_pass": True, "fast_batch_pass": True})
    write(tmp_path / "validation/gradient_checks.json", {"source_fingerprint": {}, "passed": True})
    write(tmp_path / "teachers.json", [teacher])
    for arm in ["unguided", "steer"]:
        write(tmp_path / "baselines" / f"{teacher['target']}__{arm}.json", [record(3)] * 100)
    calls = []
    def batch_run(model, root, cfg, t, b, version, skill, fast, through_round, population):
        folder = root / "versions" / version / t["teacher_id"] / f"batch_{b:03d}"
        p = folder / "histories.json"
        if through_round == 0:
            chains = [{"slot": s, "best_id": "r00", "history": [{"id": "r00", "record": record(4)}]}
                      for s in range(20)]
        else:
            assert population["previous_unselected_round"]["mean_pic50"] == 3 + through_round
            assert population["previous_unselected_round"]["attempts"] == 100
            chains = read(p)
            for c in chains:
                c["history"].append({"id": f"r{through_round:02d}", "record": record(4 + through_round)})
                c["best_id"] = f"r{through_round:02d}"
        calls.append((through_round, b))
        write(p, chains)
        write(folder / f"round_{through_round:02d}/RETENTION_COMPLETE.json", {"history_sha256": value_sha(chains)})
        if through_round == cfg["rounds"]:
            write(folder / "COMPLETE.json", {"chains": 20, "rounds": cfg["rounds"]})
    monkeypatch.setattr(runner, "run_batch", batch_run)
    runner.run(tmp_path, "v1")
    assert calls == [(r, b) for r in range(3) for b in range(5)]
    assert read(tmp_path / "versions/v1/COMPLETE.json")["chains"] == 100
