"""Run on the specified remote host. No network, real model or candidate sampling."""
import copy
import gzip
import json
import math
from pathlib import Path

import pytest

from agenticgen.uniguide.calibration import phase_profiles, bind_calibration
from agenticgen.uniguide.scale_domain import (derive_domains, derive_teacher_domain, resolve_domain,
    PHASE_LIMITS, PHASE_COUNTS, GROWTH_LIMIT)
from agenticgen.uniguide.io import write, value_sha
from agenticgen.uniguide.protocol import validate_action


TEACHERS = [dict(target="a", node_id="node_a"), dict(target="b", node_id="node_b")]


def curve(target, offset=0):
    rows = []
    for j, c in enumerate([.9, .8, .7, .5, .3, 2., 8.]):
        scale = 2.**((j+offset)/2)
        # This teacher's first numerical growth boundary is one step above optimum.
        amplitude = scale if j < 5 else scale*1000
        rows.append(dict(target=target, phase="full", attempts=50, id=f"{target}/{j}",
            schedule=[scale]*4, complete=True, C=c, gap=c, best=8.,
            window_diagnostics=[dict(increment_vector_rms_A=amplitude, max_atom_increment_A=amplitude*3)]*4))
    return rows


def phases(target, offset=0):
    rows = []
    for phase in PHASE_COUNTS:
        for r in curve(target, offset):
            row = dict(r, phase=phase, attempts=PHASE_COUNTS[phase], id=r["id"]+"/"+phase)
            if phase == "windows":
                row["schedule"] = row["schedule"][:3]+[row["schedule"][3]*math.sqrt(2)]
            rows.append(row)
    return rows


def test_another_teacher_cannot_change_any_selection_stage():
    a, b = phases("a"), phases("b", 10)
    combined = a+b
    changed = a+phases("b", -10)
    for phase in ["fine", "windows", "full"]:
        assert phase_profiles(combined, "a", phase) == phase_profiles(changed, "a", phase)
    before, after = derive_domains(combined, TEACHERS), derive_domains(changed, TEACHERS)
    assert before["by_teacher"]["a"] == after["by_teacher"]["a"]
    assert before["by_teacher"]["b"] != after["by_teacher"]["b"]
    assert "initial" not in before and before["cross_teacher_pooling"] is False


def test_different_teacher_distributions_produce_shifted_initial_and_bounds():
    domains = derive_domains(curve("a")+curve("b", 6), TEACHERS)["by_teacher"]
    a, b = domains["a"], domains["b"]
    assert b["initial"][0] == pytest.approx(a["initial"][0]*8)
    assert b["lower"] == pytest.approx(a["lower"]*8)
    assert b["upper"] == pytest.approx(a["upper"]*8)
    assert a["max_index"] == 4 and a["min_index"] == 2
    assert a["initial_source"] == "a/3" and a["performance_best_source"] == "a/4"
    assert a["risk_boundary"]["index"] == 5
    assert a["stability_guarantee"] is False


def test_absolute_jump_detected_despite_favorable_gap_and_finite_metrics():
    rows = curve("a")
    rows[5].update(C=.1, gap=0.)
    domain = derive_teacher_domain(rows, TEACHERS[0])
    assert domain["max_index"] == 4
    assert domain["risk_boundary"]["normalized_growth"]["max_atom_increment_A"] > GROWTH_LIMIT
    # Linear growth in scale alone is not a numerical-growth alarm.
    for r in rows:
        s = r["schedule"][0]
        r["window_diagnostics"] = [dict(increment_vector_rms_A=s, max_atom_increment_A=3*s)]*4
    assert derive_teacher_domain(rows, TEACHERS[0])["risk_boundary"] is None


def test_full_population_only_not_coarse_averages_or_failed_rows():
    rows = curve("a")
    baseline = derive_teacher_domain(rows, TEACHERS[0])
    coarse = dict(rows[-1], phase="coarse", attempts=10, C=0., gap=0.)
    assert derive_teacher_domain(rows+[coarse], TEACHERS[0]) == baseline
    failed = rows[4]
    failed.update(complete=False, error="nonfinite")
    failed.pop("window_diagnostics")
    domain = derive_teacher_domain(rows, TEACHERS[0])
    assert domain["max_index"] == 3 and domain["risk_boundary"]["reason"] == "failed_full_constant"
    assert domain["constant_comparison"][4]["status"] == "failed"


@pytest.mark.parametrize("change", ["population", "duplicate", "unknown", "nonfinite", "too_few"])
def test_uncertain_or_incomparable_evidence_does_not_silently_choose_range(change):
    rows = curve("a")
    if change == "population": rows[0]["attempts"] = 20
    if change == "duplicate": rows.append(copy.deepcopy(rows[0]))
    if change == "unknown": rows[0].pop("window_diagnostics")
    if change == "nonfinite": rows[0]["C"] = float("nan")
    if change == "too_few": rows = rows[:2]
    with pytest.raises(ValueError): derive_teacher_domain(rows, TEACHERS[0])


def test_identity_legacy_rejection_and_returned_copy():
    domains = derive_domains(curve("a")+curve("b"), TEACHERS)
    local = resolve_domain(domains, TEACHERS[0])
    local["initial"][0] = 0
    assert domains["by_teacher"]["a"]["initial"][0] > 0
    with pytest.raises(ValueError): resolve_domain(domains, dict(target="a", node_id="other"))
    with pytest.raises(ValueError, match="Legacy"): resolve_domain(dict(initial=[1.]*4), TEACHERS[0])


def test_teacher_specific_actions_and_phase_budget():
    domains = derive_domains(curve("a")+curve("b", 6), TEACHERS)
    a = resolve_domain(domains, TEACHERS[0])
    before = a["initial"]
    anchors = {"R0/a": dict(schedule=before)}
    proposal = dict(anchor_id="R0/a", mode="fine", branch="increase", schedule=[before[0]*math.sqrt(2)]+before[1:])
    validate_action(proposal, anchors, a, [])
    proposal.update(mode="coarse", schedule=[before[0]*2]+before[1:])
    with pytest.raises(ValueError, match="domain"): validate_action(proposal, anchors, a, [])
    assert 2*sum(PHASE_COUNTS[p]*PHASE_LIMITS[p] for p in PHASE_COUNTS) == 5000
    for phase in PHASE_COUNTS:
        assert len(phase_profiles(phases("a"), "a", phase)) <= PHASE_LIMITS[phase]


def test_legacy_artifacts_are_not_migrated_or_overwritten(tmp_path):
    folder = tmp_path/"calibration"
    write(folder/"coarse_profiles.json", [[1.]*4])
    before = (folder/"coarse_profiles.json").read_bytes()
    with pytest.raises(ValueError, match="frozen"): bind_calibration(folder, TEACHERS)
    assert (folder/"coarse_profiles.json").read_bytes() == before
    assert not (folder/"POLICY.json").exists()


def test_r0_checks_both_teachers_before_touching_first(tmp_path, monkeypatch):
    import agenticgen.uniguide.pipeline as pipeline
    domains = derive_domains(curve("a")+curve("b", 6), TEACHERS)
    write(tmp_path/"R0/b/COMPLETE.json", {"legacy": True})
    def forbidden(*args, **kwargs): raise AssertionError("No model, API, sampling or queue writes")
    monkeypatch.setattr(pipeline, "load_model", forbidden)
    monkeypatch.setattr(pipeline, "execute", forbidden)
    monkeypatch.setattr(pipeline, "enqueue", forbidden)
    with pytest.raises(ValueError, match="another scale protocol"):
        pipeline.run_r0(dict(root=str(tmp_path)), TEACHERS, domains)
    assert not (tmp_path/"R0/a").exists()


def test_r0_dispatches_the_correct_teacher_initial(tmp_path, monkeypatch):
    import agenticgen.uniguide.pipeline as pipeline
    domains = derive_domains(curve("a")+curve("b", 6), TEACHERS)
    calls = []
    monkeypatch.setattr(pipeline, "load_model", lambda *args: object())
    monkeypatch.setattr(pipeline, "enqueue", lambda *args: None)
    monkeypatch.setattr(pipeline, "execute", lambda model, config, teacher, schedule, *args:
        calls.append((teacher["target"], schedule)))
    pipeline.run_r0(dict(root=str(tmp_path)), TEACHERS, domains)
    assert calls == [(t["target"], domains["by_teacher"][t["target"]]["initial"]) for t in TEACHERS]
    assert calls[0][1] != calls[1][1]


def test_council_rejects_wrong_teacher_before_provider(tmp_path):
    from agenticgen.uniguide.council import advise
    local = derive_domains(curve("a")+curve("b"), TEACHERS)["by_teacher"]["b"]
    with pytest.raises(ValueError, match="another teacher"):
        advise({}, TEACHERS[0], "UG01", 1, [], local, tmp_path/"no_skills", tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_real_calibration_derivation_and_all_evidence_preserved():
    path = Path(__file__).parents[2]/"docs/evidence/uniguide/20261027_teacher_calibration_source.json.gz"
    data = json.loads(gzip.decompress(path.read_bytes()))
    rows = [r for p in PHASE_COUNTS for r in data[f"calibration/{p}_results.json"]]
    before = value_sha(rows)
    result = derive_domains(rows, data["teachers.json"])["by_teacher"]
    a, b = result["6pyy_H7S_A_402"], result["3w54_RNB_A_504"]
    assert a["initial"] == [128.]*4 and b["initial"] == [2.**7.5]*4
    assert (a["min_index"], a["max_index"]) == (13, 15)
    assert (b["min_index"], b["max_index"]) == (13, 16)
    assert a["risk_boundary"]["index"] == 16 and b["risk_boundary"]["index"] == 17
    assert value_sha(rows) == before
