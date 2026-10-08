"""Synthetic complete panels exercise integrity rejection, not model efficacy."""
import copy
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/validate_flowcompat_confirmation.py"
SPEC = importlib.util.spec_from_file_location("confirmation_validator", SCRIPT)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def write_json(path, data, *, crlf=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_bytes(text.replace("\n", "\r\n").encode() if crlf else text.encode())


def sha(text):
    return validator.hashlib.sha256(text.encode()).hexdigest()


def controller_raw_sha(data):
    """Match the distinct unsorted/no-trailing-newline controller serialization."""
    return sha(json.dumps(data, indent=2, default=str))


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE).decode().strip()


def refresh_retention(panel, number):
    folder = panel / "reports" / f"round{number:02d}"
    record = validator.read_json(folder / "retention.json")
    record["files"] = [{"path": path.relative_to(folder).as_posix(), "sha256": validator.digest(path), "bytes": path.stat().st_size}
                       for path in sorted(folder.rglob("*")) if path.is_file() and path.name != "retention.json"]
    write_json(folder / "retention.json", record)


@pytest.fixture
def panel(tmp_path):
    repo, cfg, docs = tmp_path, tmp_path / "configurations", tmp_path / "reports"
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    # A real miniature Git import closure, including a function-local import.
    sources = {
        "src/evomolsteer/__init__.py": "",
        "src/evomolsteer/generation/__init__.py": "",
        "src/evomolsteer/generation/endpoint_controller.py": "def main():\n    pass\n",
        "src/evomolsteer/generation/flowcompat_controller.py": "from .endpoint_controller import main\n",
        "src/evomolsteer/generation/flowcompat_v2_controller.py": "from .flowcompat_controller import main\ndef optional():\n    from . import helper\n",
        "src/evomolsteer/generation/helper.py": "VALUE = 1\n",
        "scripts/generate_affinity_endpoint_flowr.py": "from evomolsteer.generation.endpoint_controller import main\n",
        "scripts/generate_flowcompat_flowr.py": "from evomolsteer.generation.flowcompat_controller import main\n",
        "scripts/generate_flowcompat_v2_flowr.py": "from evomolsteer.generation.flowcompat_v2_controller import main\n",
        "scripts/attest_flowr_sources.py": "# synthetic read-only source attester\n",
        "scripts/bind_flowr_source_attestation.py": "# synthetic metadata-only binder\n",
    }
    for name, text in sources.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Integrity Test", "-c", "user.email=integrity@example.test", "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic numerical closure")
    commit = git(repo, "rev-parse", "HEAD")
    ref = repo / "configs/reference.bin"
    ref.parent.mkdir(exist_ok=True)
    ref.write_bytes(b"synthetic immutable teacher coordinates")
    baseline = {"seed": 42, "window": [0., .5], "reference_sha256": validator.digest(ref),
                "reward_view": "endpoint_pointcloud", "native_rms_ratio": .33,
                "derivative_path": "flowr_endpoint_vjp", "affinity_head_gradient": False}
    write_json(repo / "configs/experiments/skill_ablation_v1/incumbent.json", baseline)
    expected_code = validator.GitSources(repo).closure(commit, "scripts/generate_flowcompat_v2_flowr.py")
    candidate = {**copy.deepcopy(baseline), "generation_interface": "flowcompat_v2", "reward_view": "endpoint_branch_mixture",
                 "branch_mixture": {"virtual_mass": .1, "direction_sign": 1., "region_weight_mix": 0.},
                 "flowcompat_provenance": {"workflow_version": "flowcompat-supplemental-agent-2.0",
                     "code_files": expected_code, "request_sha256": sha("request"),
                     "parameter_updates": {"branch_mixture.virtual_mass": .1}}}
    upstream_files = {name: {"sha256": sha(name), "bytes": 400} for name in sorted(validator.UPSTREAM_FILES)}
    for number in (1, 2, 8, 9, 25, 26, 27, 28, 29, 30):
        selected = number == 9 or number in validator.CANDIDATE_ROUNDS
        p = copy.deepcopy(candidate if selected else baseline)
        if number == 8:
            p["generation_interface"] = "flowcompat_v2"
        p.update(round=number, program_id=f"flowcompat30_round{number:02d}", derivation={"administrative_round": number})
        program_path = cfg / f"round{number:02d}.json"
        write_json(program_path, p, crlf=True)
        batches = [47, 48] if number < 25 else [49 + 2 * ((number - 25) // 2), 50 + 2 * ((number - 25) // 2)]
        arms = ["gradient"] if selected or number == 8 else ["unguided", "gradient"]
        campaign = f"flowcompat_r{number:02d}"
        signatures = {f"{arm}/{batch}": sha(f"initial batch {batch}") for arm in arms for batch in batches}
        raw_sources = [{"arm": arm, "batch": batch, "sha256": sha(f"raw {number} {arm} {batch}")}
                       for arm in arms for batch in batches]
        exp = {"campaign": campaign, "seed": 42, "steps": 100, "batch": 50, "n": 100,
               "arms": ",".join(arms), "batch_indices": batches, "window_start": 0., "window": .5,
               "checkpoint": "/remote/checkpoints/model.ckpt", "input_dataset": "/remote/historical_steer",
               "reference": str(ref), "verify_passive": False, "export_terminal": True}
        ext = {"program_sha256": validator.digest(program_path), "reference_sha256": p["reference_sha256"],
               "checkpoint_sha256": sha("checkpoint"), "code_commit": commit,
               "control_domain": [0., .5], "evidence_domain": [0., .5],
               "particle_resampling": False, "affinity_head_gradient": False,
               "additional_production_forward_calls_per_step": 0, "apply_guidance": False,
               "post_window_injection": False, "derivative_path": "flowr_endpoint_vjp",
               "native_integrator_parameters": {"integration-steps": 100, "use-sde-simulation": True},
               "native_sde": True, "native_coord_noise_level": .2, "native_categorical_sampling": True}
        config = {"experiment": exp, "extension": ext, "runtime_integrator": {"use_sde_simulation": True, "coord_noise_level": .2},
                  "flowr_args": {"save_dir": "/remote/" + campaign, "scientific_option": "fixed"}}
        if number >= 25:
            before = {"schema_version": "flowr-upstream-source-attestation-1.0", "input_dataset": "/remote/flowr_root",
                      "files": upstream_files, "captured_UTC": "2026-10-09T01:00:00+00:00", "scope": "Exactly four declared files",
                      "script_sha256": sha(sources["scripts/attest_flowr_sources.py"])}
            after = {**copy.deepcopy(before), "captured_UTC": "2026-10-09T02:00:00+00:00", "source_unchanged": True,
                     "reference_sha256": validator.recorded_json_sha(before)}
            config["upstream_source_attestation"] = {
                "schema_version": "bound-flowr-upstream-source-attestation-1.0", "before": before, "after": after,
                "before_record_sha256": validator.recorded_json_sha(before), "after_record_sha256": validator.recorded_json_sha(after),
                "generation_config_sha256_before_binding": controller_raw_sha(config),
                "generation_config_canonical_sha256_before_binding": validator.recorded_json_sha(config),
                "binder_sha256": sha(sources["scripts/bind_flowr_source_attestation.py"])}
        execution = {"campaign": campaign, "code_commit": commit, "steps": 100, "window": [0., .5],
                     "no_particle_resampling": True, "outside_window_injection": False, "affinity_head_gradient": False,
                     "additional_production_calls_per_step": 0, "initial_state_signatures": signatures,
                     "sources": raw_sources, "batch_results": [{"arm": arm, "batch": batch, "n": 50, "controlled_steps": 50 if arm == "gradient" else 0}
                                                                for arm in arms for batch in batches],
                     "preflight": {"passed": True, "affinity_head_gradient": False, "production_extra_forward_calls_per_step": 0,
                                   "derivative_path": "flowr_endpoint_vjp", "frozen_self_condition_and_assignment": True,
                                   "checks": [{"analytic": 1., "numerical": 1., "epsilon": .001, "relative_error": 0.}]}}
        folder = docs / f"round{number:02d}"
        write_json(folder / "inference_config/config.json", config)
        write_json(folder / "inference_config/reward_program.json", p)
        write_json(folder / "inference_config/COMPLETE.json", {"status": "complete", "records": 100 * len(arms)})
        view = {"complete": True, "campaign": campaign, "batches": [
            {"batch_path": f'{r["arm"]}/batch_{r["batch"]:03d}', "steps": 100, "initial_state_signature": signatures[f'{r["arm"]}/{r["batch"]}'],
             "source_trajectory_sha256": r["sha256"], "snapshot_sha256": sha("snapshot " + r["sha256"])} for r in raw_sources]}
        write_json(folder / "inference_config/EVALUATION_VIEW.json", view)
        write_json(folder / "execution_report.json", execution)
        formula = {"null_exact": True, "maximum_gradient_relative_change": .1 if selected else 0.,
                   "samples": [{"time": t / 100, "null_scalar_exact": True, "null_gradient_exact": True,
                                "gradient_relative_change": .1 if selected else 0.} for t in range(50)]}
        if number != 1:
            if number == 2:
                formula["samples"] = [formula["samples"][index] for index in (0, 25, 49)]
            write_json(folder / "conditional_formula_audit.json", formula)
        feedback = {"implementation_passed": selected, "null_control_parity": number in (2, 8), "null_control_terminal_parity": number in (2, 8)}
        feedback["batch_diagnostics"] = [{"arm": arm, "batch": batch, "window_state_sha256": sha(f"window {arm} {batch}")}
                                         for arm in arms for batch in batches]
        write_json(folder / "implementation_feedback.json", feedback)
        fields = ["arm", "batch", "slot", *validator.SCORE_COLUMNS, "energy_status", "mmff_relief_per_heavy", "smiles"]
        with (folder / "candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for arm in arms:
                for batch in batches:
                    for slot in range(50):
                        score = 7. + slot / 1000 + (.2 if selected else .1 if arm == "gradient" else 0.)
                        writer.writerow({"arm": arm, "batch": batch, "slot": slot,
                                         **{field: score for field in validator.SCORE_COLUMNS},
                                         "energy_status": "converged", "mmff_relief_per_heavy": .5, "smiles": "CCO"})
        if selected:
            diagnostics = {"branch_gradient_relative_change": .1, "contrast_gradient_relative_change": .1,
                           "paired_window_coordinate_rms_A": .02, "branch_virtual_mass_mean": .05,
                           "branch_virtual_teacher_shift_rms_A": .02}
            gate_input = {"program_sha256": validator.digest(program_path), "request_sha256": p["flowcompat_provenance"]["request_sha256"],
                          "code_files": expected_code, "execution_checks": dict.fromkeys(validator.EXECUTION_CHECKS, True),
                          "no_op_control": {"exact_baseline_arithmetic": True, "max_gradient_relative_change": 0}, "diagnostics": diagnostics}
            write_json(folder / "implementation_gate_input.json", gate_input)
            write_json(folder / "implementation_gate.json", {
                "implementation_passed": True, "efficacy_assessed": False, "program_sha256": validator.digest(program_path),
                "execution_sha256": validator.digest(folder / "implementation_gate_input.json"), "request_sha256": gate_input["request_sha256"],
                "parameter_updates": p["flowcompat_provenance"]["parameter_updates"], "diagnostics": diagnostics})
        write_json(docs / f"round{number:02d}_plan.json", {"arms": ",".join(arms), "batches": batches, "program_sha256": validator.digest(program_path),
                   "reference_sha256": p["reference_sha256"], "learned_window": [0., .5]})
        write_json(folder / "retention.json", {"campaign": campaign, "status": "complete", "candidate_rows": 100 * len(arms),
                   "files": [], "inference_commit": commit, "raw_trajectory_hashes": raw_sources})
        refresh_retention(repo, number)
    write_json(docs / "frozen_validation.json", {"selected_round": 9, "screening_admissible": True,
               "program_sha256": validator.digest(cfg / "round09.json"), "reference_sha256": baseline["reference_sha256"],
               "before_confirmation_labels": True, "confirmation_batches": list(range(49, 55))})
    # The actual inference commit already contains all immutable programs and
    # the freeze declaration; completion/side reports are written afterwards.
    git(repo, "add", "configurations", "configs", "reports/frozen_validation.json")
    git(repo, "-c", "user.name=Integrity Test", "-c", "user.email=integrity@example.test", "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze synthetic confirmation before inference")
    actual_commit = git(repo, "rev-parse", "HEAD")
    for number in (1, 2, 8, 9, 25, 26, 27, 28, 29, 30):
        folder = docs / f"round{number:02d}"
        config_path = folder / "inference_config/config.json"
        config = validator.read_json(config_path)
        config["extension"]["code_commit"] = actual_commit
        if "upstream_source_attestation" in config:
            original = {key: value for key, value in config.items() if key != "upstream_source_attestation"}
            config["upstream_source_attestation"]["generation_config_sha256_before_binding"] = controller_raw_sha(original)
            config["upstream_source_attestation"]["generation_config_canonical_sha256_before_binding"] = validator.recorded_json_sha(original)
        write_json(config_path, config)
        execution = validator.read_json(folder / "execution_report.json")
        execution["code_commit"] = actual_commit
        write_json(folder / "execution_report.json", execution)
        retention = validator.read_json(folder / "retention.json")
        retention["inference_commit"] = actual_commit
        write_json(folder / "retention.json", retention)
        refresh_retention(repo, number)
    return repo


def audit(panel):
    return validator.ConfirmationAudit(panel, panel / "reports", panel / "configurations").run()


def failed(result, name):
    return [row for row in result["checks"] if row["check"] == name and not row["passed"]]


def mutate_json(panel, number, relative, change):
    path = panel / "reports" / f"round{number:02d}" / relative
    data = validator.read_json(path)
    change(data)
    write_json(path, data)
    refresh_retention(panel, number)


def mutate_csv(panel, number, change):
    path = panel / "reports" / f"round{number:02d}/candidate_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    change(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    refresh_retention(panel, number)


def test_complete_real_git_panel_and_newline_normalized_retained_program_pass(panel):
    result = audit(panel)
    assert result["integrity_passed"], [row for row in result["checks"] if not row["passed"]]
    assert result["eligible_for_efficacy_decision"]
    assert "not assessed" in result["adoption_decision"]
    assert validator.digest(panel / "configurations/round26.json") != validator.digest(panel / "reports/round26/inference_config/reward_program.json")
    binding = validator.read_json(panel / "reports/round26/inference_config/config.json")["upstream_source_attestation"]
    assert binding["generation_config_sha256_before_binding"] != binding["generation_config_canonical_sha256_before_binding"]
    closure = next(row["detail"]["code_files"] for row in result["checks"] if row["check"] == "round26_frozen_source_closure")
    assert "src/evomolsteer/generation/helper.py" in closure


def test_retained_file_tampering_fails_without_relying_on_cached_pass(panel):
    path = panel / "reports/round26/candidate_metrics.csv"
    path.write_bytes(path.read_bytes() + b"tampered\n")
    result = audit(panel)
    assert not result["integrity_passed"]
    assert "checksum mismatch" in failed(result, "round26_retention_and_data")[0]["error"]


@pytest.mark.parametrize("change", [lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                                     lambda rows: rows[0].update(batch="51"), lambda rows: rows[0].update(arm="unguided")])
def test_finite_but_missing_duplicate_or_wrong_panel_fails(panel, change):
    mutate_csv(panel, 26, change)
    result = audit(panel)
    assert not result["integrity_passed"]
    assert failed(result, "round26_retention_and_data")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", ""])
def test_every_attempt_score_must_be_finite_even_with_valid_retention(panel, value):
    mutate_csv(panel, 28, lambda rows: rows[17].update(pic50_off_upstream=value))
    result = audit(panel)
    assert not result["integrity_passed"]
    assert failed(result, "round28_retention_and_data")


def test_initial_state_pair_mismatch_is_rejected(panel):
    bad = sha("different random draw")
    mutate_json(panel, 26, "execution_report.json", lambda data: data["initial_state_signatures"].update({"gradient/49": bad}))
    mutate_json(panel, 26, "inference_config/EVALUATION_VIEW.json", lambda data: data["batches"][0].update(initial_state_signature=bad))
    result = audit(panel)
    assert failed(result, "six_batch_pairing_native_runtime_and_upstream")
    assert not result["eligible_for_efficacy_decision"]


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", sha("different checkpoint")), ("native_coord_noise_level", .4)])
def test_config_runtime_changes_are_not_exempted_as_metadata(panel, field, value):
    mutate_json(panel, 28, "inference_config/config.json", lambda data: data["extension"].update({field: value}))
    result = audit(panel)
    assert not result["integrity_passed"]


def test_executed_commit_cannot_be_substituted_with_a_changed_numeric_module(panel):
    file = panel / "src/evomolsteer/generation/helper.py"
    file.write_bytes(b"VALUE = 2\n")
    git(panel, "add", str(file.relative_to(panel)))
    git(panel, "-c", "user.name=Integrity Test", "-c", "user.email=integrity@example.test", "-c", "commit.gpgsign=false", "commit", "-qm", "Changed numeric code")
    new = git(panel, "rev-parse", "HEAD")
    mutate_json(panel, 26, "execution_report.json", lambda data: data.update(code_commit=new))
    mutate_json(panel, 26, "inference_config/config.json", lambda data: data["extension"].update(code_commit=new))
    path = panel / "reports/round26/retention.json"
    data = validator.read_json(path)
    data["inference_commit"] = new
    write_json(path, data)
    result = audit(panel)
    assert not result["integrity_passed"]
    assert failed(result, "round26_frozen_source_closure")


def test_cached_implementation_true_does_not_override_gate_mismatch(panel):
    mutate_json(panel, 30, "implementation_gate_input.json", lambda data: data["execution_checks"].update(no_extra_production_forward=False))
    # Re-bind the output hash, so rejection must inspect the actual checks.
    mutate_json(panel, 30, "implementation_gate.json", lambda data: data.update(execution_sha256=validator.digest(panel / "reports/round30/implementation_gate_input.json")))
    result = audit(panel)
    assert failed(result, "round30_implementation_gate")
    assert not result["integrity_passed"]


def test_zero_mechanism_is_rejected_even_if_cached_gate_passed(panel):
    for name in ("implementation_gate_input.json", "implementation_gate.json"):
        mutate_json(panel, 30, name, lambda data: data["diagnostics"].update(paired_window_coordinate_rms_A=0.))
    mutate_json(panel, 30, "implementation_gate.json", lambda data: data.update(execution_sha256=validator.digest(panel / "reports/round30/implementation_gate_input.json")))
    result = audit(panel)
    assert failed(result, "round30_implementation_gate")


def test_actual_null_scores_are_compared_and_cannot_be_covered_by_pass_flags(panel):
    mutate_csv(panel, 8, lambda rows: rows[0].update(pic50_on_rescore="8.5"))
    result = audit(panel)
    assert failed(result, "actual_null_control")
    assert not result["integrity_passed"]


def test_null_full_window_hash_is_checked_separately_from_terminal_scores(panel):
    mutate_json(panel, 8, "implementation_feedback.json", lambda data: data["batch_diagnostics"][0].update(window_state_sha256=sha("altered window state")))
    result = audit(panel)
    assert failed(result, "actual_null_control")


def test_frozen_reward_dose_cannot_change_even_when_round_metadata_is_allowed(panel):
    path = panel / "configurations/round28.json"
    data = validator.read_json(path)
    data["native_rms_ratio"] = .5
    write_json(path, data)
    result = audit(panel)
    assert failed(result, "round28_retention_and_data")
    assert "Reward/controller" in failed(result, "round28_retention_and_data")[0]["error"]


def test_frozen_record_presence_is_verified_in_pre_inference_git_commit(panel):
    path = panel / "reports/frozen_validation.json"
    data = validator.read_json(path)
    data["screening_admissible"] = False
    write_json(path, data)
    result = audit(panel)
    assert failed(result, "round25_pre_inference_freeze_snapshot")
    assert not result["integrity_passed"]


def test_missing_bound_upstream_evidence_fails_closed(panel):
    mutate_json(panel, 26, "inference_config/config.json", lambda data: data.pop("upstream_source_attestation"))
    result = audit(panel)
    assert failed(result, "round26_upstream_attestation")
    assert not result["eligible_for_efficacy_decision"]


def test_legacy_raw_only_binding_is_insufficient_not_proof_of_numeric_modification(panel):
    mutate_json(panel, 26, "inference_config/config.json", lambda data: data["upstream_source_attestation"].pop("generation_config_canonical_sha256_before_binding"))
    result = audit(panel)
    failure = failed(result, "round26_upstream_attestation")[0]["error"]
    assert "Legacy upstream binding" in failure and "evidence insufficient" in failure
    assert "fields mismatch" not in failure
    assert not result["integrity_passed"]


def test_real_round18_legacy_binding_is_classified_as_insufficient_evidence():
    path = SCRIPT.parents[1] / "docs/experiments/flowcompat30_20261009/round18/inference_config/config.json"
    if not path.is_file():
        pytest.skip("Historical round18 retained fixture unavailable")
    config = validator.read_json(path)
    assert "generation_config_canonical_sha256_before_binding" not in config["upstream_source_attestation"]
    with pytest.raises(ValueError, match="Legacy upstream binding.*evidence insufficient"):
        validator.verify_generation_config_binding(config)


def test_changed_original_fields_fail_canonical_digest_even_if_raw_sha_is_well_formed(panel):
    mutate_json(panel, 26, "inference_config/config.json", lambda data: data["extension"].update(native_coord_noise_level=.4))
    result = audit(panel)
    failure = failed(result, "round26_upstream_attestation")[0]["error"]
    assert "Canonical pre-binding generation configuration fields mismatch" in failure


def test_raw_sha_is_retained_identity_only_but_still_must_have_valid_syntax(panel):
    mutate_json(panel, 26, "inference_config/config.json", lambda data: data["upstream_source_attestation"].update(generation_config_sha256_before_binding="malformed"))
    result = audit(panel)
    assert "original raw" in failed(result, "round26_upstream_attestation")[0]["error"]


def test_before_after_source_changes_cannot_be_hidden_by_rehashing_metadata(panel):
    def change(data):
        bound = data["upstream_source_attestation"]
        bound["after"]["files"]["flowr/models/pocket.py"]["sha256"] = sha("modified upstream")
        bound["after_record_sha256"] = validator.recorded_json_sha(bound["after"])
    mutate_json(panel, 28, "inference_config/config.json", change)
    result = audit(panel)
    assert failed(result, "round28_upstream_attestation")


def test_consistent_within_round_but_different_across_round_upstream_is_rejected(panel):
    def change(data):
        bound = data["upstream_source_attestation"]
        for key in ("before", "after"):
            bound[key]["files"]["flowr/models/integrator.py"]["sha256"] = sha("new integrator")
        bound["before_record_sha256"] = validator.recorded_json_sha(bound["before"])
        bound["after"]["reference_sha256"] = bound["before_record_sha256"]
        bound["after_record_sha256"] = validator.recorded_json_sha(bound["after"])
    mutate_json(panel, 29, "inference_config/config.json", change)
    result = audit(panel)
    assert not failed(result, "round29_upstream_attestation")
    assert failed(result, "six_batch_pairing_native_runtime_and_upstream")


def test_integrity_pass_is_separate_from_screening_or_affinity_adoption(panel):
    path = panel / "reports/frozen_validation.json"
    data = validator.read_json(path)
    data["screening_admissible"] = False
    write_json(path, data)
    git(panel, "add", "reports/frozen_validation.json")
    git(panel, "-c", "user.name=Integrity Test", "-c", "user.email=integrity@example.test", "-c", "commit.gpgsign=false", "commit", "-qm", "Synthetic inadmissible candidate frozen before confirmation")
    commit = git(panel, "rev-parse", "HEAD")
    for number in range(25, 31):
        mutate_json(panel, number, "execution_report.json", lambda data: data.update(code_commit=commit))
        def update_config(data):
            data["extension"]["code_commit"] = commit
            original = {key: value for key, value in data.items() if key != "upstream_source_attestation"}
            data["upstream_source_attestation"]["generation_config_sha256_before_binding"] = controller_raw_sha(original)
            data["upstream_source_attestation"]["generation_config_canonical_sha256_before_binding"] = validator.recorded_json_sha(original)
        mutate_json(panel, number, "inference_config/config.json", update_config)
        retention = panel / "reports" / f"round{number:02d}/retention.json"
        record = validator.read_json(retention)
        record["inference_commit"] = commit
        write_json(retention, record)
    result = audit(panel)
    assert result["integrity_passed"]
    assert not result["eligible_for_efficacy_decision"]


def test_missing_confirmation_outputs_produce_a_report_and_failure_exit(panel):
    (panel / "reports/round30/retention.json").unlink()
    output = panel / "audit.json"
    code = validator.main(["--repo", str(panel), "--input-dataset", str(panel / "reports"),
                           "--configuration-dataset", str(panel / "configurations"), "--output", str(output)])
    assert code == 1
    assert not validator.read_json(output)["integrity_passed"]


def test_output_cannot_overwrite_any_consumed_evidence(panel):
    path = panel / "reports/frozen_validation.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        validator.main(["--repo", str(panel), "--input-dataset", str(panel / "reports"),
                        "--configuration-dataset", str(panel / "configurations"), "--output", str(path)])
    assert path.read_bytes() == original
