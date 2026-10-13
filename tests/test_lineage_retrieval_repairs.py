import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evomolsteer.lineage_retrieval.repairs import archive_round_trial, semantic_repair_targets


def test_semantic_repair_targets_supports_legacy_scientist_text_and_exact_selector_steps():
    audit = {
        "passes": False,
        "scientist_assessment": "A teacher-path description makes an unsupported claim about a reversal.",
        "unsupported_claims": [],
        "selector_assessments": [
            {"step": 5, "evidence_truth": True, "clause_applied": True},
            {"step": 15, "evidence_truth": False, "clause_applied": True},
            {"step": 50, "evidence_truth": True, "clause_applied": False},
            {"step": 7, "evidence_truth": False, "clause_applied": False},
            {"step": True, "evidence_truth": False, "clause_applied": False},
        ],
    }
    assert semantic_repair_targets(audit) == {"scientist": True, "selector_steps": [15, 50]}


def test_semantic_repair_targets_supports_explicit_scientist_flags_and_scoped_claims():
    assert semantic_repair_targets({
        "scientist_evidence_truth": True,
        "scientist_clause_applied": False,
        "scientist_assessment": "No unsupported scientist claims were found.",
        "unsupported_claims": ["Selector cites a value incorrectly."],
        "selector_assessments": [],
    }) == {"scientist": True, "selector_steps": []}
    assert semantic_repair_targets({
        "scientist_evidence_truth": True,
        "scientist_clause_applied": True,
        "scientist_assessment": "No unsupported claims; teacher descriptions are accurate.",
        "unsupported_claims": ["Selector feature citation is unsupported."],
        "selector_assessments": [],
    }) == {"scientist": False, "selector_steps": []}
    assert semantic_repair_targets({
        "scientist_assessment": "Hypotheses were mostly grounded.",
        "unsupported_claims": ["One hypotheses summary asserts a reversal not present in donor histories."],
        "selector_assessments": [],
    }) == {"scientist": True, "selector_steps": []}


def test_archive_moves_only_allowlisted_current_outputs_and_preserves_inputs_and_attempts(tmp_path):
    study = tmp_path / "study"
    round_dir = study / "rounds" / "R01"
    scientist = round_dir / "scientist"
    selector = round_dir / "selectors" / "t005"
    failed = scientist / "attempts" / "A01" / "answer.json"
    mapping = selector / "mapping.json"
    unlisted = round_dir / "oracle.json"
    for path in (scientist, selector, failed.parent):
        path.mkdir(parents=True, exist_ok=True)
    contents = {
        scientist / "answer.json": b"scientist answer",
        scientist / "functional_audit.json": b"scientist audit",
        scientist / "controller_binding.json": b"scientist binding",
        selector / "answer.json": b"selector answer",
        selector / "receipt.json": b"selector receipt",
        selector / "controller_binding.json": b"selector binding",
        mapping: b"input mapping",
        failed: b"FAILED attempt must remain untouched",
        round_dir / "metrics.json": b"metrics",
        round_dir / "hypotheses.json": b"hypotheses",
        round_dir / "_version.json": b"version",
        round_dir / "semantic_audit.json": b"audit",
        unlisted: b"do not archive arbitrary files",
    }
    for path, payload in contents.items():
        path.write_bytes(payload)

    archived = archive_round_trial(study, round_dir, "semantic evidence audit failed")
    assert archived == round_dir / "rejected" / "T01"
    manifest = json.loads((archived / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reason"] == "semantic evidence audit failed"
    records = {row["source"]: row for row in manifest["files"]}
    expected = {
        "scientist/answer.json", "scientist/functional_audit.json", "scientist/controller_binding.json",
        "selectors/t005/answer.json", "selectors/t005/receipt.json", "selectors/t005/controller_binding.json",
        "metrics.json", "hypotheses.json", "_version.json", "semantic_audit.json",
    }
    assert set(records) == expected
    for rel, record in records.items():
        archived_file = archived / rel
        assert archived_file.is_file()
        assert record["sha256"] == hashlib.sha256(archived_file.read_bytes()).hexdigest()
        assert record["size_bytes"] == archived_file.stat().st_size
    assert mapping.read_bytes() == b"input mapping"
    assert failed.read_bytes() == b"FAILED attempt must remain untouched"
    assert unlisted.read_bytes() == b"do not archive arbitrary files"
    assert not (scientist / "answer.json").exists()


def test_archive_rejects_round_paths_outside_the_study_rounds_root(tmp_path):
    study = tmp_path / "study"
    (study / "rounds" / "R01").mkdir(parents=True)
    outside = tmp_path / "R02"
    outside.mkdir()
    with pytest.raises(ValueError, match="direct child"):
        archive_round_trial(study, outside, "must remain contained")


def test_archive_requires_reason_and_does_not_overwrite_existing_trial(tmp_path):
    study = tmp_path / "study"
    round_dir = study / "rounds" / "R00"
    round_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="reason"):
        archive_round_trial(study, round_dir, " ")
    existing = round_dir / "rejected" / "T01"
    existing.mkdir(parents=True)
    (existing / "manifest.json").write_text("keep", encoding="utf-8")
    (round_dir / "metrics.json").write_text("old", encoding="utf-8")
    archived = archive_round_trial(study, round_dir, "second archive")
    assert archived.name == "T02"
    assert (existing / "manifest.json").read_text(encoding="utf-8") == "keep"
