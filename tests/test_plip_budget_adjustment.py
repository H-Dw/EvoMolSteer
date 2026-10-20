import importlib.util
from pathlib import Path

import pytest

from evomolsteer.adaptive_schedule.io import read, write, sha

spec = importlib.util.spec_from_file_location("budget_adjustment", Path(__file__).parents[1] / "scripts/adjust_plip_prompt_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup(root):
    write(root / "config.json", {"reward": "fixed", "llm": {"input_hard_tokens": 65536, "max_output_tokens": 16384}})
    write(root / "versions/v1/binding.json", {"config_sha256": sha(root / "config.json"), "skill_sha256": "same"})
    write(root / "versions/v1/example/error_01.json", {"reason": "original failure"})


def test_budget_migration_preserves_scientific_config_archive_and_is_idempotent(tmp_path):
    setup(tmp_path)
    result = module.apply_budget(tmp_path, revision="test")
    config = read(tmp_path / "config.json")
    assert config == {"reward": "fixed", "llm": {"input_hard_tokens": 131072, "max_output_tokens": 16384}}
    assert read(tmp_path / "versions/v1/binding.json") == {"config_sha256": sha(tmp_path / "config.json"), "skill_sha256": "same"}
    assert module.apply_budget(tmp_path, revision="test") == result
    import zipfile
    with zipfile.ZipFile(Path(result["operation"]) / "pre_change_records.zip") as z:
        assert "original failure" in z.read("versions/v1/example/error_01.json").decode()


def test_bad_binding_is_rejected_before_configuration_changes(tmp_path):
    setup(tmp_path)
    original = sha(tmp_path / "config.json")
    write(tmp_path / "versions/v1/binding.json", {"config_sha256": "wrong"})
    with pytest.raises(ValueError, match="binding"):
        module.apply_budget(tmp_path, revision="test")
    assert sha(tmp_path / "config.json") == original


def test_partial_update_can_resume_without_overwriting_unrelated_changes(tmp_path):
    setup(tmp_path)
    result = module.apply_budget(tmp_path, revision="test")
    journal = read(Path(result["operation"]) / "journal.json")
    write(tmp_path / "config.json", journal["updates"][0]["before"])
    module.apply_budget(tmp_path, revision="test")
    edited = read(tmp_path / "config.json")
    edited["reward"] = "unexpected"
    write(tmp_path / "config.json", edited)
    with pytest.raises(ValueError, match="concurrent"):
        module.apply_budget(tmp_path, revision="test")
