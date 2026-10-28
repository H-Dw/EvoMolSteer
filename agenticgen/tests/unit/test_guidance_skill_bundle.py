"""Offline checks for skill packaging, rendering and the real prompt adapter."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src/agenticgen/resources/skills/guidance-experiment-optimizer"
RUNTIME = ROOT / "src/agenticgen/uniguide/skills"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = load("guidance_skill_renderer", PACKAGE / "scripts/render_runtime.py")


def test_committed_runtime_and_manifest_match_standard_sources():
    assert renderer.sync(RUNTIME, check=True) == []
    rendered = renderer.render()
    binding = json.loads(rendered["BUNDLE.json"])
    for name, expected in binding["output_sha256"].items():
        assert renderer.digest(rendered[name]) == expected


def test_check_reports_drift_without_rewriting_or_removing_other_files(tmp_path):
    renderer.sync(tmp_path)
    sentinel = tmp_path / "existing-audit.json"
    sentinel.write_text("preserve", encoding="utf-8")
    target = tmp_path / "analysis.md"
    target.write_text("local change", encoding="utf-8")
    assert renderer.sync(tmp_path, check=True) == ["analysis.md"]
    assert target.read_text(encoding="utf-8") == "local change"
    renderer.sync(tmp_path)
    assert renderer.sync(tmp_path, check=True) == []
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_windows_and_linux_source_newlines_render_identically(tmp_path):
    copy = tmp_path / "skill"
    shutil.copytree(PACKAGE, copy)
    expected = renderer.render(copy)
    for name in renderer.INPUTS:
        path = copy / name
        normalized = path.read_text(encoding="utf-8")
        path.write_bytes(normalized.replace("\n", "\r\n").encode("utf-8"))
    assert renderer.render(copy) == expected


@pytest.mark.parametrize("damage", ["missing_reference", "reversed_section"])
def test_invalid_sources_do_not_partially_write_outputs(tmp_path, damage):
    copy = tmp_path / "skill"
    shutil.copytree(PACKAGE, copy)
    if damage == "missing_reference":
        (copy / "references/control-diagnostics.md").unlink()
    else:
        path = copy / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        text = text.replace(renderer.SHARED_START, "TEMP_MARKER")
        text = text.replace(renderer.SHARED_END, renderer.SHARED_START)
        path.write_text(text.replace("TEMP_MARKER", renderer.SHARED_END), encoding="utf-8")
    destination = tmp_path / "outputs"
    with pytest.raises((OSError, ValueError)):
        renderer.sync(destination, package=copy)
    assert not destination.exists()


@pytest.mark.parametrize("selection", [False, True])
def test_real_adapter_loads_shared_knowledge_and_only_requested_role(selection):
    policy = load("guidance_skill_adapter", ROOT / "scripts/uniguide_prompt_policy.py")
    text = {"type": "string", "minLength": 1}
    if selection:
        properties = {"analysis": text, "choice_evidence": text, "selected": {"type": "object"}}
        context = {"history": []}
        names = ["analysis.md", "synthesis.md"]
    else:
        action = {name: text for name in ("anchor_reason", "evidence", "prediction", "falsifier")}
        properties = {"analysis": text, "mechanism_uncertainty": text,
                      "proposals": {"type": "array", "items": {"type": "object", "properties": action}}}
        context = {"branch": "increase", "history": []}
        names = ["analysis.md", "proposal.md"]
    schema = {"type": "object", "properties": properties}
    before = json.dumps(schema)
    messages = [{"role": "system", "content": "unrouted input"},
                {"role": "user", "content": json.dumps(context)}]
    rendered, _, metrics = policy.transform(messages, schema, RUNTIME)
    expected = "\n\n".join((RUNTIME / name).read_text(encoding="utf-8") for name in names)
    assert rendered[0]["content"].startswith(expected)
    assert json.dumps(schema) == before
    assert policy.unpack_context(json.loads(rendered[1]["content"])) == context
    assert metrics["lossless_context_verified"]
