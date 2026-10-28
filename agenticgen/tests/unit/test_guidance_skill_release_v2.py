"""Offline artifact, immutability and actual prompt-router checks for release 2."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "src/agenticgen/resources/skill_releases/guidance/2.0.0"
PACKAGE = RELEASE / "skills/guidance-experiment-optimizer"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = load("guidance_release_v2_renderer", PACKAGE / "scripts/render_runtime.py")


def test_release_manifest_binds_every_published_payload():
    manifest = json.loads((RELEASE / "RELEASE.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.name != "RELEASE.json"}
    assert actual == set(manifest["sha256_bytes"])
    for name, expected in manifest["sha256_bytes"].items():
        assert hashlib.sha256((RELEASE / name).read_bytes()).hexdigest() == expected


def test_published_legacy_output_matches_sources_without_mutation():
    assert renderer.sync(RELEASE / "runtime/legacy", check=True) == []


def test_internal_distance_bundle_matches_its_distinct_source_contract():
    directory = RELEASE / "runtime/internal_distance"
    binding = json.loads((directory / "BUNDLE.json").read_text(encoding="utf-8"))
    for role in ("analysis", "proposal", "synthesis"):
        source = RELEASE / "skills" / ("distance-guidance-" + role) / "SKILL.md"
        output = directory / (role + ".md")
        assert output.read_bytes() == source.read_bytes()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == binding["output_sha256"][output.name]


def test_render_is_create_only_and_check_does_not_repair_drift(tmp_path):
    target = tmp_path / "new-version"
    renderer.sync(target)
    changed = target / "analysis.md"
    changed.write_text("local artifact must survive", encoding="utf-8")
    sentinel = target / "audit.json"
    sentinel.write_text("original audit", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert renderer.sync(target, check=True) == ["analysis.md"]
    with pytest.raises(FileExistsError):
        renderer.sync(target)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


def test_existing_empty_destination_is_also_protected(tmp_path):
    with pytest.raises(FileExistsError):
        renderer.sync(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_invalid_sources_do_not_create_partial_destination(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(PACKAGE, source)
    (source / "references/history-learning.md").unlink()
    destination = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        renderer.sync(destination, package=source)
    assert not destination.exists()


def test_cross_platform_newlines_have_identical_rendering(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(PACKAGE, source)
    expected = renderer.render(source)
    for name in renderer.INPUTS:
        path = source / name
        value = path.read_text(encoding="utf-8")
        path.write_bytes(value.replace("\n", "\r\n").encode("utf-8"))
    assert renderer.render(source) == expected


@pytest.mark.parametrize("selection", [False, True])
def test_real_legacy_router_loads_only_frozen_runtime_role_files(selection):
    policy = load("guidance_v2_prompt_adapter", ROOT / "scripts/uniguide_prompt_policy.py")
    text = {"type": "string", "minLength": 1}
    if selection:
        properties = {"analysis": text, "choice_evidence": text, "selected": {"type": "object"}}
        context = {"history": []}
        role = "synthesis.md"
    else:
        action = {name: text for name in ("anchor_reason", "evidence", "prediction", "falsifier")}
        properties = {"analysis": text, "mechanism_uncertainty": text,
                      "proposals": {"type": "array", "items": {"type": "object", "properties": action}}}
        context = {"branch": "increase", "history": []}
        role = "proposal.md"
    schema = {"type": "object", "properties": properties}
    before = json.dumps(schema)
    messages = [{"role": "system", "content": "unrouted"},
                {"role": "user", "content": json.dumps(context)}]
    directory = RELEASE / "runtime/legacy"
    transformed, _, metrics = policy.transform(messages, schema, directory)
    expected = "\n\n".join((directory / name).read_text(encoding="utf-8")
                            for name in ("analysis.md", role))
    assert transformed[0]["content"].startswith(expected)
    assert json.dumps(schema) == before
    assert policy.unpack_context(json.loads(transformed[1]["content"])) == context
    assert metrics["lossless_context_verified"]
    external = (RELEASE / "skills/guidance-postexperiment-rsi/SKILL.md").read_text(encoding="utf-8")
    log = (RELEASE / "CHANGELOG.zh-CN.md").read_text(encoding="utf-8")
    assert external not in transformed[0]["content"]
    assert log not in transformed[0]["content"]
