import copy
import gzip
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch

from evomolsteer.deepseek_api import call_json
from evomolsteer.continuous.m2_skill_api import BASE, REF, compile_reward, instruction, effective_signature
from evomolsteer.generation.coordinate_contrast import make_coordinate_reward
from evomolsteer.io import digest, read_json, write_json

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "docs/experiments/steer_dependency_20261009/deepseek_flash_m2_20261011"
SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"const": True}}}


def test_real_client_retry_secret_free_and_shared_input(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-value")
    seen = []
    def respond(request):
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer secret-test-value"
        if len(seen) == 1: return httpx.Response(429)
        return httpx.Response(200, json={"model": "deepseek-flash", "id": "test",
            "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}],
            "usage": {"total_tokens": 12}})
    path = tmp_path / "role.json"
    kwargs = {"transport": httpx.MockTransport(respond), "sleep": lambda _: None,
              "input_store": tmp_path / "inputs"}
    messages = [{"role": "user", "content": "Return JSON only."}]
    assert call_json(messages, SCHEMA, path, **kwargs) == {"ok": True}
    assert call_json(messages, SCHEMA, path, **kwargs) == {"ok": True}
    assert len(seen) == 2
    assert read_json(path.with_suffix(".receipt.json"))["attempts"] == 2
    assert all(b"secret-test-value" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    assert "content" not in read_json(path.with_suffix(".request.json"))["body"]["messages"][0]


def test_no_fallback_or_fake_success(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-secret")
    with pytest.raises(ValueError, match="exactly"):
        call_json([], SCHEMA, tmp_path / "a.json", model="deepseek-chat")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": [
        {"finish_reason": "length", "message": {"content": '{}'}}]}))
    with pytest.raises(ValueError, match="Incomplete"):
        call_json([], SCHEMA, tmp_path / "a.json", transport=transport)
    assert not (tmp_path / "a.json").exists()


def test_one_neutral_json_repair_preserves_failed_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-secret")
    sent = []
    def respond(request):
        sent.append(json.loads(request.content))
        content = '{"ok":true invalid}' if len(sent) == 1 else '{"ok":true}'
        return httpx.Response(200, json={"model": "deepseek-flash", "choices": [
            {"finish_reason": "stop", "message": {"content": content}}]})
    path = tmp_path / "a.json"
    assert call_json([{"role": "user", "content": "JSON only"}], SCHEMA, path,
                     transport=httpx.MockTransport(respond), input_store=tmp_path / "inputs") == {"ok": True}
    assert len(sent) == 2 and sent[1]["messages"][0] == sent[0]["messages"][0]
    assert "Do not add research advice" in sent[1]["messages"][-1]["content"]
    assert path.with_suffix(".attempt_1.invalid.txt").exists()
    assert len(read_json(path.with_suffix(".receipt.json"))["interface_repairs"]) == 1


@pytest.mark.parametrize("module", ["A1", "A2", "A3", "A4", "D1", "D2", "D3", "D4"])
def test_exact_module_drop_and_alternative(module):
    activation = read_json(STUDY / "activation.json")
    rows = activation["conditions"]
    registry = read_json(STUDY / "module_registry.json")["modules"]
    m = next(m for m in registry if m["id"] == module)
    prefix = "A" if m["role"] == "Analyst" else "D"
    dropped = next(r for r in rows if r["condition_id"] == prefix + "_drop_" + module)
    alternative = next(r for r in rows if r["condition_id"] == prefix + "_alt_" + module)
    drop_text = instruction(REPO, STUDY, dropped, m["role"])
    alt_text = instruction(REPO, STUDY, alternative, m["role"])
    assert m["instruction"] not in drop_text
    assert m["instruction"] not in alt_text
    assert m["alternative_instruction"] in alt_text
    for other in registry:
        if other["role"] == m["role"] and other["id"] != module:
            assert other["instruction"] in drop_text
    assert "flowr_endpoint" not in drop_text or "VJP" in drop_text


def test_compiler_rejects_dose_in_shape_lane_and_false_region(tmp_path):
    study = tmp_path / "study"
    write_json(study / "evidence_index.json", {"regions": {"regional:0": 0}})
    row = {"condition_id": "test", "lane": "G"}
    response = {"decision": "edit", "edit": {"parameter": "native_rms_ratio", "value": .5},
                "evidence_ids": []}
    with pytest.raises(ValueError, match="Illegal"):
        compile_reward(REPO, study, row, 0, response)
    response["edit"] = {"parameter": "region_index", "value": 1}
    response["evidence_ids"] = ["regional:0"]
    with pytest.raises(ValueError, match="not bound"):
        compile_reward(REPO, study, row, 0, response)


def test_declared_edit_has_coordinate_gradient_and_normalized_scalar_is_noop():
    program = read_json(REPO / BASE)
    reference = json.loads(gzip.decompress((REPO / REF).read_bytes()))
    assert reference["window"] == program["window"]
    cloud = np.asarray(reference["frames"][0]["teacher_endpoint_A"][0])
    rng = np.random.default_rng(42)
    x = torch.tensor(cloud[None] + rng.normal(0, .6, (1, len(cloud), 3)), dtype=torch.float64, requires_grad=True)
    atoms = torch.zeros((1, len(cloud)), dtype=torch.long)
    mask = torch.ones((1, len(cloud)), dtype=torch.bool)
    reward = make_coordinate_reward(program, reference)
    value, _ = reward(x, atoms, mask, reference["times"][0], x.detach())
    g = torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
    gscaled = torch.autograd.grad((7 * value).sum(), x)[0]
    assert torch.isfinite(g).all() and g.norm() > 0
    assert torch.allclose(g / g.norm(), gscaled / gscaled.norm(), atol=1e-12)
    edited = copy.deepcopy(program); edited["teacher_neighbors"] = 8
    other, _ = make_coordinate_reward(edited, reference)(x, atoms, mask, reference["times"][0], x.detach())
    g2 = torch.autograd.grad(other.sum(), x)[0]
    assert not torch.allclose(g / g.norm(), g2 / g2.norm(), atol=1e-6)
    # Coordinate finite difference with correspondences frozen at the same anchor.
    direction = g / g.norm(); h = 1e-5
    plus = reward(x.detach() + h * direction, atoms, mask, reference["times"][0], x.detach())[0]
    minus = reward(x.detach() - h * direction, atoms, mask, reference["times"][0], x.detach())[0]
    assert torch.allclose((plus - minus) / (2 * h), (g * direction).sum().reshape(1), rtol=1e-5, atol=1e-7)


def test_default_aliases_do_not_create_fake_independent_rewards():
    base = read_json(REPO / BASE)
    alias = copy.deepcopy(base)
    alias.update(pointcloud_delta_A=1, teacher_neighbors=4.0, program_id="renamed", robust_delta=99)
    assert effective_signature(base) == effective_signature(alias)
