"""Long explanations must survive the real two-role provider contract."""

import importlib.util
import json
import os
from pathlib import Path

import httpx
import jsonschema
import pytest

from agenticgen.uniguide import linear_council, linear_scale as scale
from agenticgen.uniguide.io import read, sha
from agenticgen.uniguide.tool_runtime import assets, configure
from test_linear_guidance import action, domain, prepared, row


@pytest.mark.parametrize("length", [701, 3000])
def test_action_explanations_count_characters_and_keep_identity_limits(
    tmp_path, length
):
    candidate = action(row(50))
    candidate.update(improvement_reason="探" * length, uncertainty="索" * length)
    schema = scale.action_schema([candidate["anchor_id"]], domain(tmp_path))
    jsonschema.validate(candidate, schema)
    candidate["id"] = "x" * 81
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(candidate, schema)


@pytest.mark.parametrize("field", ["improvement_reason", "uncertainty"])
def test_action_explanations_reject_3001_characters(tmp_path, field):
    candidate = action(row(50))
    candidate[field] = "探" * 3001
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(candidate, scale.action_schema([candidate["anchor_id"]]))


def test_long_role_and_action_reasons_need_no_format_repair(tmp_path, monkeypatch):
    from agenticgen.providers.deepseek_tools import call_with_tools
    from agenticgen.uniguide.absolute_coordinate import POLICY

    c, _, d, base, _, _ = prepared(tmp_path, monkeypatch)
    c = configure(
        dict(
            c,
            protocol=POLICY,
            agent_tools=dict(c["agent_tools"], skill_release="2.8.1"),
        )
    )
    d = scale.seal(
        {
            **{k: v for k, v in d.items() if k != "domain_sha256"},
            "exploration_policy": c["guidance_exploration"],
        }
    )
    skills = tmp_path / "skills"
    for name, path in assets(c).items():
        skills.mkdir(exist_ok=True)
        (skills / name).write_bytes(path.read_bytes())
    proposal = dict(
        action(base),
        exploration_tier="focused",
        improvement_reason="跳" * 3000,
        uncertainty="探" * 3000,
    )
    answers = [
        dict(decision="propose", reason="理" * 3000, proposals=[proposal]),
        dict(decision="propose", reason="由" * 3000, selected=[proposal["id"]]),
    ]
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["max_tokens"] == 16384
        return httpx.Response(
            200,
            json=dict(
                model="deepseek-flash",
                usage=dict(completion_tokens=20),
                choices=[
                    dict(
                        finish_reason="stop",
                        message=dict(
                            role="assistant",
                            content=json.dumps(
                                answers[len(requests) - 1], ensure_ascii=False
                            ),
                        ),
                    )
                ],
            ),
        )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-placeholder")
    result = linear_council.advise(
        c,
        [base],
        d,
        tmp_path / "advice",
        skills,
        provider=lambda *a, **kw: call_with_tools(
            *a, **kw, transport=httpx.MockTransport(handle)
        ),
    )
    assert len(requests) == 2 and len(result["actions"]) == 1
    for role in ("proposal", "synthesis"):
        receipt = read(tmp_path / "advice" / (role + ".receipt.json"))
        assert receipt["interface_repairs"] == 0 and receipt["steps"] == 1
        schema = read(tmp_path / "advice" / (role + ".prompt_source.json"))[
            "response_schema"
        ]
        assert schema["properties"]["reason"]["maxLength"] == 3000
        invalid = dict(answers[0 if role == "proposal" else 1], reason="字" * 3001)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


def test_release_hashes_and_previous_release_are_preserved():
    root = (
        Path(__file__).resolve().parents[2]
        / "src/agenticgen/resources/skill_releases/guidance"
    )
    release = read(root / "2.8.1/RELEASE.json")
    assert release["explanation_max_characters"] == 3000
    for name, digest in release["old_files_sha256"].items():
        assert sha(root / "2.8.0" / name) == digest
    for name, digest in release["files_sha256"].items():
        assert sha(root / "2.8.1" / name) == digest
    runtime = root / "2.8.1/runtime/linear_absolute_coordinate"
    for name in ("proposal.md", "synthesis.md"):
        assert "3000" in (runtime / name).read_text(encoding="utf-8")


def test_five_round_configuration_and_launch_contract_agree(tmp_path):
    from agenticgen.uniguide.absolute_runtime import validate_config
    from agenticgen.uniguide.linear_calibration import ensure_limits

    repo = Path(__file__).resolve().parents[2]
    config = read(repo / "configs/uniguide_absolute_coordinate.json")
    config.update(read(repo / "configs/uniguide_linear_control.json"))
    config.update(read(repo / "configs/uniguide_comparison5.json"))
    config.update(
        root=str(tmp_path / "new"), generation_context_root=str(tmp_path / "old")
    )
    config = validate_config(config)
    assert config["rounds"] == 5
    assert config["agent_tools"]["skill_release"] == "2.8.1"
    limits = ensure_limits(config, 2)
    assert limits["stage_caps"] == dict(calibration=700, formal=1500)
    path = repo / "docs/experiments/20261031_guidance_comparison5/run_suite.py"
    spec = importlib.util.spec_from_file_location("comparison5_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config["conditions"] = config["conditions"][:1]
    job = dict(rounds=5, skill_release="2.8.1")
    assert runner.validate_job_config(config, job) == 5
    with pytest.raises(ValueError, match="launch contract"):
        runner.validate_job_config(dict(config, rounds=8), job)
    # The runner reads the external count rather than imposing five itself.
    assert runner.validate_job_config(dict(config, rounds=2), dict(job, rounds=2)) == 2


def test_durable_mirror_keeps_safe_terminal_and_excludes_launcher(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/20261031_guidance_comparison5/durable_snapshot.py"
    )
    spec = importlib.util.spec_from_file_location("durable_mirror_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, target = tmp_path / "run", tmp_path / "nfs"
    source.mkdir()
    for name in (
        "aggregate.json",
        "output.pt.gz",
        "structures.sdf",
        "launcher.log",
        "trace.h5",
        "core.123",
    ):
        (source / name).write_bytes(b"fixture")
    report = module.mirror(source, target)
    assert report["files"] == 3 and not report["full_hdf5_included"]
    assert not (target / "launcher.log").exists() and not (target / "trace.h5").exists()
    assert read(target / "MIRROR_STATE.json")["output.pt.gz"]["sha256"] == sha(
        source / "output.pt.gz"
    )
    old_mtime = (source / "aggregate.json").stat().st_mtime_ns
    (source / "aggregate.json").write_bytes(b"updated")
    # NFS can expose identical size/mtime after a same-length rewrite.
    os.utime(source / "aggregate.json", ns=(old_mtime, old_mtime))
    module.mirror(source, target)
    assert (target / "aggregate.json").read_bytes() == b"updated"
    with pytest.raises(ValueError, match="outside"):
        module.mirror(source, source / "bad")
