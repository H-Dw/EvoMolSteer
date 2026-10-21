import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import httpx
import pytest

spec = importlib.util.spec_from_file_location("plip_launch", Path(__file__).parents[1] / "scripts/launch_plip_adaptive.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_authenticated_probe_does_not_write_secrets(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-private")
    monkeypatch.setenv("https_proxy", "http://test:private-proxy@proxy.invalid:3128")
    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-key-private"
        return httpx.Response(200, json={"data": []})
    receipt = module.check_api(transport=httpx.MockTransport(respond))
    assert receipt["authenticated_models_http_status"] == 200
    assert "https_proxy" in [k.lower() for k in receipt["proxy_variables_present"]]
    assert "private" not in json.dumps(receipt)


@pytest.mark.parametrize("status", [401, 403, 429, 502])
def test_bad_api_status_blocks_launch_without_leaking_body(monkeypatch, status):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        module.check_api(transport=httpx.MockTransport(lambda request: httpx.Response(status, text="secret")))
    assert "secret" not in str(error.value)


def test_network_exception_does_not_leak_proxy_credentials(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    def fail(request):
        raise httpx.ConnectError("proxy http://user:secret@host")
    with pytest.raises(RuntimeError, match="ConnectError") as error:
        module.check_api(transport=httpx.MockTransport(fail))
    assert "secret" not in str(error.value)


def test_launcher_inherits_proxy_and_preserves_prior_errors_and_config(tmp_path, monkeypatch):
    for name in ("PREPARED.json", "config.json", "teachers.json"):
        (tmp_path / name).write_text('{}')
    old = tmp_path / "versions/v1/teacher/batch_000/round_01/error_01.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"reason":"original failure"}')
    monkeypatch.setattr(module, "idle_locks", lambda root: contextlib.nullcontext())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-private")
    monkeypatch.setenv("https_proxy", "http://test:private-proxy@proxy.invalid:3128")
    captured = {}
    def spawn(command, **kwargs):
        captured.update(kwargs)
        assert "prepare" not in command
        return SimpleNamespace(pid=12345)
    receipt = module.launch(tmp_path, transport=httpx.MockTransport(lambda r: httpx.Response(200)), popen=spawn)
    assert {k.lower(): v for k, v in captured["env"].items()}["https_proxy"].endswith("proxy.invalid:3128")
    assert captured["env"]["DEEPSEEK_API_KEY"] == "test-key-private"
    assert captured["start_new_session"] is True
    operation = Path(receipt["operation"])
    assert "private" not in (operation / "request.json").read_text()
    with zipfile.ZipFile(operation / "previous_control_and_errors.zip") as archive:
        assert archive.read(old.relative_to(tmp_path).as_posix()) == old.read_bytes()
    for name in ("PREPARED.json", "config.json", "teachers.json"):
        assert (tmp_path / name).read_text() == '{}'


def test_failed_connectivity_never_spawns_or_changes_experiment(tmp_path, monkeypatch):
    (tmp_path / "PREPARED.json").write_text('{}')
    monkeypatch.setattr(module, "idle_locks", lambda root: contextlib.nullcontext())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    def forbidden(*args, **kwargs):
        pytest.fail("Must not launch after an unsuccessful probe")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        module.launch(tmp_path, transport=httpx.MockTransport(lambda r: httpx.Response(401)), popen=forbidden)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["PREPARED.json"]
