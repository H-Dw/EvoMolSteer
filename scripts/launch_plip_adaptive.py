"""Detach a prepared experiment only after an authenticated, secret-free API check."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import zipfile

import httpx


def check_api(*, transport=None):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing; no worker started")
    started = time.time()
    try:
        with httpx.Client(timeout=20, trust_env=True, transport=transport) as client:
            response = client.get("https://api.deepseek.com/models",
                                  headers={"Authorization": "Bearer " + key})
    except httpx.HTTPError as error:
        # Exception text can include authenticated proxy URLs. Record only type.
        raise RuntimeError(f"DeepSeek connectivity check failed ({type(error).__name__}); no worker started") from None
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek connectivity check HTTP {response.status_code}; no worker started")
    return {"authenticated_models_http_status": 200, "elapsed_seconds": time.time() - started,
            "api_key_present": True,
            "proxy_variables_present": sorted(k for k in os.environ
                if k.lower() in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
                and os.environ[k])}


@contextlib.contextmanager
def idle_locks(root):
    import fcntl
    with contextlib.ExitStack() as stack:
        for name in ("supervisor.lock", "worker.lock"):
            handle = stack.enter_context((root / name).open("a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Experiment is active ({name}); no duplicate launch") from None
        yield


def launch(root, version="v1", skill=None, *, transport=None, popen=subprocess.Popen):
    root = Path(root).resolve()
    if not (root / "PREPARED.json").is_file():
        raise RuntimeError("Prepared experiment missing; launcher never runs prepare")
    repo = Path(__file__).resolve().parents[1]
    with idle_locks(root):
        check = check_api(transport=transport)
        operation = root / "operations" / ("launch_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                                           + "_" + uuid.uuid4().hex[:8])
        operation.mkdir(parents=True)
        # Runner retry indices restart on resume. Preserve all previous failures
        # before their summaries can be updated; request files stay in place.
        controls = [root / n for n in ("launch.json", "workers.json", "status.json",
                                      f"supervisor_{version}_exit.json")]
        controls += list((root / "versions" / version).rglob("*error*.json"))
        with zipfile.ZipFile(operation / "previous_control_and_errors.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(controls)):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        frozen = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in ("PREPARED.json", "config.json", "teachers.json")}
        command = ["bash", str(repo / "scripts/run_plip_adaptive.sh"), "background",
                   "--root", str(root), "--skill-version", version]
        if skill is not None:
            command += ["--skill", str(Path(skill).resolve())]
        request = {"time": time.time(), "repository": str(repo), "version": version,
                   "command": command, "api_check": check, "frozen_sha256": frozen,
                   "environment_policy": "inherit_complete_parent_environment_without_serializing_values"}
        (operation / "request.json").write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    # Child obtains the same locks. A racing launcher is rejected by supervisor.
    # Keep ALL inherited settings, including lower-case authenticated proxies.
    with (root / f"supervisor_{version}.log").open("ab") as log:
        process = popen(command, cwd=repo, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"pid": process.pid, "time": time.time(), "operation": str(operation),
               "worker_ready": False, "note": "Check launch.json, workers.json and actual progress after startup"}
    (operation / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--skill-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--skill", type=Path)
    options = parser.parse_args()
    print(json.dumps(launch(options.root, options.skill_version, options.skill)))
