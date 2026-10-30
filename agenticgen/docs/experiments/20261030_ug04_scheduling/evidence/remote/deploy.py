"""Independent fixed-commit CPU regression; never start a GPU campaign."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

COMMIT = "ea9e97237d70413536b4f8b23bf51123b2940866"
AUDIT = Path("/workspace/uniguide_audits/ug04-serial-ea9e9723")
REPO = AUDIT / "checkout"
SOURCE = Path("/workspace/uniguide_audits/persistent-gpu-e45a70f0/checkout")
BUNDLE_SHA = "647af2f2ed88cb792fbd301ffe3180ecb591fe1ac584db9c0bb4d5f02996996f"
FILES = ["test_absolute_sampling_recovery.py", "test_absolute_coordinate_protocol.py",
    "test_absolute_uncapped_statistics.py", "test_coordinate_statistics.py",
    "test_uniguide_geometry_batch.py", "test_uniguide_geometry.py", "test_uniguide_geometry_dynamics.py",
    "test_geometry_tensor_contract.py", "test_geometry_reward_controls.py", "test_geometry_chemical_state.py",
    "test_guidance_skill_release_v2.py", "test_guidance_analysis_tools.py", "test_guidance_skill_bundle.py",
    "test_uncapped_campaign.py", "test_adaptive_calibration.py", "test_ug04_task_scheduling.py"]


def stamp():
    return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, data):
    (AUDIT / name).write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n")


def checked(args, *, cwd=None, timeout=120):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(str(args[:3]) + ": " + result.stderr[-2000:] + result.stdout[-2000:])
    return result.stdout


def main():
    if REPO.exists():
        raise RuntimeError("Fixed checkout already exists; do not replace or repeat it")
    checked(["git", "clone", "--shared", "--no-checkout", str(SOURCE), str(REPO)])
    attempts = []
    for prefix in ("", "https://ghfast.top/", "https://gh-proxy.com/", "https://ghproxy.net/", "https://gh.llkk.cc/"):
        try:
            result = subprocess.run(["git", "fetch", prefix + "https://github.com/H-Dw/EvoMolSteer.git", COMMIT],
                cwd=REPO, capture_output=True, text=True, timeout=8)
            attempts.append(dict(transport=prefix or "direct", exit_code=result.returncode))
            if result.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            attempts.append(dict(transport=prefix or "direct", timeout_seconds=8))
    else:
        bundle = AUDIT / "pushed.bundle"
        actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if actual != BUNDLE_SHA:
            raise ValueError("Pushed bundle changed")
        checked(["git", "bundle", "verify", str(bundle)], cwd=REPO)
        checked(["git", "fetch", str(bundle), "refs/heads/codex/uniguide-flowr-agent"], cwd=REPO)
        attempts.append(dict(transport="verified_pushed_bundle", sha256=actual, pushed_commit=COMMIT))
    checked(["git", "sparse-checkout", "init", "--cone"], cwd=REPO)
    checked(["git", "sparse-checkout", "set", "agenticgen/src", "agenticgen/tests", "agenticgen/scripts",
        "agenticgen/configs", "src", "agenticgen/docs/experiments/20261030_uncapped_coordinate",
        "agenticgen/docs/experiments/20261030_adaptive_calibration",
        "agenticgen/docs/experiments/20261030_ug04_scheduling",
        "agenticgen/docs/experiments/20261029_absolute_coordinate_formal",
        "agenticgen/docs/experiments/20261029_chemical_interact_formal"], cwd=REPO)
    checked(["git", "checkout", "-B", "codex/uniguide-flowr-agent", COMMIT], cwd=REPO)
    save("deployment.json", dict(timestamp=stamp(), commit=COMMIT, fetch=attempts,
        clean=not checked(["git", "status", "--porcelain"], cwd=REPO), gpu_started=False, llm_calls=0))
    environment = (
        "source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1\n"
        f"export PYTHONPATH={REPO}/agenticgen/src:{REPO}/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root\n"
        "export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        "unset DEEPSEEK_API_KEY\n"
    )
    executable = "/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python"
    arguments = [executable, "-m", "pytest", "-q", "-c", "agenticgen/pyproject.toml",
        *("agenticgen/tests/unit/" + name for name in FILES),
        "--basetemp=" + str(AUDIT / "pytest"), "--junitxml=" + str(AUDIT / "regression.xml")]
    save("STATUS.json", dict(timestamp=stamp(), phase="cpu_regression", commit=COMMIT, files=FILES))
    with (AUDIT / "regression.log").open("w") as log:
        result = subprocess.run(["bash", "-c", environment + shlex.join(arguments)], cwd=REPO,
            stdout=log, stderr=subprocess.STDOUT, timeout=300)
    suite = ET.parse(AUDIT / "regression.xml").getroot().find("testsuite")
    summary = dict(timestamp=stamp(), commit=COMMIT, exit_code=result.returncode, files=FILES,
        tests=int(suite.get("tests")), failures=int(suite.get("failures")),
        errors=int(suite.get("errors")), skipped=int(suite.get("skipped")),
        seconds=float(suite.get("time")), gpu_started=False, real_model_acceptance=False,
        scientific_attempts=0, llm_calls=0)
    summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
    save("REGRESSION.json", summary)
    if result.returncode:
        raise RuntimeError("CPU regression failed; inspect the preserved log")
    command = [executable, "agenticgen/docs/experiments/20261030_ug04_scheduling/audit_sources.py",
               str(AUDIT / "SOURCE_AUDIT.json")]
    checked(["bash", "-c", environment + shlex.join(command)], cwd=REPO)
    summary["runtime_clean"] = not checked(["git", "status", "--porcelain"], cwd=REPO)
    save("COMPLETE.json", summary)
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save("FAILED.json", dict(timestamp=stamp(), error=repr(error), gpu_started=False, no_automatic_retry=True))
        raise
