"""Compare the restored scheduler to Git objects, without importing the runtime."""
import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASELINE = "930db03280c66c3f8a39f131ae01df3202544af4"
BEFORE = "7e2c0234"
PACKAGE = "agenticgen/src/agenticgen/uniguide"


def blob(repo, revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{PACKAGE}/{name}"], cwd=repo)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def run(repo, destination):
    package = repo / PACKAGE
    chunking = (package / "chunking.py").read_bytes()
    assert chunking == blob(repo, BASELINE, "chunking.py")
    scientific = {}
    for name in ("sampler.py", "absolute_coordinate.py", "condition.py", "prepare.py",
                 "geometry_dynamics.py", "geometry_chemistry.py", "chem_inpainting.py", "random_streams.py"):
        current = (package / name).read_bytes()
        assert current == blob(repo, BEFORE, name), name
        scientific[name] = sha(current)
    removed = ("gpu_execution.py", "gpu_worker.py", "gpu_recovery.py",
               "gpu_stability.py", "gpu_resources.py", "workspace_counter.cpp")
    assert all(not (package / name).exists() for name in removed)
    paths = ("chunking.py", "sampler.py", "geometry_batch.py", "geometry_safety.py",
             "absolute_runtime.py", "geometry_runtime.py", "adaptive_calibration.py",
             "geometry_calibration.py", "absolute_acceptance.py", "geometry_acceptance.py")
    prohibited = {"ThreadPoolExecutor", "ProcessPoolExecutor", "torch.cuda.Stream",
                  "TeacherProcess", "subprocess.Popen", "threading.Thread", "threading.RLock"}
    matched = {}
    for name in paths:
        tree = ast.parse((package / name).read_text(encoding="utf-8"))
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        matched[name] = sorted(calls & prohibited)
    assert not any(matched.values()), matched
    config = json.loads((repo / "agenticgen/configs/uniguide_absolute_coordinate.json").read_text())
    assert config["task_scheduling"] == "ug04-930db032-serial" and "gpu_execution" not in config
    assert config["population"] == 50 and config["output_policy"] == "raw999"
    assert config["windows"] == [[10, 25], [26, 50], [51, 75], [76, 99]]
    assert config["chemical_inpainting"] == {"enabled": True}
    result = dict(timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        baseline=BASELINE, before=BEFORE, chunking_byte_identical_to_ug04=True,
        chunking_sha256=sha(chunking), unchanged_scientific_modules=scientific,
        removed_modules=list(removed), prohibited_gpu_scheduling_calls=matched,
        task_scheduling=config["task_scheduling"], logical_population=50, network_microbatch_size=10,
        gpu_network_concurrency=1, cross_configuration_batching=False,
        automatic_oom_fallback=False, gpu_worker_restart=False,
        new_scientific_attempts=0, llm_calls=0, real_model_acceptance_performed=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((json.dumps(result, ensure_ascii=True, indent=2) + "\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[4], args.destination)
    print(json.dumps({key: result[key] for key in
        ("chunking_byte_identical_to_ug04", "task_scheduling", "automatic_oom_fallback")}, ensure_ascii=True))
