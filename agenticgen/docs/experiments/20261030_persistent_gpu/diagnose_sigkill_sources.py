"""Read saved evidence and compare execution paths; never load a model or CUDA."""
import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np

NORMAL = "930db03280c66c3f8a39f131ae01df3202544af4"
TRANSIENT = "7670549a4b8f4e2fbddf3c8da8e6f9c620954cdd"
CURRENT = "399590dd9d723ec3bbfce86fb0f017cf2c6ecb61"
PACKAGE = "agenticgen/src/agenticgen/uniguide/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(repo, revision, name):
    return subprocess.check_output(
        ["git", "show", revision + ":" + PACKAGE + name], cwd=repo
    ).decode("utf-8")


def function(text, name, class_name=None):
    tree = ast.parse(text)
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef)
                    and node.name == class_name).body
    return next(node for node in body if isinstance(node, ast.FunctionDef)
                and node.name == name)


def call_names(node):
    return [ast.unparse(item.func) for item in ast.walk(node)
            if isinstance(item, ast.Call)]


def run(repo, archive, destination):
    modules = ("prepare.py", "native_tools.py", "random_streams.py", "batching.py")
    unchanged = {}
    for name in modules:
        old, current = (source(repo, rev, name) for rev in (NORMAL, CURRENT))
        unchanged[name] = dict(normal_sha256=digest(old.encode()),
                               current_sha256=digest(current.encode()), identical=old == current)
    vjp = {}
    coefficients = {}
    for revision in (NORMAL, CURRENT):
        forward = function(source(repo, revision, "sampler.py"), "forward", "Control")
        vjp[revision] = [ast.dump(item, include_attributes=False) for item in ast.walk(forward)
                         if isinstance(item, ast.Call)
                         and ast.unparse(item.func) == "torch.autograd.grad"]
        update = function(source(repo, revision, "condition.py"), "guidance_delta")
        coefficients[revision] = [ast.dump(item, include_attributes=False) for item in ast.walk(update)
                                  if isinstance(item, ast.Assign)
                                  and any(ast.unparse(target) in ("coefficient[active]", "delta")
                                          for target in item.targets)]
    transient = function(source(repo, TRANSIENT, "chunking.py"), "concurrent_map")
    serial = function(source(repo, CURRENT, "gpu_execution.py"), "map", "TeacherGpuExecutor")
    serial_branch = next(node for node in serial.body if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "self.workers == 1")
    serial_calls = call_names(serial_branch)
    acceptance = function(source(repo, CURRENT, "absolute_acceptance.py"), "run")
    acceptance_calls = call_names(acceptance)
    root = archive / "gpu399/root"
    batch = root / "acceptance/3w54_RNB_A_504/candidate_batch/resource_stability/batch_007"
    reference = batch.parent / "batch_005"
    steps = json.loads((batch / "steps.json").read_text())
    same, nonfinite, shapes = {}, [], {}
    with h5py.File(batch / "trace.h5") as left, h5py.File(reference / "trace.h5") as right:
        for name in left:
            a, b = left[name][:len(steps)], right[name][:len(steps)]
            same[name] = np.array_equal(a, b)
            if not np.isfinite(a).all():
                nonfinite.append(name)
            shapes[name] = dict(shape=list(left[name].shape), dtype=str(left[name].dtype),
                                full_uncompressed_bytes=left[name].size * left[name].dtype.itemsize)
    ranges = {}
    for key in ("loss", "gradient_norm", "max_atom_increment_A", "delta_native_ratio"):
        values = np.asarray([row[key] for row in steps], dtype=float)
        ranges[key] = dict(minimum=float(values.min()), maximum=float(values.max()),
                           finite=bool(np.isfinite(values).all()))
    resources = {}
    for target in ("6pyy_H7S_A_402", "3w54_RNB_A_504"):
        folder = root / "acceptance" / target / "candidate_batch/resource_stability"
        rows = json.loads((folder / "RESOURCES.json").read_text())
        keys = ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
                "blas_workspace_count", "blaslt_workspace_count", "native_gpu_queue_count")
        resources[target] = dict(completed_batches=len(rows),
            ranges={key: dict(minimum=min(row[key] for row in rows),
                              maximum=max(row[key] for row in rows)) for key in keys},
            recorded_fields=sorted(rows[0]),
            host_memory_time_series_recorded=any("rss" in key or "cgroup" in key for key in rows[0]))
    started = json.loads((archive / "gpu399/audit/acceptance_started.json").read_text())
    ended = json.loads((archive / "gpu399/audit/acceptance_finished.json").read_text())
    elapsed = (datetime.fromisoformat(ended["timestamp"]) -
               datetime.fromisoformat(started["timestamp"])).total_seconds()
    result = dict(timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="read_only_source_and_saved_tensor_comparison", normal_commit=NORMAL,
        prior_transient_commit=TRANSIENT, actual_failed_gpu_commit=CURRENT,
        unchanged_modules=unchanged, vjp_call_ast_equal=vjp[NORMAL] == vjp[CURRENT],
        guidance_increment_arithmetic_ast_equal=coefficients[NORMAL] == coefficients[CURRENT],
        previous_transient_map=dict(calls=call_names(transient),
            constructs_threads="ThreadPoolExecutor" in call_names(transient),
            constructs_streams="torch.cuda.Stream" in call_names(transient)),
        current_serial_branch=dict(calls=serial_calls,
            constructs_threads="ThreadPoolExecutor" in serial_calls,
            constructs_streams="torch.cuda.Stream" in serial_calls,
            submits_futures=any("submit" in call for call in serial_calls)),
        acceptance_path=dict(direct_model_load="load_model" in acceptance_calls,
            uses_teacher_process="TeacherProcess" in acceptance_calls,
            uses_execution_model="execution_model" in acceptance_calls,
            binds_executor="bind_executor" in acceptance_calls,
            kill_or_terminate_calls=[call for call in acceptance_calls
                                     if call.endswith(("kill", "killpg", "terminate"))]),
        termination=dict(pid=started["pid"], exit_code=ended["exit_code"], elapsed_seconds=elapsed,
                         runner_timeout_seconds=7200, timeout_reached=elapsed >= 7200),
        interrupted_prefix=dict(saved_steps=len(steps), saved_last_step=steps[-1]["step"],
            all_hdf5_datasets_exact=all(same.values()), comparisons=same,
            nonfinite_datasets=nonfinite, diagnostic_ranges=ranges, datasets=shapes,
            terminal_present_in_selected_export=(batch / "terminal.pt.gz").exists(),
            checkpoint_100_present_in_selected_export=(batch / "checkpoint_100.pt.gz").exists(),
            missing_terminal_status_also_verified_by_remote_terminal_artifact_audit=True),
        resources=resources, new_scientific_attempts=0, llm_calls=0,
        signal_sender_identified=False,
        no_new_gpu_probe_or_sampling=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    receipt = run(Path(__file__).resolve().parents[4], args.archive, args.destination)
    print(json.dumps({key: receipt[key] for key in
        ("vjp_call_ast_equal", "guidance_increment_arithmetic_ast_equal", "termination")}, ensure_ascii=True))
