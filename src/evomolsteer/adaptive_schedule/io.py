"""Atomic artifacts and frozen experiment configuration (no sampler imports)."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("/opt/MolSteer/generated_datasets/plip_schedule_agent_top1_5nn4_r12_i4_s42_20261022")
HISTORY = Path("/opt/MolSteer/generated_datasets/plip_guidance_5nn4_5ost_s42_20261020")
SEEDS = [42, 100045, 200048, 300051, 400054]
TARGETS = ["5nn4_SC2_A_1017"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def canonical(value):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_directory(path.parent)


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_fingerprint():
    root = Path(__file__).parent
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*.py"))}
    files["../deepseek_api.py"] = sha(root.parent / "deepseek_api.py")
    return files


def repository():
    return Path(__file__).resolve().parents[3]


def commit():
    return subprocess.check_output(["git", "-C", str(repository()), "rev-parse", "HEAD"], text=True).strip()


def defaults():
    from .cohort import TEACHERS
    from .grid import INITIAL_TICKS, CHECKPOINT_STEPS, CONTROL_NODES, description
    return {
        "schema": "plip-adaptive-schedule-5nn4-r12-i4-11762-v6",
        "ssh": {"host": "ssh.zzai.scnet.cn", "port": 11762, "user": "root"},
        "history": str(HISTORY), "flowr": "/opt/MolSteer/flowr_root",
        "targets": list(TARGETS), "teacher_selection": [dict(t) for t in TEACHERS],
        "seeds": list(SEEDS), "batch_size": 20, "batches": 5, "steps": 100,
        "execution": {"scheduler": "interleaved_prefetch", "proposal_lookahead_batches": 2,
                      "gpu_owners": 1, "native_batch_size": 20},
        "features": "C", "mode": "endpoint", "initial_ticks": list(INITIAL_TICKS), "rounds": 10,
        "window_steps_inclusive": [10, 90], "control_nodes": [s/100 for s in CONTROL_NODES],
        "control_intervals": description(), "checkpoint_steps": list(CHECKPOINT_STEPS),
        "max_atom_step_A": .025, "tick_bounds": [0, 10],
        "agent_objective": {"priority": ["mean_pic50", "best_pic50"], "scope": "teacher_100_chains",
                            "policy": "independent_chain_schedules", "round_barrier": "all_five_batches",
                            "validity": "report eligible counts; never improve mean by dropping failed slots"},
        "llm": {"model": "deepseek-flash", "key_env": "DEEPSEEK_API_KEY", "thinking": True,
                "reasoning_effort": "high", "max_output_tokens": 16384, "input_soft_tokens": 32768,
                "input_hard_tokens": 131072, "concurrency": 4, "timeout": 300, "attempts": 3},
        "versions": ["v1", "v2"], "max_recoveries": 3,
        "evaluation": {"plip": "3.0.1", "posebusters": "dock", "clash_A": 1.2},
        "selection": "valid_connected_finite_pic50; retain incumbent on ties",
        "paired_history_slots": False,
        "retention": {"mode": "top1_per_chain", "restart_parent": "incumbent_only",
                      "preserve_shared_r0": True, "profile_before_prune": True,
                      "cleanup_boundary": "after_round_commit_profile_and_verified_snapshot"},
        "reporting": {"primary": "best_per_chain_mean_pic50", "fixed_round": 10,
                      "unselected_comparison": "fixed_round_mean_pic50",
                      "search_output": "best_per_chain", "shared_baseline_targets": list(TARGETS),
                      "secondary": "best_pic50",
                      "scope": "two selected development teachers from one target; two teacher-origin lineages"},
    }


def eligible(record):
    v = record.get("pic50_on_rescore")
    return (record.get("build_success") is True and record.get("connected") is True
            and not record.get("failure_reason") and isinstance(v, (int, float)) and math.isfinite(v))


def better(new, old):
    return eligible(new) and (not eligible(old) or new["pic50_on_rescore"] > old["pic50_on_rescore"])
