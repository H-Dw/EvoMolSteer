"""Atomic artifacts and frozen experiment configuration (no sampler imports)."""
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

DEFAULT_ROOT = Path("/opt/MolSteer/generated_datasets/plip_schedule_agent_5nn4_5ost_s42_20261021")
HISTORY = Path("/opt/MolSteer/generated_datasets/plip_guidance_5nn4_5ost_s42_20261020")
SEEDS = [42, 100045, 200048, 300051, 400054]
TARGETS = ["5nn4_SC2_A_1017", "5ost_AEZ_A_915"]


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
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
    return {
        "schema": "plip-adaptive-schedule-v1", "ssh": {"host": "ksai.scnet.cn", "port": 10104, "user": "root"},
        "history": str(HISTORY), "flowr": "/root/private_data/MolSteer/flowr_root",
        "targets": TARGETS, "seeds": SEEDS, "batch_size": 20, "batches": 5, "steps": 100,
        "features": "C", "mode": "endpoint", "initial_ticks": [5] * 8, "rounds": 10,
        "window_steps": [10, 90], "max_atom_step_A": .025, "tick_bounds": [0, 10],
        "llm": {"model": "deepseek-flash", "key_env": "DEEPSEEK_API_KEY", "thinking": True,
                "reasoning_effort": "high", "max_output_tokens": 16384, "input_soft_tokens": 32768,
                "input_hard_tokens": 65536, "concurrency": 4, "timeout": 300, "attempts": 3},
        "versions": ["v1", "v2"], "max_recoveries": 3,
        "evaluation": {"plip": "3.0.1", "posebusters": "dock", "clash_A": 1.2},
        "selection": "valid_connected_finite_pic50; retain incumbent on ties",
        "paired_history_slots": False,
    }


def eligible(record):
    v = record.get("pic50_on_rescore")
    return (record.get("build_success") is True and record.get("connected") is True
            and not record.get("failure_reason") and isinstance(v, (int, float)) and math.isfinite(v))


def better(new, old):
    return eligible(new) and (not eligible(old) or new["pic50_on_rescore"] > old["pic50_on_rescore"])
