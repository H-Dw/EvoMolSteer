"""Deploy only a pushed fixed commit, run CPU checks, and prepare six fresh jobs."""

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def command(args, cwd=None, timeout=120):
    result = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(
            repr(args[:3]) + ": " + result.stderr[-2000:] + result.stdout[-2000:]
        )
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    short = args.commit[:8]
    audit = Path("/workspace/uniguide_audits") / ("comparison8-" + short)
    repo = Path("/workspace/uniguide_runs") / ("guidance-comparison8-" + short)
    suite = Path("/workspace/agenticgen_guidance_comparison8_20261031")
    source = Path("/workspace/uniguide_runs/absolute-ug04serial-48de3e9f")
    if repo.exists() or suite.exists():
        raise RuntimeError("Fresh deployment/root already exists; do not overwrite")
    command(["git", "clone", "--shared", "--no-checkout", str(source), str(repo)])
    transports = []
    for prefix in (
        "",
        "https://ghfast.top/",
        "https://gh-proxy.com/",
        "https://ghproxy.net/",
        "https://gh.llkk.cc/",
    ):
        try:
            result = subprocess.run(
                [
                    "git",
                    "fetch",
                    prefix + "https://github.com/H-Dw/EvoMolSteer.git",
                    args.commit,
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=8,
            )
            transports.append(
                dict(transport=prefix or "direct", code=result.returncode)
            )
            if not result.returncode:
                break
        except subprocess.TimeoutExpired:
            transports.append(dict(transport=prefix or "direct", timeout=8))
    else:
        bundle = audit / "pushed.bundle"
        if sha(bundle) != args.bundle_sha256:
            raise ValueError("Pushed bundle checksum differs")
        command(["git", "bundle", "verify", str(bundle)], cwd=repo)
        command(
            ["git", "fetch", str(bundle), "refs/heads/codex/uniguide-flowr-agent"],
            cwd=repo,
        )
        transports.append(
            dict(transport="verified_pushed_bundle", sha256=args.bundle_sha256)
        )
    command(["git", "sparse-checkout", "init", "--cone"], cwd=repo)
    directories = [
        "agenticgen/src",
        "agenticgen/tests",
        "agenticgen/scripts",
        "agenticgen/configs",
        "src",
        "agenticgen/docs/experiments/20261031_guidance_comparison8",
        "agenticgen/docs/experiments/20261030_adaptive_calibration",
        "agenticgen/docs/experiments/20261030_uncapped_coordinate",
        "agenticgen/docs/experiments/20261029_template_distance_formal",
    ]
    command(["git", "sparse-checkout", "set", *directories], cwd=repo)
    command(
        ["git", "checkout", "-B", "codex/uniguide-flowr-agent", args.commit], cwd=repo
    )
    if command(["git", "rev-parse", "HEAD"], cwd=repo).strip() != args.commit:
        raise ValueError("Fixed checkout differs")
    write(
        audit / "DEPLOYMENT.json",
        dict(
            timestamp=now(), commit=args.commit, repo=str(repo), transports=transports
        ),
    )
    python = "/opt/miniforge3/envs/molsteer-flowr-dtk/bin/python"
    env = (
        "source /opt/MolSteer/scripts/scnet/activate_dtk.sh >/dev/null 2>&1\n"
        f"export PYTHONPATH={repo}/agenticgen/src:{repo}/src:/workspace/uniguide_deps:/opt/MolSteer/flowr_root\n"
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n"
        "export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=\n"
        "unset DEEPSEEK_API_KEY\n"
    )
    names = (
        "comparison8_contracts",
        "dynamic_matching_jump",
        "linear_guidance",
        "adaptive_calibration",
        "geometry_tensor_contract",
        "geometry_chemical_state",
        "uniguide_geometry",
        "uniguide_geometry_batch",
        "uniguide_geometry_dynamics",
        "guidance_skill_bundle",
        "guidance_skill_release_v2",
        "guidance_analysis_tools",
        "deepseek_tools",
        "coordinate_statistics",
        "absolute_uncapped_statistics",
        "absolute_coordinate_protocol",
        "ug04_task_scheduling",
    )
    tests = [
        python,
        "-m",
        "pytest",
        "-q",
        "-c",
        "agenticgen/pyproject.toml",
        *("agenticgen/tests/unit/test_" + name + ".py" for name in names),
        "--basetemp=" + str(audit / "pytest"),
        "--junitxml=" + str(audit / "regression.xml"),
    ]
    with (audit / "regression.log").open("wb") as log:
        result = subprocess.run(
            ["bash", "-c", env + shlex.join(tests)],
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    element = ET.parse(audit / "regression.xml").getroot().find("testsuite")
    regression = dict(
        timestamp=now(),
        commit=args.commit,
        returncode=result.returncode,
        **{k: int(element.get(k)) for k in ("tests", "failures", "errors", "skipped")},
        cpu_only=True,
        scientific_attempts=0,
        llm_calls=0,
    )
    write(audit / "REGRESSION.json", regression)
    if result.returncode:
        raise RuntimeError("Remote CPU regression failed; no scientific jobs launched")
    config = json.loads(
        (repo / "agenticgen/configs/uniguide_absolute_coordinate.json").read_text()
    )
    control = json.loads(
        (repo / "agenticgen/configs/uniguide_linear_control.json").read_text()
    )
    previous = json.loads(
        Path(
            "/workspace/uniguide_runs/absolute-ug04serial-48de3e9f-config.json"
        ).read_text()
    )
    config.update(
        control,
        rounds=8,
        conditions=previous["conditions"],
        plip_dependencies=previous["plip_dependencies"],
        checkpoint_learning=dict(enabled=True),
    )
    config["agent_tools"]["skill_release"] = "2.8.0"
    for name in ("flowr/models/mol_builder.py", "flowr/util/functional.py"):
        config["native_hashes"][name] = sha(Path(config["flowr"]) / name)
    windows = dict(
        baseline=[[10, 25], [26, 50], [51, 75], [76, 99]],
        prefix_joint=[[10, 25], [26, 50], [51, 75], [76, 99]],
        five_windows=[[10, 20], [21, 30], [31, 40], [41, 50], [51, 99]],
    )
    jobs = []
    for experiment in ("baseline", "prefix_joint", "five_windows"):
        for condition in config["conditions"]:
            target = condition["teacher_id"]
            job_id = f"{len(jobs)+1:02d}_{experiment}_{target[:4]}"
            local = copy.deepcopy(config)
            local.update(
                root=str(suite / "jobs" / job_id / "attempt_001"),
                conditions=[condition],
                windows=windows[experiment],
                prefix_inpainting=dict(enabled=experiment == "prefix_joint"),
            )
            path = Path(str(repo) + "-configs") / (job_id + ".json")
            write(path, local)
            jobs.append(
                dict(
                    id=job_id,
                    experiment=experiment,
                    teacher=target,
                    config=str(path),
                    config_sha256=sha(path),
                    commit=args.commit,
                    root=local["root"],
                )
            )
    protected = json.loads(
        Path(
            "/workspace/agenticgen_absolute_coordinate_ug04serial_20261030/SOURCE_PRESERVATION.json"
        ).read_text()
    )
    for path in Path(
        "/workspace/agenticgen_absolute_coordinate_ug04serial_20261030"
    ).rglob("*"):
        if (
            path.is_file()
            and "monitoring" not in path.parts
            and not path.name.endswith(".lock")
        ):
            protected[str(path)] = sha(path)
    for condition in config["conditions"]:
        protected[condition["features_path"]] = sha(condition["features_path"])
    for path, digest in protected.items():
        if sha(path) != digest:
            raise ValueError("Historical protected file changed: " + path)
    suite.mkdir(parents=True, exist_ok=False)
    write(suite / "SOURCE_PRESERVATION.json", protected)
    plan = dict(
        timestamp=now(),
        root=str(suite),
        repo=str(repo),
        commit=args.commit,
        generation_context_root=config["generation_context_root"],
        jobs=jobs,
        plan_document_sha256=sha(
            repo
            / "agenticgen/docs/experiments/20261031_guidance_comparison8/plan.zh-CN.md"
        ),
        fresh_calibration=True,
        maximum_calibration_attempts=2100,
        maximum_formal_attempts=7200,
        failure_policy="preserve_skip_continue; audited engineering repairs use new fixed commits and fresh attempts",
    )
    write(suite / "PLAN.json", plan)
    write(
        audit / "READY.json",
        dict(
            timestamp=now(),
            plan=str(suite / "PLAN.json"),
            plan_sha256=sha(suite / "PLAN.json"),
            commit=args.commit,
            protected_files=len(protected),
            scientific_attempts=0,
            llm_calls=0,
        ),
    )
    print(
        json.dumps(
            dict(
                ready=True,
                audit=str(audit),
                plan=str(suite / "PLAN.json"),
                jobs=len(jobs),
            ),
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
