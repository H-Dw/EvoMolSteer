"""Deploy only a pushed fixed commit, run CPU checks, and prepare six fresh jobs."""

import argparse
import copy
import hashlib
import json
import shlex
import shutil
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
    parser.add_argument(
        "--storage", default="/root/private_data/uniguide_comparison5_20261031"
    )
    args = parser.parse_args()
    short = args.commit[:8]
    storage = Path(args.storage)
    audit = storage / "audits" / ("comparison5-" + short)
    repo = storage / "runs" / ("guidance-comparison5-" + short)
    suite = Path("/workspace/agenticgen_guidance_comparison5_20261031")
    if shutil.disk_usage(storage).free < (1 << 30):
        raise RuntimeError("Persistent storage has less than 1 GiB free")
    if repo.exists() or suite.exists():
        raise RuntimeError("Fresh deployment/root already exists; do not overwrite")
    repo.mkdir(parents=True, exist_ok=False)
    command(["git", "init", "-q", str(repo)])
    command(
        ["git", "remote", "add", "origin", "https://github.com/H-Dw/EvoMolSteer.git"],
        cwd=repo,
    )
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
        "agenticgen/docs/experiments/20261031_guidance_comparison5",
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
        "guidance_comparison5",
        "comparison8_contracts",
        "prefix_promotion_recovery",
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
            "/workspace/uniguide_runs/absolute-batch50-7670549a-config.json"
        ).read_text()
    )
    config.update(
        control,
        rounds=5,
        conditions=previous["conditions"],
        plip_dependencies=previous["plip_dependencies"],
        checkpoint_learning=dict(enabled=True),
    )
    config.update(
        json.loads((repo / "agenticgen/configs/uniguide_comparison5.json").read_text())
    )
    # Snapshot the small frozen generation context and native source onto NFS.
    original_context = Path(previous["generation_context_root"])
    context = storage / "inputs/context"
    if context.exists():
        raise RuntimeError("Input snapshot already exists; do not overwrite")
    context.mkdir(parents=True)
    copies = {}

    def copy_file(source_path, destination):
        source_path, destination = Path(source_path), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        digest = sha(source_path)
        if sha(destination) != digest:
            raise ValueError("Frozen input copy differs")
        copies[str(source_path)] = dict(destination=str(destination), sha256=digest)

    for name in ("initials", "random_stream_reference", "teachers"):
        for old in (original_context / name).rglob("*"):
            if old.is_file():
                copy_file(old, context / old.relative_to(original_context))
    catalog = json.loads((original_context / "teachers.json").read_text())
    copy_file(
        original_context / "teachers.json", storage / "inputs/ORIGINAL_teachers.json"
    )
    for teacher in catalog:
        for name in ("protein", "sdf", "native_config", "source_sdf"):
            old = Path(teacher[name])
            destination = context / old.relative_to(original_context)
            copy_file(old, destination)
            teacher[name] = str(destination)
        old = Path(teacher["protein"]).parent
        source_input = (
            Path(previous["generation_context_root"])
            / "sources/steer"
            / teacher["target"]
            / "inputs"
        )
        for original in source_input.iterdir():
            if original.is_file():
                copy_file(original, old / original.name)
    write(context / "teachers.json", catalog)
    copy_file(
        original_context / "INITIALS_VERIFIED.json", context / "INITIALS_VERIFIED.json"
    )
    for item in config["conditions"]:
        old = Path(item["features_path"])
        destination = storage / "inputs/conditions" / old.name
        copy_file(old, destination)
        item["features_path"] = str(destination)
    flowr_source = Path(config["flowr"])
    flowr_snapshot = storage / "inputs/flowr_root"
    shutil.copytree(
        flowr_source / "flowr",
        flowr_snapshot / "flowr",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for original in (flowr_source / "flowr").rglob("*"):
        if (
            original.is_file()
            and "__pycache__" not in original.parts
            and original.suffix != ".pyc"
        ):
            destination = flowr_snapshot / original.relative_to(flowr_source)
            if sha(destination) != sha(original):
                raise ValueError("Native source snapshot differs")
            copies[str(original)] = dict(
                destination=str(destination), sha256=sha(original)
            )
    config["flowr"] = str(flowr_snapshot)
    config["generation_context_root"] = str(context)
    write(
        audit / "INPUT_SNAPSHOT.json",
        dict(
            timestamp=now(),
            copies=copies,
            original_context=str(original_context),
            copied_context=str(context),
            checkpoint_sha256=sha(config["checkpoint"]),
        ),
    )
    if sha(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Model checksum differs")
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
                    rounds=local["rounds"],
                    skill_release=local["agent_tools"]["skill_release"],
                )
            )
    # Protect available history; missing post-snapshot runs are not invented.
    protected = {}
    historical_manifest = Path(
        "/workspace/agenticgen_absolute_coordinate_batch50_20261030/SOURCE_PRESERVATION.json"
    )
    unavailable = []
    if historical_manifest.exists():
        for name, digest in json.loads(historical_manifest.read_text()).items():
            if not Path(name).exists():
                unavailable.append(name)
                continue
            if sha(name) != digest:
                raise ValueError("Historical protected file changed: " + name)
            protected[name] = digest
    for name in ("initials", "teachers", "random_stream_reference"):
        for path in (original_context / name).rglob("*"):
            if path.is_file():
                protected[str(path)] = sha(path)
    protected[str(original_context / "teachers.json")] = sha(
        original_context / "teachers.json"
    )
    for path in (storage / "inputs").rglob("*"):
        if path.is_file():
            protected[str(path)] = sha(path)
    write(
        audit / "HISTORICAL_AVAILABILITY.json",
        dict(
            unavailable_historical_paths=unavailable,
            new_run_imported_attempts=0,
            old_cancelled_suite_not_restarted=True,
        ),
    )
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
            / "agenticgen/docs/experiments/20261031_guidance_comparison5/plan.zh-CN.md"
        ),
        fresh_calibration=True,
        maximum_calibration_attempts=2100,
        maximum_formal_attempts=4500,
        rounds=5,
        explanation_max_characters=3000,
        global_executor_lock=str(original_context / ".executor.lock"),
        persistent_storage=str(storage),
        durable_snapshot_root=str(storage / "result_snapshots"),
        trajectory_storage=str(suite),
        trajectory_storage_is_overlay=True,
        source_preservation_count=len(protected),
        runtime_source_sha256={
            str(p.relative_to(repo)): sha(p)
            for p in (repo / "agenticgen/src/agenticgen/uniguide").rglob("*.py")
        },
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
