"""Prepare audited assembly/interface repairs without changing the original suite."""

import argparse
import copy
import json
import shlex
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

from deploy_suite import command, now, sha, write


def fetch(repo, audit, commit, bundle_hash):
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
                    commit,
                ],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=8,
            )
            transports.append(
                dict(transport=prefix or "direct", returncode=result.returncode)
            )
            if result.returncode == 0:
                return transports
        except subprocess.TimeoutExpired:
            transports.append(dict(transport=prefix or "direct", timeout=8))
    bundle = audit / "pushed.bundle"
    if sha(bundle) != bundle_hash:
        raise ValueError("Pushed bundle checksum differs")
    command(["git", "bundle", "verify", str(bundle)], cwd=repo)
    command(
        ["git", "fetch", str(bundle), "refs/heads/codex/uniguide-flowr-agent"], cwd=repo
    )
    return transports + [dict(transport="verified_pushed_bundle", sha256=bundle_hash)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--preservation", required=True)
    args = parser.parse_args()
    original = Path("/workspace/agenticgen_guidance_comparison8_20261031")
    source_plan = json.loads((original / "PLAN.json").read_text())
    repo = Path("/workspace/uniguide_runs") / (
        "guidance-comparison8-repair-" + args.commit[:8]
    )
    audit = Path("/workspace/uniguide_audits") / (
        "comparison8-engineering-repair-" + args.commit[:8]
    )
    suite = Path(str(original) + "_repairs")
    if repo.exists() or suite.exists():
        raise ValueError(
            "Repair requires fresh fixed checkout and suite; do not overwrite"
        )
    command(
        ["git", "clone", "--shared", "--no-checkout", source_plan["repo"], str(repo)]
    )
    transports = fetch(repo, audit, args.commit, args.bundle_sha256)
    command(["git", "sparse-checkout", "init", "--cone"], cwd=repo)
    command(
        [
            "git",
            "sparse-checkout",
            "set",
            "agenticgen/src",
            "agenticgen/tests",
            "agenticgen/scripts",
            "agenticgen/configs",
            "src",
            "agenticgen/docs/experiments/20261031_guidance_comparison8",
        ],
        cwd=repo,
    )
    command(
        ["git", "checkout", "-B", "codex/uniguide-flowr-agent", args.commit], cwd=repo
    )
    if command(["git", "rev-parse", "HEAD"], cwd=repo).strip() != args.commit:
        raise ValueError("Fixed repair commit differs")
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
        "prefix_promotion_recovery",
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
    with (audit / "regression.log").open("wb") as stream:
        result = subprocess.run(
            ["bash", "-c", env + shlex.join(tests)],
            cwd=repo,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    xml = ET.parse(audit / "regression.xml").getroot().find("testsuite")
    write(
        audit / "REGRESSION.json",
        dict(
            timestamp=now(),
            commit=args.commit,
            returncode=result.returncode,
            **{k: int(xml.get(k)) for k in ("tests", "failures", "errors", "skipped")},
            cpu_only=True,
        ),
    )
    if result.returncode:
        raise RuntimeError("CPU regression failed; no repair tasks launched")
    preservation = Path(args.preservation)
    report = json.loads(preservation.read_text())
    # Reconstruct exactly the saved failed views before authorizing any continuation.
    check = [
        python,
        str(
            repo
            / "agenticgen/docs/experiments/20261031_guidance_comparison8/audit_saved_promotions.py"
        ),
        "--source-plan",
        str(original / "PLAN.json"),
        "--preservation",
        str(preservation),
        "--output",
        str(audit / "saved_promotion_checks"),
    ]
    with (audit / "saved_promotion_checks.log").open("wb") as stream:
        result = subprocess.run(
            ["bash", "-c", env + shlex.join(check)],
            cwd=repo,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    if result.returncode:
        raise RuntimeError(
            "Saved promotion verification failed; no repair tasks launched"
        )
    protected = json.loads((original / "SOURCE_PRESERVATION.json").read_text())
    jobs = []
    for job in source_plan["jobs"]:
        if job["id"] not in report["jobs"]:
            continue
        source = Path(job["root"])
        hashes = report["jobs"][job["id"]]["source_files_sha256"]
        relative = {
            str(Path(p).relative_to(source)): digest for p, digest in hashes.items()
        }
        manifest = audit / (job["id"] + "_source_preservation.json")
        write(manifest, relative)
        protected.update(hashes)
        config = json.loads(Path(job["config"]).read_text())
        config.update(
            root=str(suite / "jobs" / job["id"] / "attempt_002"),
            postprocessing_recovery=dict(
                source_root=str(source),
                source_commit=source_plan["commit"],
                source_package=str(
                    Path(source_plan["repo"]) / "agenticgen/src/agenticgen/uniguide"
                ),
                preservation_manifest=str(manifest),
                preservation_sha256=sha(manifest),
            ),
        )
        path = Path(str(repo) + "-configs") / (job["id"] + ".json")
        write(path, config)
        jobs.append(
            dict(
                copy.deepcopy(job),
                root=config["root"],
                config=str(path),
                config_sha256=sha(path),
                commit=args.commit,
                source_root=str(source),
                attempt=2,
                imported_budget_aliases=sum(
                    r["count"] for r in report["jobs"][job["id"]]["budget"].values()
                ),
            )
        )
    for path, digest in protected.items():
        if sha(path) != digest:
            raise ValueError("Protected source changed before queuing: " + path)
    suite.mkdir(parents=True, exist_ok=False)
    write(suite / "SOURCE_PRESERVATION.json", protected)
    plan = dict(
        timestamp=now(),
        root=str(suite),
        repo=str(repo),
        commit=args.commit,
        jobs=jobs,
        generation_context_root=source_plan["generation_context_root"],
        wait_for_global_lock=True,
        original_suite=str(original),
        original_commit=source_plan["commit"],
        maximum_calibration_attempts=350 * len(jobs),
        imported_budget_aliases=sum(job["imported_budget_aliases"] for job in jobs),
        maximum_new_calibration_attempts=sum(
            350 - job["imported_budget_aliases"] for job in jobs
        ),
        maximum_formal_attempts=1200 * len(jobs),
        preserve_original_failures=True,
        baseline_llm_failures_not_retried=True,
        same_seed_slot_measurements_not_independent=True,
    )
    write(suite / "PLAN.json", plan)
    write(
        audit / "READY.json",
        dict(
            timestamp=now(),
            commit=args.commit,
            plan=str(suite / "PLAN.json"),
            plan_sha256=sha(suite / "PLAN.json"),
            protected_files=len(protected),
            new_sampling_attempts=0,
            actual_llm_calls=0,
            original_suite_untouched=True,
        ),
    )
    print(
        json.dumps(
            dict(
                ready=True,
                plan=str(suite / "PLAN.json"),
                audit=str(audit),
                jobs=len(jobs),
            ),
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
