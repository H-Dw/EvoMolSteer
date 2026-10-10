"""Local background real-API study; remote host executes FLOWR inference only.

Source and declarative programs go through Git before each remote job. Status,
hashes, compact final metrics and functional audits remain locally reproducible.
No screen result automatically deletes a Skill or declares contextual necessity.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import paramiko

from evomolsteer.continuous.m2_skill_api import (
    BASE, REF, prepare, role_call, compile_reward, review_function, effective_signature)
from evomolsteer.io import digest, read_json, write_json
from analyze_dependency_study import metrics
from remote_session import bridge


def shell(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=120)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote exit {code}: " + error[-1800:])
    return output


class Study:
    def __init__(self, repo, config):
        self.repo = repo.resolve()
        self.config = config
        self.study = self.repo / config["study"]
        self.state_path = self.study / "run_status.json"
        self.config_root = self.repo / config["config_output"]
        self.transport = self.repo / "test/m2_skill_api_transport" / self.study.name
        self.transport.mkdir(parents=True, exist_ok=True)
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "status": "starting", "backend": "real_api", "model": "deepseek-flash",
            "started_unix": time.time(), "completed_tasks": [], "program_results": {}, "chains": []}

    def save(self, **values):
        self.state.update(values, updated_unix=time.time())
        write_json(self.state_path, self.state)
        print(json.dumps({k: self.state.get(k) for k in ["status", "phase", "task", "updated_unix"]}), flush=True)

    @contextlib.contextmanager
    def ssh(self, proxy=False):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        for attempt in range(3):
            try:
                client.connect(self.config["host"], port=self.config["port"], username="root",
                    password=os.environ["MOLSTEER_SCNET_PASSWORD"], look_for_keys=False,
                    allow_agent=False, timeout=25, banner_timeout=30, auth_timeout=25)
                break
            except (OSError, paramiko.SSHException):
                if attempt == 2: raise
                time.sleep(5)
        transport = client.get_transport()
        transport.set_keepalive(30)
        if proxy:
            def forwarded(channel, origin, server):
                threading.Thread(target=bridge, args=(channel, "127.0.0.1", self.config["proxy_port"]), daemon=True).start()
            transport.request_port_forward("127.0.0.1", self.config["remote_proxy_port"], handler=forwarded)
        try:
            yield client
        finally:
            client.close()

    def commit(self, message):
        paths = [self.study.relative_to(self.repo).as_posix(), self.config_root.relative_to(self.repo).as_posix()]
        self.config_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "add", "--", *paths], cwd=self.repo, check=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.repo).returncode
        if changed:
            # Refuse to absorb any separately staged changes from another task.
            staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=self.repo, text=True).splitlines()
            if any(not any(p == base or p.startswith(base + "/") for base in paths) for p in staged):
                raise RuntimeError("Unrelated staged files: refusing automatic commit")
            subprocess.run(["git", "commit", "-m", message], cwd=self.repo, check=True)
        subprocess.run(["git", "-c", "http.proxy=http://127.0.0.1:" + str(self.config["proxy_port"]),
                        "push", "origin", "main"], cwd=self.repo, check=True)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def preflight(self):
        activation = read_json(self.study / "activation.json")
        for path, sha in activation["bound_inputs"].items():
            if digest(self.repo / path) != sha:
                raise ValueError("Frozen local input changed: " + path)
        if digest(self.study / "evidence.json") != activation["evidence_sha256"]:
            raise ValueError("Frozen evidence changed")
        work = PurePosixPath(self.config["remote_work"])
        flowr = PurePosixPath(self.config["flowr"])
        if not work.is_relative_to(flowr / "experiments") or not work.name.startswith("evomolsteer_dependency_"):
            raise ValueError("Unsafe remote workspace")
        with self.ssh() as client:
            check = """import pathlib,json
root=pathlib.Path(REPO)/'configs/experiments'
def indices(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if k in {'batches','batch_indices','screen_batches'} and isinstance(v,list):
    for i in v:
     if isinstance(i,int): yield i
   yield from indices(v)
 elif isinstance(x,list):
  for v in x: yield from indices(v)
for p in root.rglob('*.json'):
 try: used=set(indices(json.loads(p.read_text())))
 except (ValueError,UnicodeError): continue
 if used.intersection(BATCHSET): print(p)
""".replace("REPO", repr(self.config["remote_repo"])).replace("BATCHSET", repr(set(activation["screen_batches"])))
            result = shell(client, shlex.join([self.config["remote_python"], "-c", check]))
            # Own new study files are expected after deployment; any historical
            # occurrence requires manual collision assessment before generation.
            unexpected = [p for p in result.splitlines() if self.config["config_output"] not in p
                          and "/m2_skill_api_" not in p]
            if unexpected:
                raise ValueError("Prospective batch collision: " + str(unexpected))
            shell(client, "mkdir -p " + shlex.quote(str(work)))
            shell(client, f"test $(df -Pk {shlex.quote(str(work))} | awk 'NR==2 {{print $4}}') -gt 409600")
        self.save(preflight_passed=True)

    def infer(self, program_path, signature, arms="gradient", lane="screen"):
        key = signature + ":" + arms + ":" + lane
        if key in self.state["program_results"]:
            return self.state["program_results"][key]
        activation = read_json(self.study / "activation.json")
        batches = activation["screen_batches"]
        name = "DSF_" + signature[:12] + "_b" + str(batches[0]) + "_" + ("controls" if "," in arms else "g")
        spec_path = self.config_root / "jobs" / (name + ".json")
        spec = {"name": name, "arms": arms, "batch": 50, "batches": batches,
            "n": 50 * len(batches), "seed": 42, "steps": 100,
            "program": program_path.relative_to(self.repo).as_posix(), "program_sha256": digest(program_path),
            "reference": REF, "reference_sha256": digest(self.repo / REF)}
        write_json(spec_path, spec)
        commit = self.commit("experiment(m2-api): bind " + name + " real API reward")
        self.save(status="running", task=name, code_commit=commit)
        remote_repo, work = self.config["remote_repo"], self.config["remote_work"]
        status_file = work + "/" + name + ".status.json"
        with self.ssh(proxy=True) as client:
            shell(client, f"cd {shlex.quote(remote_repo)} && test -z \"$(git status --porcelain --untracked-files=no)\" && "
                  f"git -c http.proxy=http://127.0.0.1:{self.config['remote_proxy_port']} pull --ff-only origin main && "
                  f"test $(git rev-parse HEAD) = {shlex.quote(commit)}")
            with client.open_sftp() as sftp:
                try:
                    with sftp.open(status_file) as stream: prior = json.load(stream)
                except FileNotFoundError: prior = None
            if prior is None:
                command = self.remote_command(spec_path.relative_to(self.repo).as_posix())
                log = work + "/" + name + ".driver.log"
                shell(client, f"nohup bash -c {shlex.quote(command)} </dev/null >{shlex.quote(log)} 2>&1 & "
                      f"echo $! > {shlex.quote(work + '/' + name + '.pid')}")
        start = time.monotonic()
        while True:
            with self.ssh() as client:
                with client.open_sftp() as sftp:
                    try:
                        with sftp.open(status_file) as stream: remote = json.load(stream)
                    except FileNotFoundError: remote = {"status": "starting"}
                    write_json(self.study / "transport_records" / (name + ".remote.json"), remote)
                    if remote["status"] == "complete":
                        for suffix in [".tar.gz", ".tar.gz.json"]:
                            dest = self.transport / (name + suffix)
                            temporary = dest.with_suffix(dest.suffix + ".partial")
                            sftp.get(work + "/archives/" + dest.name, str(temporary))
                            temporary.replace(dest)
                        break
                    if remote["status"] == "failed":
                        raise RuntimeError("FLOWR failed: " + remote.get("error", name))
                    # Detect a process that exits before the cohort can write status.
                    health = shell(client, f"if test -f {shlex.quote(work + '/' + name + '.pid')}; then "
                        f"kill -0 $(cat {shlex.quote(work + '/' + name + '.pid')}) 2>/dev/null && echo alive || echo exited; fi")
                    if "exited" in health and remote["status"] != "complete":
                        raise RuntimeError("Remote driver exited: inspect " + name + ".driver.log")
            if time.monotonic() - start > 21600:
                raise TimeoutError("Remote inference exceeded six-hour cohort limit")
            time.sleep(20)
        self.save(status="evaluating_local", task=name)
        reports = self.study / "transport_records"
        output = self.repo / "results" / self.study.name / name
        command = [sys.executable, str(self.repo / "scripts/evaluate_dependency_archive.py"),
            "--name", name, "--transport", str(self.transport), "--reports", str(reports),
            "--output", str(output), "--reference", str(self.repo / "configs/experiments/ck2_terminal_seed42_v1/local_reference.json.gz"),
            "--workers", "2"]
        subprocess.run(command, cwd=self.repo, check=True)
        restored = self.transport / ("restored" + name)
        root = restored / "results" / name
        delivery = self.delivery(root)
        write_json(reports / (name + ".delivery.json"), delivery)
        self.forecast_response(root, reports / (name + ".forecast_features.parquet"))
        outcomes = {}
        for arm in arms.split(","):
            frame, summary = metrics(output / "candidate_metrics.csv", arm)
            compact_path = reports / (name + "." + arm + ".metrics.csv")
            columns = [c for c in ["batch", "slot", "pic50_on_rescore", "valid_connected", "pb_fast_pass",
                "energy_status", "mmff_relief_per_heavy", "smiles"] if c in frame]
            frame[columns].to_csv(compact_path, index=False)
            outcomes[arm] = {"summary": summary, "metrics": compact_path.relative_to(self.repo).as_posix()}
        result = {"name": name, "effective_signature": signature, "code_commit": remote["code_commit"],
            "outcomes": outcomes, "delivery": delivery, "archive_sha256": digest(self.transport / (name + ".tar.gz")),
            "initial_state_signatures": remote["initial_state_signatures"]}
        self.state["program_results"][key] = result
        self.state["completed_tasks"].append(name)
        self.save(status="running", task=name)
        # Raw current-experiment data retire only after verified transport,
        # execution audit, compact metrics and report persistence. Original Steer
        # and all older studies are outside these exact owned targets.
        self.cleanup(name, restored)
        self.commit("report(m2-api): verify " + name + " final quality and retire raw data")
        return result

    def remote_command(self, spec):
        c = self.config
        env = ("source /opt/MolSteer/scripts/scnet/activate_dtk.sh\n"
            "export LD_LIBRARY_PATH=/opt/miniforge3/envs/molsteer-flowr-dtk/lib:${LD_LIBRARY_PATH:-}\n"
            f"export PYTHONPATH={shlex.quote(c['remote_repo'] + '/src:' + c['flowr'] + ':' + c['flowr'] + '/experiments/evomolsteer_online_20261004/runtime_deps')}\n"
            "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1\n")
        argv = [c["remote_python"], "-u", c["remote_repo"] + "/scripts/run_dependency_cohort.py",
            "--repo", c["remote_repo"], "--flowr-root", c["flowr"], "--work", c["remote_work"], "--spec", c["remote_repo"] + "/" + spec]
        return "set -euo pipefail\n" + env + "cd " + shlex.quote(c["remote_repo"]) + "\n" + shlex.join(argv)

    @staticmethod
    def delivery(root):
        rows = []
        for folder in sorted(root.glob("*/batch_*")):
            if folder.parent.name != "gradient": continue
            trace = [json.loads(line) for line in (folder / "guidance_trace.jsonl").read_text().splitlines()]
            active = [r for r in trace if r["active"]]
            injection = np.asarray([r["injection_rms_A"] for r in active])
            calibration = np.asarray([r["calibration_rms_A"] for r in active])
            rows.append({"batch": int(folder.name.split("_")[-1]), "controlled_steps": len(active),
                "nonzero_steps": sum(np.any(np.asarray(r["injection_l2_A"]) > 0) for r in active),
                "mean_injection_rms_A": float(np.mean([r["injection_rms_A"] for r in active])),
                "mean_predictive_calibration_rms_A": float(calibration.mean()),
                "mean_delivered_over_calibration": float(np.mean(injection / np.maximum(calibration, 1e-12))),
                "nearest_teacher_rms_A_first": float(np.asarray(active[0]["nearest_standardized_rms"]).mean()),
                "nearest_teacher_rms_A_last": float(np.asarray(active[-1]["nearest_standardized_rms"]).mean()),
                "first_order_positive_fraction": float(np.mean([np.asarray(r["first_order_reward_change"]) > 0 for r in active])),
                "last_controlled_state_time": max(r["state_time"] for r in active),
                "post_window_nonzero": sum(np.any(np.asarray(r["injection_l2_A"]) > 0) for r in trace if not r["active"]),
                "trace_sha256": digest(folder / "guidance_trace.jsonl")})
        return {"batches": rows, "semantics": "Actual coordinate injection; positive scalar response is not an affinity derivative"}

    def forecast_response(self, root, destination):
        """Small full-window batch means; no particle-expanded feature cache."""
        import gzip
        reference = json.loads(gzip.decompress((self.repo / REF).read_bytes()))
        features = reference["features"]
        rows = []
        for folder in sorted(root.glob("gradient/batch_*")):
            trace = [json.loads(line) for line in (folder / "guidance_trace.jsonl").read_text().splitlines()]
            for item in trace:
                if not item["active"]: continue
                values = np.asarray(item["observables"])
                if values.ndim != 2 or values.shape[1] != len(features):
                    raise ValueError("Forecast feature representation mismatch")
                rows.append({"batch": int(folder.name.split("_")[-1]),
                    "score_time": item["score_time"], "state_time": item["state_time"],
                    **dict(zip(features, values.mean(0)))})
        pd.DataFrame(rows).to_parquet(destination, index=False, compression=None)
        write_json(destination.with_suffix(".semantics.json"), {
            "reference_sha256": digest(self.repo / REF), "features": features,
            "representation": "Executed reward endpoint observables before this step's injection",
            "rows": len(rows), "aggregation": "mean over fixed slots in each independent RNG batch and actual control step",
            "numeric_type": "float64", "range": reference["window"],
            "no_stage_binning": True, "limitation": "Batch means cannot reconstruct individual paths; observed evolution includes native flow"})

    def cleanup(self, name, restored):
        if not restored.resolve().is_relative_to(self.transport.resolve()) or restored.name != "restored" + name:
            raise ValueError("Unsafe local retirement target")
        shutil.rmtree(restored)
        for suffix in [".tar.gz", ".tar.gz.json"]:
            (self.transport / (name + suffix)).unlink()
        with self.ssh() as client:
            # Resolve and validate within the new study root on the remote host.
            script = """import pathlib,shutil,json
w=pathlib.Path(WORK).resolve()
name=NAME
targets=[w/'archives'/(name+'.tar.gz'),w/'archives'/(name+'.tar.gz.json'),w/'seed_structures'/name]
removed=[]
for target in targets:
 p=target.resolve()
 if not p.is_relative_to(w) or p==w: raise ValueError('Unsafe retirement')
 if p.is_dir(): shutil.rmtree(p);removed.append(str(p))
 elif p.exists(): p.unlink();removed.append(str(p))
print(json.dumps({'removed':removed}))
""".replace("WORK", repr(self.config["remote_work"])).replace("NAME", repr(name))
            record = json.loads(shell(client, shlex.join([self.config["remote_python"], "-c", script])))
        write_json(self.study / "transport_records" / (name + ".retirement.json"), record)

    def run(self):
        if not (self.study / "activation.json").exists(): prepare(self.repo, self.study, self.config.get("screen_batches"))
        self.preflight()
        activation = read_json(self.study / "activation.json")
        self.save(status="running", phase="P0")
        signature = "m2" + digest(self.repo / BASE)[:29]
        controls = self.infer(self.repo / BASE, signature, "gradient,unguided")
        self.state["controls"] = controls
        # Associate the incumbent's effective signature with its already run
        # gradient arm, so a retain decision never triggers duplicate inference.
        base = read_json(self.repo / BASE)
        base_sha = effective_signature(base)
        self.state["program_results"][base_sha + ":gradient:screen"] = controls
        self.save()
        rows = activation["conditions"]
        # The full structured Analyst pool freezes before any ablation result.
        pstruct = next(r for r in rows if r["condition_id"] == "P_struct")
        ordered = [pstruct] + [r for r in rows if r["intervention_role"] != "none" and r is not pstruct]
        for row in ordered:
            self.save(status="running", phase=row["phase"] + "-" + row["lane"], task=row["condition_id"])
            for chain in range(6):
                identity = row["condition_id"] + ":" + str(chain)
                if any(x["identity"] == identity for x in self.state["chains"]): continue
                if row["phase"] == "P2":
                    analyst = read_json(self.study / "calls/P_struct" / f"chain_{chain:02d}" / "Analyst.json")
                    folder = self.study / "calls" / row["condition_id"] / f"chain_{chain:02d}"
                    write_json(folder / "Analyst.json", analyst)
                    write_json(folder / "Analyst.frozen_input.json", {
                        "source": f"calls/P_struct/chain_{chain:02d}/Analyst.json",
                        "sha256": digest(folder / "Analyst.json"), "new_call": False})
                else:
                    analyst = role_call(self.repo, self.study, row, chain, "Analyst")
                designer = role_call(self.repo, self.study, row, chain, "Designer", analyst,
                                     controls["delivery"] if row["lane"] == "U" else None)
                path, sha = compile_reward(self.repo, self.study, row, chain, designer)
                result = self.infer(path, sha)
                # Across distinct programs require the same initial state for
                # every corresponding RNG batch, not merely the same seed text.
                for batch in activation["screen_batches"]:
                    key = f"gradient/batch_{batch:03d}"
                    if result["initial_state_signatures"][key] != controls["initial_state_signatures"][key]:
                        raise ValueError("Cross-program initial state mismatch")
                self.state["chains"].append({"identity": identity, "condition": row["condition_id"],
                    "chain": chain, "effective_sha256": sha, "result_name": result["name"],
                    "mean_pic50": result["outcomes"]["gradient"]["summary"]["affinity_mean_all"],
                    "metrics": result["outcomes"]["gradient"]["metrics"]})
                self.save(status="running")
            try:
                review_function(self.study, row)
            except Exception as error:
                # A failed semantic audit cannot erase measured generation or
                # masquerade as a module response. Keep the independent screens
                # moving; interpretation/pruning for this condition stays gated.
                write_json(self.study / "reviews" / (row["condition_id"] + ".unconfirmed.json"),
                    {"status": "functional_review_unconfirmed", "error_type": type(error).__name__,
                     "error": str(error), "generation_results_preserved": True,
                     "module_response_and_necessity_claims_allowed": False})
            self.report()
            self.commit("ablation(m2-api): record " + row["condition_id"] + " six-chain screen and functional audit")
        self.save(status="screen_complete", task=None,
            next_phase="Confirmation and crossed semantic/statistical review before pruning or final-context necessity")
        self.report()
        self.commit("report(m2-api): complete ordered Analyst and Designer module screens")

    def report(self):
        conditions = {}
        for chain in self.state["chains"]:
            conditions.setdefault(chain["condition"], []).append(chain)
        values = []
        m2 = self.state["controls"]["outcomes"]["gradient"]["summary"]["affinity_mean_all"]
        native = self.state["controls"]["outcomes"]["unguided"]["summary"]["affinity_mean_all"]
        for condition, chains in conditions.items():
            means = [x["mean_pic50"] for x in chains]
            values.append({"condition": condition, "chains": len(chains), "distinct_rewards": len({x["effective_sha256"] for x in chains}),
                "mean_chain_pic50": float(np.mean(means)), "delta_vs_M2": float(np.mean(means) - m2),
                "delta_vs_native": float(np.mean(means) - native)})
        write_json(self.study / "screen_summary.json", {"conditions": values,
            "M2_mean": m2, "native_mean": native, "backend": "real DeepSeek API",
            "model": "deepseek-flash", "limitations": "Exploratory screen. Identical rewards reuse matched inference. No module removal, necessity or equivalence claim from this table. Original Steer retained as historical unpaired comparator."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    study = Study(Path(args.repo), read_json(args.config))
    try:
        study.run()
    except Exception as error:
        study.save(status="failed", error_type=type(error).__name__, error=str(error))
        raise
