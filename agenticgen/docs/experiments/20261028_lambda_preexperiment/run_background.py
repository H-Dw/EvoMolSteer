"""One-shot supervisor for two authorized, independent lambda calibrations."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def validate(manifest):
    checkout=Path(manifest["checkout"])
    commit=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(checkout),"status","--porcelain"],text=True)
    if commit!=manifest["implementation_commit"] or dirty:
        raise ValueError("Fixed execution checkout differs")
    for path,digest in manifest["protected_sources"].items():
        if sha(path)!=digest:raise ValueError("Protected source changed: "+path)
    for job in manifest["teachers"]:
        if sha(job["config"])!=job["config_sha256"]:raise ValueError("Frozen configuration changed")
        config=read(job["config"])
        if len(config["conditions"])!=1 or config["conditions"][0]["teacher_id"]!=job["teacher_id"]:
            raise ValueError("Teacher-specific configuration differs")
        if sha(config["conditions"][0]["features_path"])!=job["features_sha256"]:
            raise ValueError("Frozen feature input changed")
        condition=read(Path(config["root"])/"conditions"/(job["teacher_id"]+".json"))
        if condition["condition_sha256"]!=job["condition_sha256"]:
            raise ValueError("Prepared reference differs from accepted condition")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args();manifest=read(args.manifest);root=Path(manifest["root"])
    validate(manifest)
    if args.validate_only:
        print(json.dumps(dict(time=now(),validated=True,teachers=[j["teacher_id"] for j in manifest["teachers"]])))
        return 0
    with (root/".background.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/"EXECUTION_CLAIM.json").open("x",encoding="utf-8") as stream:
            json.dump(dict(time=now(),pid=os.getpid(),manifest_sha256=sha(args.manifest),supervisor_sha256=sha(__file__)),stream)
        outcomes=[]
        for job in manifest["teachers"]:
            teacher=job["teacher_id"];config=read(job["config"]);target=Path(config["root"])
            result=dict(teacher_id=teacher,started_at=now(),root=str(target),stages=[])
            try:
                validate(manifest)
                for stage in ("acceptance","calibrate"):
                    command=[sys.executable,str(Path(manifest["checkout"])/"agenticgen/scripts/run_uniguide_geometry.py"),
                        "--config",job["config"],stage]
                    log=root/(teacher+"."+stage+".log")
                    with log.open("xb") as stream:
                        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,cwd=manifest["checkout"])
                        write(root/"status.json",dict(time=now(),supervisor_pid=os.getpid(),teacher_id=teacher,
                            stage=stage,child_pid=process.pid,log=str(log),completed_teachers=outcomes))
                        code=process.wait()
                    result["stages"].append(dict(stage=stage,exit_code=code,log=str(log),finished_at=now()))
                    if code:raise RuntimeError(stage+" exited with status "+str(code)+"; no retry")
                domain=read(target/"calibration/domains.json")[teacher]
                result.update(state="complete",calibrated_guidance_multiplier=domain["calibrated_guidance_multiplier"],
                    domain_sha256=domain["domain_sha256"],calibration_key=domain["calibration_key"])
            except Exception as error:
                result.update(state="failed",error=str(error),traceback=traceback.format_exc(),automatic_retry=False)
            ledger=read(target/"attempts.json") if (target/"attempts.json").exists() else {}
            result.update(finished_at=now(),charged_attempts=sum(r["count"] for r in ledger.values()))
            if result["charged_attempts"]>job["maximum_attempts"]:
                result.update(state="failed",error="Attempt ceiling exceeded; retain evidence for audit")
            outcomes.append(result)
            write(root/(teacher+".result.json"),result)
        validate(manifest)
        complete=all(r["state"]=="complete" for r in outcomes)
        summary=dict(time=now(),state="complete" if complete else "finished_with_failures",teachers=outcomes,
            total_charged_attempts=sum(r["charged_attempts"] for r in outcomes),
            original_sources_unchanged=True,no_formal_agent_run=True,no_automatic_retry=True)
        write(root/"SUMMARY.json",summary);write(root/"status.json",summary)
        write(root/("COMPLETE.json" if complete else "FINISHED_WITH_FAILURES.json"),summary)
        return 0 if complete else 2


if __name__=="__main__":
    try:sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
