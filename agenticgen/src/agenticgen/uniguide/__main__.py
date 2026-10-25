"""Run with python -m agenticgen.uniguide. Every command uses the explicit UG config."""
import argparse
import json
import sys
from pathlib import Path
from .io import read,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    subs=parser.add_subparsers(dest="command",required=True)
    for name in ["prepare","acceptance","calibrate","init-strategy","status"]: subs.add_parser(name)
    run=subs.add_parser("run-version");run.add_argument("version")
    review=subs.add_parser("accept-review");review.add_argument("version");review.add_argument("review_path")
    args=parser.parse_args();config=read(args.config)
    # A second launcher must not mutate budgets, checkpoints or provider caches.
    if args.command!="status":
        import fcntl
        Path(config["root"]).mkdir(parents=True,exist_ok=True)
        run_lock=open(Path(config["root"])/".executor.lock","a+")
        fcntl.flock(run_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,config["plip_dependencies"])
    if args.command=="prepare":
        from .prepare import prepare_teachers,prepare_initials
        prepare_teachers(config);prepare_initials(config)
    elif args.command=="init-strategy":
        from .pipeline import init_strategy
        init_strategy(config)
    elif args.command=="acceptance":
        from .acceptance import run
        run(config)
    elif args.command=="calibrate":
        from .calibration import run
        from .pipeline import source_binding
        acceptance=read(Path(config["root"])/"acceptance/COMPLETE.json")
        if acceptance["source_binding"]!=source_binding(): raise ValueError("Remote acceptance is stale")
        run(config)
    elif args.command=="run-version":
        from .pipeline import run_version
        print(json.dumps(run_version(config,args.version),ensure_ascii=False,indent=2))
    elif args.command=="accept-review":
        from .review import accept_review
        from .pipeline import skill_location
        print(json.dumps(accept_review(config,args.version,args.review_path,skill_location()),ensure_ascii=False))
    else:
        root=Path(config["root"])
        print(json.dumps({"teachers":(root/"TEACHERS_PREPARED.json").exists(),"initials":(root/"INITIALS_VERIFIED.json").exists(),
            "acceptance":(root/"acceptance/COMPLETE.json").exists(),"calibrated":(root/"calibration/domain.json").exists(),
            "finalized":[p.parent.name for p in sorted((root/"versions").glob("UG*/FINALIZED.json"))],
            "awaiting_review":[p.parent.name for p in sorted((root/"versions").glob("UG*/AWAITING_GPT_REVIEW.json")) if not (p.parent/"REVIEW_ACCEPTED.json").exists()]},indent=2))


if __name__=="__main__": main()
