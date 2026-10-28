"""Explicit isolated entrypoint; no automatic calibration, execution or retry."""
import argparse
import fcntl
import json
from pathlib import Path
from agenticgen.uniguide.io import read
from agenticgen.uniguide import geometry_runtime as runtime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("prepare","acceptance","calibrate"):sub.add_parser(name)
    run=sub.add_parser("run-version");run.add_argument("version")
    review=sub.add_parser("accept-review");review.add_argument("version");review.add_argument("review_path")
    reuse=sub.add_parser("reuse-calibration");reuse.add_argument("--sources",required=True);reuse.add_argument("--initial-scales",required=True)
    initial=sub.add_parser("select-initial");initial.add_argument("--initial-scales",required=True)
    args=parser.parse_args();config=runtime.validate_config(read(args.config))
    root=Path(config["root"]);root.mkdir(parents=True,exist_ok=True)
    with (root/".executor.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=="prepare":result=runtime.prepare(config)
        elif args.command=="acceptance":
            from agenticgen.uniguide.geometry_acceptance import run
            result=run(config)
        elif args.command=="calibrate":result=runtime.calibrate(config)
        elif args.command=="run-version":result=runtime.run_version(config,args.version)
        elif args.command=="reuse-calibration":
            from agenticgen.uniguide.geometry_reuse import reuse
            result=reuse(config,read(args.sources),read(args.initial_scales))
        elif args.command=="select-initial":
            from agenticgen.uniguide.geometry_reuse import select_initial
            result=select_initial(config,read(args.initial_scales))
        else:result=runtime.accept_review(config,args.version,args.review_path)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
