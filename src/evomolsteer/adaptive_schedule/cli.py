import argparse
import json
from pathlib import Path

from .io import DEFAULT_ROOT, read


def main():
    p = argparse.ArgumentParser(description="Frozen-reward, checkpoint-resumed PLIP schedule study")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    subs = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "validate", "status"]:
        subs.add_parser(name)
    for name in ["run", "analyze", "review-pack", "profile"]:
        s = subs.add_parser(name)
        s.add_argument("--skill-version", choices=["v1", "v2"], default="v1")
        if name == "profile":
            s.add_argument("--watch", action="store_true")
        if name == "run":
            s.add_argument("--skill", type=Path)
            s.add_argument("--teacher")
            s.add_argument("--batch", type=int, choices=range(5))
            s.add_argument("--resume", action="store_true", help="Existing immutable successful work is always reused")
    args = p.parse_args()
    if args.command == "prepare":
        from .runner import prepare
        result = prepare(args.root)
    elif args.command == "validate":
        from .validation import validate
        result = validate(args.root)
    elif args.command == "run":
        from .runner import run
        result = run(args.root, args.skill_version, args.skill, args.teacher, args.batch)
    elif args.command == "analyze":
        from .analysis import analyze
        result = analyze(args.root, args.skill_version)
    elif args.command == "review-pack":
        from .analysis import review_pack
        result = review_pack(args.root, args.skill_version)
    elif args.command == "profile":
        from .quality import profile
        result = profile(args.root, args.skill_version, args.watch)
    else:
        result = read(args.root / "status.json") if (args.root / "status.json").exists() else {"phase": "not_started"}
        result["completed_batches"] = {
            v: len(list((args.root / "versions" / v).glob("*/batch_*/COMPLETE.json"))) for v in ["v1", "v2"]}
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
