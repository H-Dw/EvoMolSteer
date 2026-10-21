"""Run the persistent supervisor, then drain late profiling work and render reports."""
import argparse
from pathlib import Path

from evomolsteer.adaptive_schedule.background import main
from evomolsteer.adaptive_schedule.io import DEFAULT_ROOT, read, write
from evomolsteer.adaptive_schedule.quality import profile
from evomolsteer.adaptive_schedule.analysis import review_pack
from analyze_plip_adaptive import report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skill-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    options = parser.parse_args()
    main()
    if options.preflight_only:
        raise SystemExit(0)
    # Histories can advance between a watcher scan and its completion check.
    # A final, non-watching drain ensures the last committed round is profiled.
    profile(options.root, options.skill_version, watch=False)
    rendered = report(options.root, options.skill_version)
    review = review_pack(options.root, options.skill_version)
    complete = options.root / "versions" / options.skill_version / "COMPLETE.json"
    if complete.exists():
        write(options.root / f"FINALIZED_{options.skill_version}.json",
              {"cohort": read(complete), "report": rendered, "review_decisions": review["decisions"]})
