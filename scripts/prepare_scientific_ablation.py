"""Create a frozen scientific plan, compact evidence and execution activation."""
import argparse
from pathlib import Path
from evomolsteer.ablation.protocol import PLAN, FULL_DROP_PLAN, create_plan, create_full_drop_plan, prepare
from evomolsteer.io import read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default="configs/experiments/m2_full_drop_ablation_20261012/runner.json")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    settings = read_json(repo/args.config)
    plan_name = settings.get("plan",PLAN)
    plan = create_full_drop_plan(repo) if plan_name==FULL_DROP_PLAN else create_plan(repo)
    activation = prepare(repo, repo/settings["study"],plan=plan_name)
    print({"plan": str(plan), "study": settings["study"], "status": activation["status"], "modules": 17})
