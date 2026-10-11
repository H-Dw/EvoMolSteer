"""Create a frozen scientific plan, compact evidence and execution activation."""
import argparse
from pathlib import Path
from evomolsteer.ablation.protocol import create_plan, prepare
from evomolsteer.io import read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default="configs/experiments/m2_scientific_ablation_20261011/runner.json")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    settings = read_json(repo/args.config)
    plan = create_plan(repo)
    activation = prepare(repo, repo/settings["study"])
    print({"plan": str(plan), "study": settings["study"], "status": activation["status"], "modules": 17})
