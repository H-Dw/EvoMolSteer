"""Fresh full evaluation using the restored UG04 task scheduling contract."""
from pathlib import Path
import runpy
import sys

from agenticgen.uniguide.io import read
from agenticgen.uniguide.task_scheduling import POLICY


def main():
    repo, config_path = map(Path, sys.argv[1:3])
    config = read(config_path)
    if config.get("task_scheduling") != POLICY or "gpu_execution" in config:
        raise ValueError("This evaluation requires the restored UG04 serial scheduler")
    if config.get("sampling_recovery") or config.get("calibration", {}).get("measurement_source_root"):
        raise ValueError("This evaluation requires fresh measurements and calibration")
    # Existing wrappers enforce teacher identities, nine PLIP atoms, five rounds,
    # the global execution lock and the unchanged scientific sampling protocol.
    runpy.run_path(str(repo / "agenticgen/docs/experiments/20261030_adaptive_calibration/run_campaign.py"),
                   run_name="__main__")


if __name__ == "__main__":
    main()
