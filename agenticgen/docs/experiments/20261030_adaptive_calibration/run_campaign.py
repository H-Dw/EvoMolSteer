"""Authorized adaptive calibration and five-round live evaluation campaign."""
from pathlib import Path
import runpy
import sys

from agenticgen.uniguide.io import read
from agenticgen.uniguide.calibration_search import POLICY


def main():
    repo, config_path = map(Path, sys.argv[1:3])
    config = read(config_path)
    if config.get('calibration', {}).get('search_policy') != POLICY:
        raise ValueError('This campaign requires adaptive calibration')
    # The existing wrapper enforces both teachers, chemical-only inpainting,
    # uncapped VJP, five rounds, the API environment and the global execution lock.
    runpy.run_path(str(repo/'agenticgen/docs/experiments/20261030_uncapped_coordinate/run_campaign.py'),
                  run_name='__main__')


if __name__ == '__main__':
    main()
