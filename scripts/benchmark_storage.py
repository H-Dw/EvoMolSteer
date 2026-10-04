"""Deterministic read benchmark for the local storage trial."""
import argparse
from evomolsteer.storage.benchmark import benchmark

p = argparse.ArgumentParser()
p.add_argument("--raw-root", required=True)
p.add_argument("--trajectory-trials", required=True)
p.add_argument("--feature-source", required=True)
p.add_argument("--feature-packages", required=True)
p.add_argument("--output", required=True)
p.add_argument("--repeats", type=int, default=3)
a = p.parse_args()
benchmark(a.raw_root, a.trajectory_trials, a.feature_source, a.feature_packages, a.output, a.repeats)
