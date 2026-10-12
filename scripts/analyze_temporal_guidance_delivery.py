"""Audit actual per-node dose and control support without additional inference."""
import argparse

from evomolsteer.io import read_json
from evomolsteer.temporal_guidance.delivery import audit_delivery

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--program", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit_delivery(a.traces, read_json(a.program), a.output, a.steps)
