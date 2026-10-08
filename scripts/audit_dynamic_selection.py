"""Compact branching consistency audit using existing event aggregates."""
import argparse
import json
from evomolsteer.continuous.dynamic_selection_audit import audit


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    print(json.dumps(audit(a.events, a.manifest, a.output, a.seed)))
