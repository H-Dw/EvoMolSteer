"""Verify retained inputs and frozen validation; write one compact JSON report."""
import argparse
from evomolsteer.generation.motif_verification import audit
from evomolsteer.generation.prototypes import write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    result = audit(a.evidence, a.config, require_complete=not a.allow_incomplete)
    write_json(a.output, result)
    print(f"Verified {result['rounds_verified']} rounds / {result['attempts']} attempts")
