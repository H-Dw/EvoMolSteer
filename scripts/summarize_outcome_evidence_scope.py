"""Compact scope-explicit evidence interface for local Analyst/reporting use."""
import argparse
from evomolsteer.continuous.outcome_evidence_scope import PREFIXES, summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--scope', choices=PREFIXES, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.evidence, args.scope, args.output)
    print({k: result[k] for k in ('scope', 'feature_count', 'minimum_q', 'corrected_significant_n')})
