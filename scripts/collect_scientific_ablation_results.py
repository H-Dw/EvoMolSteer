"""Collect an immutable analysis snapshot without mutating the running study."""
import argparse
from pathlib import Path

from evomolsteer.ablation.collection import collect


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--study',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    manifest=collect(args.repo,args.study,args.output)
    print({k:manifest[k] for k in ['status','phase','completed_comparisons','skipped_comparisons',
                                 'completed_cohorts','generated_attempts_deduplicated','ongoing_study']})
