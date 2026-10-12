"""Collect an immutable analysis snapshot without mutating the running study."""
import argparse
from pathlib import Path

from evomolsteer.ablation.collection import collect, compare_snapshots


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--study',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--previous',help='Optional prior snapshot directory; compare added observations')
    args=parser.parse_args()
    manifest=collect(args.repo,args.study,args.output)
    if args.previous:
        repo=Path(args.repo)
        previous=Path(args.previous);output=Path(args.output)
        compare_snapshots(previous if previous.is_absolute() else repo/previous,
                          output if output.is_absolute() else repo/output)
    print({k:manifest[k] for k in ['status','phase','completed_comparisons','skipped_comparisons',
                                 'completed_cohorts','generated_attempts_deduplicated','ongoing_study']})
