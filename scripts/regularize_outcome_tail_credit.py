"""Convert an outcome reference to count-regularized tail priors."""
import argparse
from evomolsteer.continuous.outcome_tail_regularization import build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('evidence', 'metrics', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--shrinkage', type=float, default=2.)
    args = parser.parse_args()
    build(args.evidence, args.metrics, args.output, args.shrinkage)
