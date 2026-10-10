import argparse
from evomolsteer.continuous.outcome_matched_reference import build

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('dataset', 'campaign', 'labels', 'metrics', 'evidence', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--score-tolerance', type=float, default=.25)
    a = p.parse_args()
    build(a.dataset, a.campaign, a.labels, a.metrics, a.evidence, a.output, a.score_tolerance)
