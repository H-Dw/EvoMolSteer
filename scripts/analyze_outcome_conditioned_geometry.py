import argparse
from evomolsteer.continuous.outcome_conditioned_geometry import analyze

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('dataset', 'campaign', 'labels', 'evidence', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--score-tolerance', type=float, default=.25)
    a = p.parse_args()
    analyze(a.dataset, a.campaign, a.labels, a.evidence, a.output, a.score_tolerance)
