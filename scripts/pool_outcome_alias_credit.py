import argparse
from evomolsteer.continuous.outcome_alias_credit import pool

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('dataset', 'campaign', 'labels', 'metrics', 'evidence', 'output'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--tail-weight', type=float, default=0.)
    a = p.parse_args()
    pool(a.dataset, a.campaign, a.labels, a.metrics, a.evidence, a.output, a.tail_weight)
