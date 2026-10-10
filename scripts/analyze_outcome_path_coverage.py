import argparse
from evomolsteer.continuous.outcome_path_coverage import coverage

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('dataset', 'campaign', 'labels', 'metrics', 'evidence', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    coverage(a.dataset, a.campaign, a.labels, a.metrics, a.evidence, a.output)
