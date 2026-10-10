import argparse
from evomolsteer.continuous.outcome_summary import summarize

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['labels', 'evidence', 'output']:
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    summarize(a.labels, a.evidence, a.output)
