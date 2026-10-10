import argparse
from evomolsteer.continuous.outcome_feedback import summarize

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('evidence', 'reports', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args()
    summarize(a.evidence, a.reports.split('|'), a.output)
