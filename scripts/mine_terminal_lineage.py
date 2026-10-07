import argparse
from evomolsteer.continuous.terminal_lineage import mine

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('reference', 'dataset', 'campaign', 'output'):
        p.add_argument('--'+field, required=True)
    p.add_argument('--radius-A', type=float, default=5.)
    p.add_argument('--teachers-per-batch', type=int, default=2)
    a = p.parse_args()
    print(mine(a.reference, a.dataset, a.campaign, a.output, a.radius_A, a.teachers_per_batch))
