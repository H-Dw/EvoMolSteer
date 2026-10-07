import argparse
from evomolsteer.continuous.elite_path_credit import build_terminal_credit

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['dataset','graph','terminal-metrics','output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--quantile', type=float, default=.95)
    a = p.parse_args()
    print(build_terminal_credit(a.dataset, a.graph, a.terminal_metrics, a.output, a.quantile))
