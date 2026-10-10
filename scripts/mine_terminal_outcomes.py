"""Deterministic input/output interface for decoded final-outcome mining."""
import argparse
from evomolsteer.continuous.terminal_outcome import build

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['dataset', 'campaign', 'metrics', 'baseline', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--score-start', type=float, default=0.)
    p.add_argument('--score-end', type=float, default=.5)
    p.add_argument('--mode', choices=['mean', 'distribution', 'p75', 'instantaneous', 'hierarchical'], default='mean')
    p.add_argument('--shrinkage', type=float, default=2.)
    p.add_argument('--budget', type=int, default=2)
    p.add_argument('--threshold', type=float, default=8.258901977539063)
    p.add_argument('--tail-weight', type=float, default=.5)
    p.add_argument('--branch-mode', choices=['instantaneous', 'outcome_matched'], default='instantaneous')
    p.add_argument('--score-tolerance', type=float, default=.25)
    a = p.parse_args()
    r = build(a.dataset, a.campaign, a.metrics, a.baseline, a.output, [a.score_start, a.score_end],
              a.mode, a.budget, a.threshold, a.tail_weight, a.branch_mode, a.score_tolerance, a.shrinkage)
    print({'score_window': r['score_window'], 'control_window': r['window'], 'labels': r['label_semantics']})
