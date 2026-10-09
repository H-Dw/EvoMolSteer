"""Compact tail structures and exact Steer genealogy; no generation or mutation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evomolsteer.continuous.affinity_tail import run_tail_analysis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Original dataset root with inputs and results')
    p.add_argument('--campaign', required=True)
    p.add_argument('--metrics', required=True, help='Rescored original single-arm terminal metrics CSV')
    p.add_argument('--analysis-config', required=True)
    p.add_argument('--threshold', type=float, required=True, help='Frozen affinity tail threshold')
    p.add_argument('--output', required=True)
    p.add_argument('--comparison', action='append', default=[], help='NAME=terminal_metrics.csv#ARM; native required for strain budget; #ARM required for multi-arm files')
    args = p.parse_args()
    comparisons = dict(item.split('=', 1) for item in args.comparison)
    tables = run_tail_analysis(args.root, args.campaign, args.metrics, args.analysis_config,
                               args.output, args.threshold, comparisons)
    print({name: len(table) for name, table in tables.items()})


if __name__ == '__main__':
    main()
