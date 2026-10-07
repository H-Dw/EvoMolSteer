"""Rebuild compact campaign evidence using retained reports only."""
import argparse
from evomolsteer.generation.path_campaign_report import build_campaign_report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', required=True)
    parser.add_argument('--campaign-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steer-metrics')
    parser.add_argument('--donor-batches', help='Comma-separated predeclared donor batch IDs')
    args = parser.parse_args()
    build_campaign_report(args.reports, args.campaign_config, args.output, args.steer_metrics,
                          [int(v) for v in args.donor_batches.split(',')] if args.donor_batches else None)
