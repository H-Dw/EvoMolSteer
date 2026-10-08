"""Rebuild the final compact comparison after all ten rounds are retained."""
import argparse
import json

from evomolsteer.generation.dynamic_campaign_report import build_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reports', required=True)
    parser.add_argument('--configs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steer-metrics')
    args = parser.parse_args()
    result = build_report(args.reports, args.configs, args.output, args.steer_metrics)
    print(json.dumps({'rounds': 10, 'candidate_round': result['candidate_round'],
                      'independent_confirmation': result['selected_vs_incumbent_panel_a']}))
