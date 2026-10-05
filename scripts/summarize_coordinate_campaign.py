import argparse
from evomolsteer.generation.coordinate_campaign_report import summarize

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Summarize verified retained coordinate campaign reports')
    p.add_argument('--campaign', required=True)
    p.add_argument('--evidence', required=True)
    p.add_argument('--original-terminal', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--decision', required=True)
    args = p.parse_args()
    result = summarize(args.campaign, args.evidence, args.original_terminal, args.output, args.decision)
    print({'rounds_completed': result['rounds_completed'], 'output': args.output})
