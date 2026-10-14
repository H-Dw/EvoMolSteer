import argparse
from pathlib import Path
from evomolsteer.io import read_json
from evomolsteer.partial_search.mining import build
from evomolsteer.partial_search.agents import analyze

parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--repo',default='.')
parser.add_argument('--action',choices=['mine','analyze'],required=True);parser.add_argument('--feedback')
args=parser.parse_args();cfg=read_json(args.config)
if args.action=='mine':print({'regions':len(build(args.repo,cfg)['regions'])})
else:print({'programs':len(analyze(args.repo,cfg,read_json(args.feedback) if args.feedback else {}))})
