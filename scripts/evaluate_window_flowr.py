import argparse
import json
from evomolsteer.generation.window_evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--original',required=True);p.add_argument('--original-campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=evaluate(a.dataset,a.campaign,a.original,a.original_campaign,a.output)
    print(json.dumps(r['results'],indent=2))
