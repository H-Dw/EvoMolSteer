import argparse
import json
from evomolsteer.generation.terminal_evaluation import evaluate_terminal

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arms',required=True);p.add_argument('--batches',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();r=evaluate_terminal(a.dataset,a.campaign,a.reference,a.output,a.arms.split(','),[int(v) for v in a.batches.split(',')],a.workers)
    print(json.dumps(r['results'],indent=2))
