import argparse
from evomolsteer.generation.comparison import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline',default='unguided');a=p.parse_args()
    print(evaluate(a.campaign,a.output,a.baseline).to_string(index=False))
