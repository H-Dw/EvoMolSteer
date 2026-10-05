import argparse
from evomolsteer.generation.local_execution import audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(audit(a.dataset,a.campaign,a.reference,a.output)['batch_results'])
