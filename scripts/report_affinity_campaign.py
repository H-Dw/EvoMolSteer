import argparse
from evomolsteer.generation.affinity_reporting import report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    report(a.evidence,a.output,not a.allow_incomplete)
