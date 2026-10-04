"""Independent whole-window analysis entry; reuse a v3 extracted feature cache."""
import argparse
from evomolsteer.continuous.pipeline import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True)
    p.add_argument('--splits',nargs='+',default=['discovery','validation','heldout'])
    a=p.parse_args();run(a.analysis,tuple(a.splits))
