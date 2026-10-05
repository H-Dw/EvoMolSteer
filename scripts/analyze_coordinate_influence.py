import argparse
from evomolsteer.continuous.coordinate_influence import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mining',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.mining,a.output)
