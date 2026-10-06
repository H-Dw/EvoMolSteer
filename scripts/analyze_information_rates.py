import argparse
from evomolsteer.continuous.information_rates import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.input,a.output)
