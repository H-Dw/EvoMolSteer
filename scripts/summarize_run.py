import argparse
from evomolsteer.reporting import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(summarize(p.parse_args().output))
