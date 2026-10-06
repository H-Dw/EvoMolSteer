import argparse
from evomolsteer.generation.head_response import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--window',required=True);p.add_argument('--native-labels');a=p.parse_args()
    print(analyze(a.root,a.output,list(map(float,a.window.split(','))),a.native_labels))
