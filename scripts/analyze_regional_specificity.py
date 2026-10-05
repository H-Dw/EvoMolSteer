import argparse
from evomolsteer.continuous.regional_specificity import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mining',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=analyze(a.mining,a.output);print({'tested':len(result),'q_below_05':int(result.q.lt(.05).sum())})
