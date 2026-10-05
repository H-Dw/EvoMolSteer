import argparse
from evomolsteer.generation.coordinate_evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('dataset','campaign','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(evaluate(a.dataset,a.campaign,a.output)['batch_results'])
