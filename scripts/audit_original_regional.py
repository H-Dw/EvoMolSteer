import argparse
from evomolsteer.generation.local_execution import original_regional

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['dataset','campaign','reference','output','batches']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    original_regional(a.dataset,a.campaign,a.reference,[int(v) for v in a.batches.split(',')],a.output)
