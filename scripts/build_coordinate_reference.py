import argparse
from evomolsteer.generation.coordinate_reference import build

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('dataset','campaign','mining','output','regions'):p.add_argument('--'+k,required=True)
    p.add_argument('--channel',choices=['all','NOS'],default='all');p.add_argument('--std-floor-A',type=float,default=.15)
    a=p.parse_args();build(a.dataset,a.campaign,a.mining,a.output,a.regions.split(','),a.channel,a.std_floor_A)
