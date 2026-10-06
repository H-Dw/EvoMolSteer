import argparse
from evomolsteer.continuous.affinity_geometry import mine
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batches',default='0,1,2,3,4,5,6,7,8,9,10,11,12,13');p.add_argument('--window',default='0,.5');p.add_argument('--teachers-per-batch',type=int,default=2)
    a=p.parse_args();mine(a.dataset,a.campaign,a.output,list(map(int,a.batches.split(','))),tuple(map(float,a.window.split(','))),a.teachers_per_batch)
