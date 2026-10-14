"""Stream original Steer into a compact endpoint-only geometry evidence set."""
import argparse
from evomolsteer.continuous.affinity_endpoint import mine_endpoint

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--output',required=True);p.add_argument('--batches',help='Explicit IDs; omit for observed-batch dynamic discovery split')
    p.add_argument('--window',default='0,.5');p.add_argument('--coordinate-library');a=p.parse_args()
    mine_endpoint(a.dataset,a.campaign,a.output,None if a.batches is None else list(map(int,a.batches.split(','))),tuple(map(float,a.window.split(','))),a.coordinate_library)
