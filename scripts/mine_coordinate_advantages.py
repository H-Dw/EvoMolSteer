"""CLI dataset -> compact coordinate/affinity evidence on every actual selection time."""
import argparse
from evomolsteer.continuous.coordinate_mining import mine

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batches',help='optional comma-separated engineering subset');p.add_argument('--spatial-width-A',type=float,default=4.)
    a=p.parse_args();mine(a.dataset,a.campaign,a.analysis,a.output,
                         [int(b) for b in a.batches.split(',')] if a.batches else None,a.spatial_width_A)
