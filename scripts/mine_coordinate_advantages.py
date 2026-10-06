"""CLI dataset -> compact coordinate/affinity evidence on every actual selection time."""
import argparse
from evomolsteer.continuous.coordinate_mining import mine

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batches',help='optional comma-separated engineering subset');p.add_argument('--spatial-width-A',type=float,default=4.)
    p.add_argument('--spatial-anchor',choices=['current','endpoint'],default='current')
    p.add_argument('--control-representation',choices=['current','proposal'],default='current');p.add_argument('--regions')
    p.add_argument('--feature-family',choices=['geometry','transport','joint','shape','composition'],default='geometry')
    p.add_argument('--control-lag-nuisance',action='store_true',help='Optional survivor-parent lag diagnostic controlling global pose/size/endpoint composition; not causal')
    a=p.parse_args();mine(a.dataset,a.campaign,a.analysis,a.output,
                         [int(b) for b in a.batches.split(',')] if a.batches else None,a.spatial_width_A,
                         a.spatial_anchor,a.regions.split(',') if a.regions else None,a.control_representation,a.feature_family,a.control_lag_nuisance)
