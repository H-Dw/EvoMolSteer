"""Stream motifs through the existing full-window lineage/statistics contract."""
import argparse
from evomolsteer.continuous.coordinate_mining import mine
p=argparse.ArgumentParser()
for key in ('dataset','campaign','analysis','output'):p.add_argument('--'+key,required=True)
p.add_argument('--regions',required=True);p.add_argument('--batches',default=None)
a=p.parse_args()
mine(a.dataset,a.campaign,a.analysis,a.output,
     batches=list(map(int,a.batches.split(','))) if a.batches else None,
     spatial_anchor='endpoint',regions=a.regions.split(','),control_representation='proposal',
     feature_family='motif',control_lag_nuisance=True)
