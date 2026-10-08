"""Dataset-to-compact-evidence interface for adaptive regional score contrasts."""
import argparse,json
from evomolsteer.continuous.dynamic_regions import mine

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['dataset','campaign','incumbent-reference','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--batches',default=','.join(map(str,range(14))))
    p.add_argument('--window',nargs=2,type=float)
    p.add_argument('--hard-mix',type=float,default=0.)
    p.add_argument('--margin-fraction',type=float,default=.10)
    p.add_argument('--gap-weight',action='store_true')
    p.add_argument('--maximum-regions',type=int,default=4)
    a=p.parse_args();print(json.dumps(mine(a.dataset,a.campaign,a.incumbent_reference,a.output,
        list(map(int,a.batches.split(','))),a.window,a.hard_mix,a.margin_fraction,a.gap_weight,a.maximum_regions)))
