import argparse
from evomolsteer.continuous.endpoint_regions import mine

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','dataset','campaign','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--radius-A',type=float,default=5.)
    a=p.parse_args();r=mine(a.reference,a.dataset,a.campaign,a.output,a.radius_A)
    print({'supported_regions':r['supported_region_indices'],'supported_fields':len(r['supported_node_functions'])})
