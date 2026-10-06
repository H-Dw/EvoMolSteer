import argparse
from evomolsteer.continuous.endpoint_kinematics import mine

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','dataset','campaign','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=mine(a.reference,a.dataset,a.campaign,a.output)
    print({'independent_batches':result['independent_batches'],'nodes':len(result['times']),'feature_count':len(result['features'])})
