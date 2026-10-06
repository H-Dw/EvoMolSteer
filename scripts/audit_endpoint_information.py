import argparse
from evomolsteer.continuous.endpoint_information import audit

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','prior','dataset','campaign','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=audit(a.reference,a.prior,a.dataset,a.campaign,a.output)
    print({'independent_batches':r['independent_batches'],'metrics':r['metrics']})
