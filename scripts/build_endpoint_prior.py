import argparse
from evomolsteer.continuous.endpoint_prior import build_prior

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('reference','effects','functions','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--q-max',type=float,default=.05);p.add_argument('--min-effect',type=float,default=.1)
    p.add_argument('--min-batch-fraction',type=float,default=12/14)
    a=p.parse_args();r=build_prior(a.reference,a.effects,a.functions,a.output,a.q_max,a.min_effect,a.min_batch_fraction)
    print({'selected_count':r['selected_count'],'independent_batches':r['n_independent_batches']})
