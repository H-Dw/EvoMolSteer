"""Mine observed lag-two coordinate/score mutations without expanding a cache."""
import argparse
import json
from evomolsteer.continuous.branch_mutation import mine

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--augmented-reference',help='Optional full teacher reference; budgeted separately from compact statistics')
    a=p.parse_args();r=mine(a.dataset,a.campaign,a.reference,a.output,a.seed,a.augmented_reference)
    print(json.dumps({'schema_version':r['schema_version'],'independent_batches':r['independent_batches'],
        'feature_count':r['feature_count'],'maximum_parent_transport_error_A':r['maximum_parent_transport_error_A']},sort_keys=True))

if __name__=='__main__':main()
