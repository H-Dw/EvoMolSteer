"""Create or verify a portable, lossless trajectory input bundle."""
import argparse
import json
from evomolsteer.io import read_json
from evomolsteer.storage.input_bundle import build_input_bundle,verify_input_bundle


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root');p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if a.verify_only:
        print(json.dumps(verify_input_bundle(a.output),indent=2))
    else:
        if not a.root or not a.config:p.error('--root and --config are required to create a bundle')
        result=build_input_bundle(a.root,a.output,read_json(a.config),a.workers)
        print(json.dumps({k:result[k] for k in ['source_trajectory_bytes','package_trajectory_bytes','verified_arrays','complete']},indent=2))
