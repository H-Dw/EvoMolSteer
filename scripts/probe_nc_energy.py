import argparse
from evomolsteer.io import read_json
from evomolsteer.nc_guidance.energy_probe import run,compare_confirmations

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.')
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--input');source.add_argument('--config')
    p.add_argument('--output');a=p.parse_args()
    if a.config:print(compare_confirmations(a.repo,read_json(a.config)).to_string(index=False))
    else:
        if not a.output:p.error('--output is required with --input')
        print(run(a.repo,read_json(a.input),a.output))
