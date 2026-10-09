"""Build an explicit HiQBind test input collection from the official FLOWR release."""
import argparse
import json
from evomolsteer.generation.hiqbind_inputs import prepare_hiqbind

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--metadata-directory')
    a=p.parse_args()
    proof=prepare_hiqbind(a.archive_cache,a.output,metadata_directory=a.metadata_directory)
    print(json.dumps({key:value for key,value in proof.items() if key!='structures'},indent=2))
