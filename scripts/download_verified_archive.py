"""Download a published archive in verified, resumable HTTP ranges."""
import argparse
import json
from evomolsteer.storage.download import download_verified

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bytes',type=int,required=True);p.add_argument('--md5',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--proxy')
    a=p.parse_args()
    print(json.dumps(download_verified(a.url,a.output,size=a.bytes,md5=a.md5,workers=a.workers,proxy=a.proxy),indent=2))
