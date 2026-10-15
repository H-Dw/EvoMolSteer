import argparse
from evomolsteer.io import read_json
from evomolsteer.nc_guidance.research_report import report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--config',required=True)
    a=p.parse_args();print(report(a.repo,read_json(a.config)))
