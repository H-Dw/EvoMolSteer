import argparse
import json
from evomolsteer.generation.coordinate_backtracking import diagnose,freeze

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    d=sub.add_parser('diagnose');d.add_argument('--evidence',required=True);d.add_argument('--output',required=True)
    f=sub.add_parser('freeze')
    for name in ('campaign','parent-program','reference','output','evidence','name','reason'):f.add_argument('--'+name,required=True)
    f.add_argument('--round',required=True,type=int);f.add_argument('--kind',choices=['exact_replay','single_factor'],default='single_factor');f.add_argument('--changes',default='{}')
    a=p.parse_args()
    if a.action=='diagnose':diagnose(a.evidence,a.output)
    else:freeze(a.campaign,a.parent_program,a.reference,a.output,a.evidence,a.round,a.name,json.loads(a.changes),a.reason,a.kind)
