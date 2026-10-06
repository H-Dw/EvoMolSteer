import argparse
import json
from evomolsteer.generation.coordinate_backtracking import diagnose,freeze,record,freeze_contrast,freeze_shape,freeze_conditioning

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    d=sub.add_parser('diagnose');d.add_argument('--evidence',required=True);d.add_argument('--output',required=True)
    d.add_argument('--parent',type=int,default=3);d.add_argument('--comparisons',default='4,5')
    f=sub.add_parser('freeze')
    for name in ('campaign','parent-program','reference','output','evidence','name','reason'):f.add_argument('--'+name,required=True)
    f.add_argument('--round',required=True,type=int);f.add_argument('--kind',choices=['exact_replay','single_factor'],default='single_factor');f.add_argument('--changes',default='{}')
    r=sub.add_parser('record');r.add_argument('--campaign',required=True);r.add_argument('--evidence',required=True);r.add_argument('--round',required=True,type=int)
    h=sub.add_parser('freeze-contrast')
    for name in ('campaign','parent-program','parent-reference','new-reference','output','evidence','name','reason','designer-contract'):h.add_argument('--'+name,required=True)
    h.add_argument('--round',required=True,type=int)
    s=sub.add_parser('freeze-shape')
    for name in ('campaign','parent-program','parent-reference','new-reference','output','evidence','name','reason','designer-contract'):s.add_argument('--'+name,required=True)
    s.add_argument('--round',required=True,type=int)
    k=sub.add_parser('freeze-conditioning')
    for name in ('campaign','parent-program','parent-reference','new-reference','output','evidence','name','reason','designer-contract'):k.add_argument('--'+name,required=True)
    k.add_argument('--round',required=True,type=int)
    a=p.parse_args()
    if a.action=='diagnose':diagnose(a.evidence,a.output,a.parent,tuple(int(v) for v in a.comparisons.split(',')))
    elif a.action=='record':
        result=record(a.campaign,a.evidence,a.round)
        print({k:result[k] for k in ('round','shape_improvement_fraction','all_head_change_vs_native','surround_RMS_improvement_fraction')})
    elif a.action=='freeze-contrast':freeze_contrast(a.campaign,a.parent_program,a.parent_reference,a.new_reference,a.output,a.evidence,a.round,a.name,a.reason,a.designer_contract)
    elif a.action=='freeze-shape':freeze_shape(a.campaign,a.parent_program,a.parent_reference,a.new_reference,a.output,a.evidence,a.round,a.name,a.reason,a.designer_contract)
    elif a.action=='freeze-conditioning':freeze_conditioning(a.campaign,a.parent_program,a.parent_reference,a.new_reference,a.output,a.evidence,a.round,a.name,a.reason,a.designer_contract)
    else:freeze(a.campaign,a.parent_program,a.reference,a.output,a.evidence,a.round,a.name,json.loads(a.changes),a.reason,a.kind)
