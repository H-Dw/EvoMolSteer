import argparse
from evomolsteer.continuous.coordinate_agents import export,import_response,compile_design

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mining',required=True);p.add_argument('--action',choices=['export','import','api','compile'],required=True)
    p.add_argument('--role',choices=['Analyst','Designer']);p.add_argument('--request');p.add_argument('--response')
    p.add_argument('--dataset');p.add_argument('--campaign');p.add_argument('--output');p.add_argument('--round',type=int)
    a=p.parse_args()
    if a.action=='export':export(a.mining,a.role)
    elif a.action=='import':import_response(a.request,a.response,a.mining)
    elif a.action=='compile':compile_design(a.mining,a.dataset,a.campaign,a.output,a.round)
    else:
        from evomolsteer.agents import call_api
        call_api(a.request,a.mining)
