"""Standard literal Analyst/Designer workflow over a supplied evidence dataset."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json
from evomolsteer.continuous.skill_ablation import export_request,validate_response,compile_response
from evomolsteer.agents import call_api


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','validate','compile','api'])
    p.add_argument('--role',required=True,choices=['Analyst','Designer']);p.add_argument('--evidence',required=True)
    p.add_argument('--output',required=True);p.add_argument('--analyst');p.add_argument('--response')
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));a=p.parse_args()
    folder=Path(a.output);request=folder/(a.role+'.request.json')
    response=Path(a.response) if a.response else folder/(a.role+'.response.json')
    if a.action=='export':
        variant=dict(id='standard_core',analyst='compact',designer='compact',a=[],d=[])
        analyst=Path(a.analyst) if a.analyst else folder/'Analyst.response.json'
        print(export_request(a.repo,a.evidence,folder,variant,a.role,analyst if a.role=='Designer' else None))
    elif a.action=='validate':
        validate_response(request,response,a.evidence);print({'valid':True,'role':a.role})
    elif a.action=='api':
        call_api(request,folder);print({'response_saved':True,'role':a.role})
    else:
        if a.role!='Designer':raise ValueError('Only Designer returns an executable reward')
        value=compile_response(a.repo,request,response,a.evidence,folder/'reward_program.json')
        print({'compiled':value is not None,'decision':read_json(response)['decision']})
