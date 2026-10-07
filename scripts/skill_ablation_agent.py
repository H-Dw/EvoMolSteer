"""Export, validate and compile literal Analyst/Designer Skill experiments."""
import argparse
from pathlib import Path
from evomolsteer.io import read_json,write_json
from evomolsteer.continuous.skill_ablation import export_request,validate_response,compile_response


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['designer_request','validate','compile'])
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--study',required=True)
    p.add_argument('--folder',required=True);p.add_argument('--analyst');p.add_argument('--output');a=p.parse_args()
    study=Path(a.study);folder=Path(a.folder);evidence=study/'evidence.json'
    if a.action=='designer_request':
        v=read_json(folder/'Analyst.request.json')['variant']
        analyst=Path(a.analyst) if a.analyst else folder/'Analyst.response.json'
        print(export_request(a.repo,evidence,folder,v,'Designer',analyst))
    elif a.action=='validate':
        role='Designer' if (folder/'Designer.response.json').is_file() else 'Analyst'
        result=validate_response(folder/(role+'.request.json'),folder/(role+'.response.json'),evidence)
        print({'valid':True,'role':role,'decision':result.get('decision')})
    else:
        result=compile_response(a.repo,folder/'Designer.request.json',folder/'Designer.response.json',evidence,a.output)
        print({'compiled':result is not None,'output':a.output})
