"""Real-API interface check on a frozen Analyst; never starts a generation round."""
import argparse
import json
from pathlib import Path

from evomolsteer.deepseek_api import call_json
from evomolsteer.io import digest, read_json, write_json
from evomolsteer.partial_search.agents import compile_programs, execution_contract, signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='.');p.add_argument('--round',default='R13')
    p.add_argument('--config',required=True);a=p.parse_args()
    repo=Path(a.repo).resolve();master=read_json(repo/a.config)
    cfg=read_json(repo/'configs/experiments/partial_region_search_20261014'/(a.round+'.json'))
    study=repo/master['campaign_study'];folder=repo/cfg['study']/'candidates'/cfg['candidate_id']
    output=study/'interface_verification';request=read_json(folder/'Designer.request.json')
    messages=[]
    for m in request['body']['messages']:
        path=folder/m['content_relative_path']
        if digest(path)!=m['content_sha256']:raise ValueError('Frozen request input changed')
        messages.append({'role':m['role'],'content':path.read_text(encoding='utf-8')})
    skill=(repo/master['designer_skill']).read_text(encoding='utf-8')
    messages[0]['content']=skill;packet=json.loads(messages[1]['content'])
    packet['execution_contract']=execution_contract(cfg)
    packet['reward_registry']='Actual half-state robust point matching ONLY; no final-view loss or mixture'
    packet['task']='Replay the frozen Analyst-selected reward. Describe only enabled execution_contract views, never an inactive Skills branch. This is interface verification, not a new optimization or generation round.'
    messages[1]['content']=json.dumps(packet,separators=(',',':'))
    schema=request['response_schema'];schema['properties']['execution_contract']={'const':execution_contract(cfg)}
    schema['properties']['endpoint_mix']={'const':cfg.get('endpoint_mix',0.)}
    for key in ('execution_contract','endpoint_mix'):
        if key not in schema['required']:schema['required'].append(key)
    analyst=read_json(folder/'Analyst.json');teacher=read_json(folder/'teacher.json')
    def validate(designer):
        compile_programs(cfg,analyst,designer,teacher,digest(folder/'Analyst.json'),'interface-check',digest(folder/'teacher.json'))
    design=call_json(messages,schema,output/'Designer.json',input_store=study/'inputs',
        max_output_tokens=4000,response_validator=validate,max_format_repairs=2)
    programs=compile_programs(cfg,analyst,design,teacher,digest(folder/'Analyst.json'),digest(output/'Designer.json'),digest(folder/'teacher.json'))
    frozen=[read_json(repo/b['program']) for b in read_json(folder/'bindings.json')['programs']]
    unchanged=[signature(x) for x in programs]==[signature(x) for x in frozen]
    if not unchanged:raise ValueError('Interface clarification changed frozen numeric reward')
    result={'verification_kind':'API format and coupling only; outside the 20 optimization rounds',
        'frozen_analyst_round':a.round,'execution_contract':design['execution_contract'],
        'same_numerical_reward_signatures':unchanged,'additional_generation_attempts':0,
        'original_api_decisions_preserved':True,'raw_free_text_requires_scientific_audit':True}
    write_json(output/'result.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
