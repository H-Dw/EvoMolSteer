"""Transport-independent Analyst/Designer: export/import or OpenAI-compatible HTTP."""
import json,os
from pathlib import Path
import httpx
from .io import read_json,write_json,digest
from .contracts import ANALYST,DESIGNER,validate

PROJECT=Path(__file__).resolve().parents[2]

def export_request(analysis,role):
    if role not in ['Analyst','Designer']:raise ValueError('Role required')
    if read_json(Path(analysis)/'config.json').get('time_analysis')=='continuous_window':
        if role!='Analyst':raise ValueError('Continuous curve evidence cannot use the legacy stage-based Designer compiler')
        from .continuous.pipeline import build_evidence
        build_evidence(analysis)
        return read_json(Path(analysis)/'agents/Analyst.continuous.request.json')
    analysis=Path(analysis);dest=analysis/'agents';bundle=read_json(dest/'evidence_bundle.json')
    payload={'evidence_bundle':bundle};schema=ANALYST
    if role=='Designer':
        analyst=read_json(dest/'Analyst.response.json');validate('Analyst',analyst,bundle)
        payload['analyst_document']=analyst;schema=DESIGNER
    from .skill_guidance import render
    guidance=read_json(analysis/'config.json').get('agent_guidance_modules',{}).get(role,[])
    skill=render(role,guidance)
    prompt=(PROJECT/'prompts'/f'{role}.txt').read_text(encoding='utf-8')
    request={'role':role,'bundle_sha256':digest(dest/'evidence_bundle.json'),'response_schema':schema,
             'messages':[{'role':'system','content':skill+'\n\nReturn only JSON matching this schema:\n'+json.dumps(schema)},
                         {'role':'user','content':prompt+'\n\n'+json.dumps(payload,ensure_ascii=False)}]}
    write_json(dest/f'{role}.request.json',request)
    return request

def render_report(data):
    lines=[f'# {data["agent"]} structured result','',f'Dataset: `{data["dataset_id"]}`','']
    if data['agent']=='Analyst':
        lines.extend(['## Observations',''])
        for o in data['observations']:lines += [f'- **{o["observation_id"]}** {o["finding"]} Evidence: {", ".join(o["evidence_ids"])}. Uncertainty: {o["uncertainty"]}','']
        lines+=['## Selection and rejected controls',data['selection_comparison'],'','## Stage dynamics',data['stage_dynamics'],'','## Regional rules','']
        for r in data['rules']:lines += [f'- **{r["rule_id"]} ({r["status"]})** {r["region"]}, t={r["stage_start"]}–{r["stage_end"]}, {r["direction"]}: {r["rationale"]}','']
        lines+=['## Limitations','']+['- '+s for s in data['limitations']]
    else:
        lines += [data['design_rationale'],'',f'Status: **{data["status"]}**',f'Gradient path: `{data["gradient_path"]}`','',
                  '## Program','', '```json',json.dumps(data['program'],ensure_ascii=False,indent=2),'```','','## Failure modes','']
        lines += ['- '+s for s in data['failure_modes']]+['','## Validation plan','']+['- '+s for s in data['validation_plan']]
    return '\n'.join(lines)+'\n'

def import_response(request_path,response_path,analysis):
    if read_json(request_path).get('schema_version')=='literal-skill-ablation-1.0':
        from .continuous.skill_ablation import validate_response,compile_response
        request=read_json(request_path);folder=Path(request_path).parent
        data=validate_response(request_path,response_path,request['bindings']['evidence_path'])
        write_json(folder/f'{request["role"]}.response.json',data)
        if request['role']=='Designer':
            compile_response(PROJECT,request_path,folder/'Designer.response.json',request['bindings']['evidence_path'],folder/'reward_program.json')
        return data
    if read_json(request_path).get('schema_version')=='motif-agent-1.0':
        from .continuous.motif_agents import import_response as motif_import
        return motif_import(request_path,response_path,Path(request_path).parent)
    if read_json(request_path).get('schema_version')=='coordinate-1.0':
        from .continuous.coordinate_agents import import_response as coordinate_import
        return coordinate_import(request_path,response_path,analysis)
    if read_json(request_path).get('schema_version')=='continuous-3.0':
        from .continuous.agent_contract import import_response as continuous_import
        return continuous_import(request_path,response_path,analysis)
    request=read_json(request_path);analysis=Path(analysis);dest=analysis/'agents';bundle_path=dest/'evidence_bundle.json'
    if digest(bundle_path)!=request['bundle_sha256']:raise ValueError('Evidence changed since request export')
    data=read_json(response_path);bundle=read_json(bundle_path);role=request['role']
    analyst=read_json(dest/'Analyst.response.json') if role=='Designer' else None
    validate(role,data,bundle,analyst)
    write_json(dest/f'{role}.response.json',data)
    (dest/f'{role}.report.md').write_text(render_report(data),encoding='utf-8')
    write_json(dest/f'{role}.validation.json',{'schema_valid':True,'semantic_grounding_valid':True,
        'request_sha256':digest(request_path),'response_sha256':digest(response_path),'bundle_sha256':digest(bundle_path)})
    if role=='Designer':write_json(dest/'reward_program.json',data['program'])
    return data

def call_api(request_path,analysis):
    request=read_json(request_path);base=os.environ.get('EVOMOLSTEER_BASE_URL');model=os.environ.get('EVOMOLSTEER_MODEL');key=os.environ.get('EVOMOLSTEER_API_KEY')
    if not all([base,model,key]):raise ValueError('Set EVOMOLSTEER_BASE_URL, EVOMOLSTEER_MODEL and EVOMOLSTEER_API_KEY; no endpoint defaults')
    if request.get('schema_version')=='literal-skill-ablation-1.0':
        from .continuous.skill_ablation import payload
        messages=[{'role':'system','content':request['system_instruction']+'\nResponse schema:\n'+json.dumps(request['response_schema'])},
                  {'role':'user','content':json.dumps(payload(request),ensure_ascii=False)}]
    else:messages=request['messages']
    body={'model':model,'messages':messages,'temperature':0,
          'response_format':{'type':'json_schema','json_schema':{'name':request['role'],'strict':True,'schema':request['response_schema']}}}
    with httpx.Client(timeout=180) as client:
        response=client.post(base.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+key},json=body)
        response.raise_for_status();content=response.json()['choices'][0]['message']['content']
    destination=Path(request_path).parent if request.get('schema_version') in ('motif-agent-1.0','literal-skill-ablation-1.0') else Path(analysis)/'agents'
    data=json.loads(content);path=destination/f'{request["role"]}.raw_api.json';write_json(path,data)
    # No fallback to a weaker schema; errors remain visible.
    return import_response(request_path,path,analysis)
