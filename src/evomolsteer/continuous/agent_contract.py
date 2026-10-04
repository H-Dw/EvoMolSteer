"""A continuous-curve Analyst contract distinct from stage/reward contracts."""
from pathlib import Path
import jsonschema
from ..io import read_json,write_json,digest

STATEMENT={'type':'object','additionalProperties':False,
    'properties':{'finding':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string'}},
        'uncertainty':{'type':'string'}},'required':['finding','evidence_ids','uncertainty']}
RULE={'type':'object','additionalProperties':False,'properties':{
    'feature':{'type':'string'},'representation':{'enum':['predicted_endpoint','proposal_state']},
    'model_id':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string'}},
    'interpretation':{'type':'string'},'status':{'enum':['association_hypothesis','deferred']},
    'limitation':{'type':'string'}},
    'required':['feature','representation','model_id','evidence_ids','interpretation','status','limitation']}
SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'schema_version':{'const':'continuous-3.0'},'agent':{'const':'Analyst'},
    'observations':{'type':'array','items':STATEMENT},
    'retained_seed_characteristics':{'type':'array','items':STATEMENT},
    'temporal_rules':{'type':'array','items':RULE},
    'counterevidence':{'type':'array','items':STATEMENT},
    'limitations':{'type':'array','items':{'type':'string'}}},
    'required':['schema_version','agent','observations','retained_seed_characteristics','temporal_rules','counterevidence','limitations']}


def import_response(request_path,response_path,analysis):
    analysis=Path(analysis);dest=analysis/'agents';request=read_json(request_path)
    bundle_path=dest/'continuous_evidence.json'
    if digest(bundle_path)!=request['bundle_sha256']:raise ValueError('Continuous evidence changed since export')
    data=read_json(response_path);jsonschema.validate(data,SCHEMA)
    bundle=read_json(bundle_path);evidence={e['evidence_id']:e for e in bundle['evidence']}
    for item in data['observations']+data['retained_seed_characteristics']+data['counterevidence']+data['temporal_rules']:
        if not item['evidence_ids'] or set(item['evidence_ids'])-set(evidence):
            raise ValueError('Missing or unknown evidence ID')
    for rule in data['temporal_rules']:
        candidates=[evidence[e]['model'] for e in rule['evidence_ids'] if 'model' in evidence[e]]
        if not any(m['model_id']==rule['model_id'] and m['feature']==rule['feature']
                   and m['representation']==rule['representation'] for m in candidates):
            raise ValueError('Rule must identify a supplied frozen feature curve')
    write_json(dest/'Analyst.continuous.response.json',data)
    write_json(dest/'Analyst.continuous.validation.json',{'schema_valid':True,'semantic_grounding_valid':True,
        'request_sha256':digest(request_path),'response_sha256':digest(response_path),'bundle_sha256':digest(bundle_path)})
    lines=['# Analyst: continuous selection-window evidence','']
    for field in ['observations','retained_seed_characteristics','counterevidence']:
        lines+=['## '+field,'']
        for item in data[field]:lines += [item['finding'],f"Evidence: {', '.join(item['evidence_ids'])}",item['uncertainty'],'']
    lines+=['## Temporal hypotheses','']
    for item in data['temporal_rules']:lines += [f"{item['feature']} ({item['status']}): {item['interpretation']}",item['limitation'],'']
    lines+=['## Limitations','']+['- '+s for s in data['limitations']]
    (dest/'Analyst.continuous.report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return data
