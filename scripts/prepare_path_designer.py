"""Bind one sequential Designer call to the literal Analyst response and contract."""
import argparse,json
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest

def prepare(repo,folder):
    root,out=Path(repo).resolve(),Path(folder)
    request=read_json(out/'Analyst.request.json');response=read_json(out/'Analyst.response.json')
    for key in ['input_sha256','instruction_sha256']:
        if response[key]!=request[key]:raise ValueError('Analyst request binding mismatch')
    packet=read_json(out/'Analyst.input.json')
    inp={'schema_version':'path-designer-input-1.0','analyst':response,'analyst_sha256':digest(out/'Analyst.response.json'),
        'evidence':packet['evidence'],'execution_contract':packet['next_test'],'incumbent_program':packet['incumbent_program'],
        'candidate_reference_sha256':digest(root/packet['next_test']['candidate_reference'])}
    write_json(out/'Designer.input.json',inp)
    skill=root/'skills/designer/SKILL.md'
    text=skill.read_text(encoding='utf-8')+'\n\nTask: Design only the proposed teacher-source experiment, not later modules. Keep the incumbent formula and every controller parameter unchanged. Only reference_sha256 and target_definition may change. Use the supplied candidate reference hash. Return JSON schema_version=path-designer-1.0, role=Designer, input_sha256, instruction_sha256, decision (test_source or defer), updates, evidence_ids, justification, validation_hypotheses, failure_modes. Treat this as a falsifiable exploration, not a deployment claim or proof of benefit. No private reasoning transcript.\n'
    (out/'Designer.instructions.md').write_text(text,encoding='utf-8',newline='\n')
    req={'role':'Designer','input_sha256':digest(out/'Designer.input.json'),'instruction_sha256':digest(out/'Designer.instructions.md'),
        'skill_sha256':digest(skill),'analyst_sha256':digest(out/'Analyst.response.json'),'transport':request['transport']}
    write_json(out/'Designer.request.json',req);return req
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--folder',required=True)
    a=p.parse_args();print(prepare(a.repo,a.folder))
