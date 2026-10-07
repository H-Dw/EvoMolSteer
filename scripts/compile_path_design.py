"""Compile one agent-reviewed teacher-source intervention without scalar drift."""
import argparse,copy,json
from pathlib import Path
from evomolsteer.io import read_json,write_json,digest

def compile_design(folder,parent,output):
    folder=Path(folder);req=read_json(folder/'Designer.request.json');r=read_json(folder/'Designer.response.json')
    inp=read_json(folder/'Designer.input.json')
    for key in ['input_sha256','instruction_sha256']:
        if r[key]!=req[key]:raise ValueError('Designer binding mismatch')
    if r['decision']!='test_source':raise ValueError('Designer deferred this experiment')
    updates=r['updates']
    if set(updates)!={'reference_sha256','target_definition'} or updates['reference_sha256']!=inp['candidate_reference_sha256']:
        raise ValueError('Only frozen teacher-source change permitted')
    if not isinstance(updates['target_definition'],str) or not updates['target_definition']:raise ValueError('Explicit label semantics required')
    p=copy.deepcopy(read_json(parent));p.update(updates)
    p.update(round=5,program_id='elite_path20_r05_terminal_source',objective_profile='sequential_path20',
        derivation={'module':'terminal_path_teacher_source','analyst_sha256':req['analyst_sha256'],
        'designer_sha256':digest(folder/'Designer.response.json'),'designer_request_sha256':digest(folder/'Designer.request.json'),
        'scientific_justification':r['justification'],'validation_hypotheses':r['validation_hypotheses'],
        'one_change':'Replace online-stratified teacher source with decoded terminal paths; formula/dose unchanged'})
    Path(output).write_text(json.dumps(p,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    return p
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['folder','parent','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();compile_design(a.folder,a.parent,a.output)
