"""Freeze literal generic Skills and compact path evidence for an Analyst call."""
import argparse,json
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest

def prepare(repo,output):
    repo,out=Path(repo).resolve(),Path(output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    docs=repo/'docs/experiments/elite_path20_20261007'
    obs=pd.read_csv(repo/'results/elite_path20/round02_credit/observability.csv')
    packet={'schema_version':'path-evidence-packet-1.0','objective':'Final predicted target affinity and valid high-score paths; strain secondary',
        'window':read_json(repo/'results/elite_path20/round02_credit/manifest.json')['window'],
        'coordinate_representation':'FLOWR endpoint forecasts in receptor world coordinates, Angstrom',
        'evidence':{'E_graph':read_json(docs/'round01_graph_manifest.json'),
            'E_credit':read_json(docs/'round02_credit_manifest.json'),
            'E_library':read_json(docs/'round03_library_manifest.json'),
            'E_observability':obs.to_dict('records'),
            'E_control':read_json(docs/'round04/comparison.json')},
        'incumbent_program':read_json(repo/'configs/experiments/elite_path20_v1/round04.json'),
        'next_test':{'module':'terminal_path_teacher_source','only_change':'Teacher source/labels/identity; executable formula and all control parameters fixed',
            'candidate_reference':'configs/experiments/elite_path20_v1/terminal_path_reference.json.gz',
            'registry':['endpoint_pointcloud'],'batches':[30],'n':50,'master_seed':42},
        'limitations':['Observed futures under continued Steer, not native continuation utility',
            'No new region-level terminal enrichment computed yet','Frozen screening batch selection is adaptive, not independent validation']}
    write_json(out/'Analyst.input.json',packet)
    skill=repo/'skills/analyst/SKILL.md'
    instructions=skill.read_text(encoding='utf-8')+'\n\nTask: Analyze only the next teacher-source module. Do not optimize a later module or invent spatial enrichment. Cite evidence IDs. Discuss censoring, clone dependence, early/late identifiability, terminal versus online labels, and a testable hypothesis. Return JSON with schema_version=path-analyst-1.0, role=Analyst, input_sha256, instruction_sha256, findings (objects with claim and evidence_ids), limitations, recommendation (test or defer), and next_module. No private reasoning transcript.\n'
    (out/'Analyst.instructions.md').write_text(instructions,encoding='utf-8',newline='\n')
    request={'role':'Analyst','input_sha256':digest(out/'Analyst.input.json'),'instruction_sha256':digest(out/'Analyst.instructions.md'),
        'skill_sha256':digest(skill),'transport':'Subagent simulation explicitly requested by user; not a hosted API benchmark'}
    write_json(out/'Analyst.request.json',request);return request
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--output',required=True)
    a=p.parse_args();print(prepare(a.repo,a.output))
