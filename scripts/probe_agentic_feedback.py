"""Small real-API checks of repaired scientific interpretation and code use."""
import argparse
import inspect
import json
from pathlib import Path

import jsonschema
import numpy as np

from evomolsteer.agentic_steer.panel import candidate_indices
from evomolsteer.agentic_steer.workflow import numeric_initial,obj,arr
from evomolsteer.agentic_steer.rules import evaluate_rule
from evomolsteer.deepseek_api import call_json
from evomolsteer.io import read_json,write_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();repo=Path.cwd();cfg=read_json(a.config);study=repo/cfg['study']
    panel=read_json(study/'panel.json');folder=study/'probes'/'scientific_contract_v2';folder.mkdir(parents=True,exist_ok=True)
    initial=read_json(study/'conditions/C6/initial.execution.json');feedback=read_json(study/'conditions/C6/feedback.execution.json')
    requested=[(x['node_id'],r) for x in initial['directions'] for r in range(cfg['initial_replicates'])]+[(x['node_id'],x['replicate']) for x in feedback['next_trials']]
    records=[read_json(study/'trials'/f'{n}_r{r:02d}.json') for n,r in requested]
    summaries=[]
    for node in sorted({n for n,r in requested}):
        rows=[r for r in records if r['trial']['node_id']==node]
        summaries.append({'source_id':'measured/'+node,'node_id':node,'n':len(rows),
            'mean_pic50_all':float(np.mean([r['pic50'] for r in rows])),
            'qualified':sum(r['qualified_structure'] for r in rows),'elite_hits':sum(r['elite_hit'] for r in rows),
            'chemical_decode_failures':sum(not r['valid_connected'] for r in rows),
            'infrastructure_failures':0,'execution_status':'complete'})
    claims=['Euclidean distance maps opposite signed vectors to the same point',
        'A completed inference that produces a chemically undecodable molecule is a numerical job failure',
        'The candidate features/shape_* table contains current coordinates',
        'Identical selection code proves the program adapted to the feedback']
    payload={'claims_to_check':dict(enumerate(claims)),'numerical_counterexample':{'vectors':[[1.,0.],[-1.,0.]],'euclidean_distance':2.,'distance_after_taking_feature_magnitudes':0.},
        'panel_code':inspect.getsource(candidate_indices),'actual_allocation_code':inspect.getsource(numeric_initial),
        'feature_representation':'features/* is endpoint forecast geometry; only current_Rg_A describes current coordinates',
        'measured_summaries':summaries,'execution_definition':'No worker exceptions in these records. Complete inference followed by decode failure is a chemical failure.',
        'initial_rule':initial['code'],'feedback_rule':feedback['code']}
    skill=repo/'skills/agentic-steer';interface=(skill/'interface/SKILL.md').read_text(encoding='utf-8')
    text={'type':'string','maxLength':1600}
    audit_schema=obj({'checks':arr(obj({'claim_id':{'type':'integer','enum':list(range(4))},'verdict':{'type':'string','enum':['supported','refuted','unknown']},'evidence':text,'selection_consequence':text}),4,4)})
    answer=call_json([{'role':'system','content':interface+'\n'+(skill/'omissions/SKILL.md').read_text(encoding='utf-8')},
        {'role':'user','content':json.dumps(payload,separators=(',',':'))}],audit_schema,folder/'omissions.json',input_store=study/'inputs',max_output_tokens=2000)
    checks={x['claim_id']:x for x in answer['checks']}
    omissions_pass=all(checks[i]['verdict']=='refuted' for i in range(4))
    code_schema=obj({'decision':{'type':'string','enum':['retain','revise']},'rationale':text,
        'counterexample':text,'code':{'type':'string','minLength':1,'maxLength':4500},
        'sources':arr({'type':'string','enum':['current_rule','definitions','feedback']+[s['source_id'] for s in summaries]},1,8)})
    features={k:np.asarray([c['features'][k] for c in panel['candidate_cards']]) for k in panel['candidate_cards'][0]['features']}
    def validate(result):
        try:evaluate_rule(result['code'],features)
        except (ValueError,TypeError,SyntaxError) as error:raise jsonschema.ValidationError(str(error),validator='code_contract',validator_value=str(error),schema={}) from error
    packet={'measured_summaries':summaries,'candidate_cards':[{k:v for k,v in c.items() if k!='standardized_features'} for c in panel['candidate_cards']],
        'current_rule':initial['code'],'feature_representation':payload['feature_representation'],
        'objective':panel['objective'],'task':'Retain or revise the current experiment-ranking rule after actual feedback. Numerical means and failure labels were computed by the host. Use raw feature units. Cite a measured counterexample; do not assume terminal success is known for an untested candidate. This is an interpretation and executable-rule test only.'}
    coded=call_json([{'role':'system','content':interface+'\n'+(skill/'feedback/SKILL.md').read_text(encoding='utf-8')+'\n'+(skill/'coding/SKILL.md').read_text(encoding='utf-8')},
        {'role':'user','content':json.dumps(packet,separators=(',',':'))}],code_schema,folder/'coding.json',input_store=study/'inputs',max_output_tokens=3000,response_validator=validate,max_format_repairs=2)
    old,_=evaluate_rule(initial['code'],features);new,audit=evaluate_rule(coded['code'],features)
    oldrank=[panel['candidate_cards'][int(i)]['node_id'] for i in np.argsort(-old,kind='stable')[:4]]
    newrank=[panel['candidate_cards'][int(i)]['node_id'] for i in np.argsort(-new,kind='stable')[:4]]
    result={'omissions_counterexamples_correct':omissions_pass,'coding_decision':coded['decision'],
        'code_changed':coded['code']!=initial['code'],'numeric_scores_changed':bool(np.max(np.abs(old-new))>1e-8),
        'old_top4':oldrank,'new_top4':newrank,'ranking_changed':oldrank!=newrank,'rule_audit':audit,
        'skill_sha256':{m:digest(skill/m/'SKILL.md') for m in ['interface','omissions','feedback','coding']},
        'scope':'Real API function-response probe. No generation performance claim without native tests.'}
    write_json(folder/'response_check.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
