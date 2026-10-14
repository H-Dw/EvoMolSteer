"""Local, evidence-bound DeepSeek decisions and compact direction archives."""
import argparse
import json
from pathlib import Path

import jsonschema
import numpy as np
from scipy.stats import beta

from evomolsteer.deepseek_api import call_json
from evomolsteer.io import read_json,write_json,digest
from .panel import make_panel,make_job
from .rules import evaluate_rule


def obj(properties,required=None):
    return {'type':'object','additionalProperties':False,'properties':properties,
            'required':list(properties) if required is None else required}


def arr(item,minimum=0,maximum=24):
    return {'type':'array','items':item,'minItems':minimum,'maxItems':maximum}


def schema(panel,cfg,phase):
    text={'type':'string','maxLength':1400}
    node={'type':'string','enum':[c['node_id'] for c in panel['candidate_cards']]}
    fields=list(panel['candidate_cards'][0]['features'])
    sources=[c['source_id'] for c in panel['candidate_cards']]+[c['source_id'] for c in panel['discovery_contrasts']]+[c['source_id'] for c in panel['terminal_paths']]+['pocket_context','definitions','selection_code','feedback']
    refs=arr({'type':'string','enum':sources},1,8)
    return obj({'hypotheses':arr(obj({'id':text,'observable':arr({'type':'string','enum':fields},1,6),
        'direction':text,'sources':refs,'reject_if':text}),0,6),
        'directions':arr(obj({'id':text,'node_id':node,'rationale':text}),cfg['initial_directions'] if phase=='initial' else 0,cfg['initial_directions'] if phase=='initial' else 0),
        'audits':arr(obj({'module':{'type':'string','enum':cfg['modules']},'statement':text,'sources':refs,'action_consequence':text}),0,12),
        'comparisons':arr(obj({'left':node,'right':node,'observable':text,'remaining_confounding':text,'distinguishing_outcome':text}),0,6),
        'updates':arr(obj({'node_id':node,'measured_basis':text,'next_test_reason':text}),0,8),
        'next_trials':arr(obj({'node_id':node,'replicate':{'type':'integer','minimum':0,'maximum':cfg['max_replicates']-1},'rationale':text}),cfg['followup_trials'] if phase=='feedback' else 0,cfg['followup_trials'] if phase=='feedback' else 0),
        'code':{'type':'string','maxLength':4500},'code_sources':arr({'type':'string','enum':sources},0,8)})


def observed_trials(study,requests):
    observations=[]
    for node,replicate in requests:
        path=Path(study)/'trials'/f'{node}_r{replicate:02d}.json'
        if not path.exists():raise FileNotFoundError('Requested feedback missing: '+str(path))
        record=read_json(path)
        if record['status']!='complete' or record['trial']['node_id']!=node or record['trial']['replicate']!=replicate:raise ValueError('Trial binding mismatch')
        observations.append({k:record.get(k) for k in ['pic50','elite_hit','valid_connected','qualified_structure','mmff_relief_per_heavy','energy_status','severe_pairs']}
            |{'node_id':node,'replicate':replicate})
    return observations


def numeric_initial(panel,cfg,diverse=False):
    cards=panel['candidate_cards'];indices=np.argsort([-c['joint_head_pic50'] for c in cards],kind='stable')
    chosen=[int(indices[0])]
    if diverse:
        z=np.asarray([list(c['standardized_features'].values()) for c in cards])
        while len(chosen)<cfg['initial_directions']:
            distance=np.linalg.norm(z[:,None]-z[chosen][None],axis=2).min(1);distance[chosen]=-1
            chosen.append(int(distance.argmax()))
    else:chosen=list(map(int,indices[:cfg['initial_directions']]))
    return {'hypotheses':[],'directions':[{'id':f'd{i}','node_id':cards[j]['node_id'],
        'rationale':'Deterministic head plus diversity' if diverse else 'Deterministic current head rank'} for i,j in enumerate(chosen)],
        'audits':[],'comparisons':[],'updates':[],'next_trials':[],'code':'','code_sources':[]}


def numeric_feedback(panel,cfg,observations):
    used={(x['node_id'],x['replicate']) for x in observations};by_node={}
    for x in observations:by_node.setdefault(x['node_id'],[]).append(x)
    # Terminal measured affinity among qualified molecules; no textual confidence.
    ranking=sorted(by_node,key=lambda n:(-sum(x['elite_hit'] for x in by_node[n])/len(by_node[n]),
        -np.mean([x['pic50'] if x['qualified_structure'] else 0. for x in by_node[n]]),n))
    requests=[]
    for n in ranking:
        for r in range(cfg['max_replicates']):
            if (n,r) not in used:requests.append({'node_id':n,'replicate':r,'rationale':'Measured qualified terminal feedback'});used.add((n,r))
            if len(requests)==cfg['followup_trials']:return requests
    raise ValueError('Insufficient remaining baseline budget')


def response_validator(panel,cfg,phase,modules,already):
    def validate(answer):
        def require(ok,message):
            if not ok:raise jsonschema.ValidationError(message)
        directions=answer['directions']
        require(len({x['node_id'] for x in directions})==len(directions),'Choose distinct initial nodes')
        require(len({h['id'] for h in answer['hypotheses']})==len(answer['hypotheses']),'Unique hypothesis IDs required')
        next_pairs=[(x['node_id'],x['replicate']) for x in answer['next_trials']]
        require(len(set(next_pairs))==len(next_pairs) and not set(next_pairs)&set(already),'Follow-up trials must be unique and unobserved')
        require(all(a['module'] in modules for a in answer['audits']),'Audit only active modules')
        if 'hypotheses' in modules:require(len(answer['hypotheses'])>=2,'Active hypotheses skill requires at least two testable hypotheses')
        for module in modules[1:]:require(any(a['module']==module for a in answer['audits']),'Active module must have a concrete audit entry: '+module)
        if 'discriminating-experiments' in modules:require(bool(answer['comparisons']),'Active experiment skill requires a real candidate comparison')
        if phase=='feedback' and 'feedback' in modules:require(bool(answer['updates']),'Active feedback skill requires measured updates')
        if 'coding' in modules:
            require(bool(answer['code_sources']),'Executable rule needs evidence citations')
            features={k:np.asarray([c['standardized_features'][k] for c in panel['candidate_cards']]) for k in panel['candidate_cards'][0]['standardized_features']}
            try:evaluate_rule(answer['code'],features)
            except (ValueError,SyntaxError,TypeError) as error:raise jsonschema.ValidationError(str(error)) from error
        else:require(answer['code']=='','Code is disabled in this condition')
    return validate


def decision(repo,cfg,condition,phase):
    repo=Path(repo);study=repo/cfg['study'];panel=read_json(study/'panel.json')
    folder=study/'conditions'/condition;folder.mkdir(parents=True,exist_ok=True)
    level=0 if condition in {'C0','B_diversity'} else int(condition[1:]);modules=cfg['modules'][:level]
    initial=read_json(folder/'initial.execution.json') if phase=='feedback' else None
    already=[(x['node_id'],r) for x in initial['directions'] for r in range(cfg['initial_replicates'])] if initial else []
    observations=observed_trials(study,already) if phase=='feedback' else []
    if not level:
        answer=numeric_initial(panel,cfg,condition=='B_diversity') if phase=='initial' else {
            **initial,'directions':[],'next_trials':numeric_feedback(panel,cfg,observations)}
    else:
        skills_root=repo/'skills'/'agentic-steer'
        skills=[(skills_root/'interface'/'SKILL.md').read_text(encoding='utf-8')]+[(skills_root/m/'SKILL.md').read_text(encoding='utf-8') for m in modules]
        # Compact but multiple outcome-conditioned historical paths. The full
        # blinded panel remains saved; no unrequested bank outcome is disclosed.
        packet=dict(panel)
        paths=panel['terminal_paths'];packet['terminal_paths']=[p for label in ['elite','ordinary'] for p in [x for x in paths if x['stratum']==label][:6]]
        packet['phase']=phase;packet['measured_feedback']=observations
        packet['previous_decision']=initial
        packet['budget']={'initial_directions':cfg['initial_directions'],'replicates_per_direction':cfg['initial_replicates'],
            'followup_trials':cfg['followup_trials'],'max_replicates':cfg['max_replicates']}
        packet['active_modules']=modules
        packet['output_instructions']='Return all schema keys, using empty arrays for unused optional scientific fields. Initial phase selects four distinct nodes and no next_trials; feedback phase has no directions and four unique unobserved next_trials. Cite only supplied source IDs. Code is empty unless coding is active. For coding use one def score(features): return expression, only np.abs/tanh/clip/sqrt/minimum/maximum/exp and registered standardized feature lookups. The host executes the rule ranking to choose next candidate priorities; use numerical measured feedback in updates.'
        messages=[{'role':'system','content':'You are an experimental acquisition analyst, not a binding oracle. Apply only active scientific Skills.\n\n'+'\n\n'.join(skills)},
            {'role':'user','content':json.dumps(packet,ensure_ascii=False,separators=(',',':'))}]
        answer=call_json(messages,schema(panel,cfg,phase),folder/(phase+'.json'),
            input_store=study/'inputs',max_output_tokens=7000,response_validator=response_validator(panel,cfg,phase,modules,already),max_format_repairs=2)
    code_audit=None
    if level==6:
        features={k:np.asarray([c['standardized_features'][k] for c in panel['candidate_cards']]) for k in panel['candidate_cards'][0]['standardized_features']}
        scores,code_audit=evaluate_rule(answer['code'],features)
        order=np.argsort(-scores,kind='stable');ranked=[panel['candidate_cards'][int(i)]['node_id'] for i in order]
        if phase=='initial':
            proposed=[x['node_id'] for x in answer['directions']]
            answer=dict(answer,directions=[{'id':f'code-{i}','node_id':n,'rationale':'Audited executable acquisition score'} for i,n in enumerate(ranked[:cfg['initial_directions']])])
            code_audit['llm_proposed_nodes']=proposed
        else:
            chosen=[];used=set(already)
            # Score selects candidate priorities; true measured success adds
            # information through the LLM's new evidence-bound expression.
            for n in ranked:
                for r in range(cfg['max_replicates']):
                    if (n,r) not in used:chosen.append({'node_id':n,'replicate':r,'rationale':'Audited feedback acquisition score'});used.add((n,r));break
                if len(chosen)==cfg['followup_trials']:break
            answer=dict(answer,next_trials=chosen)
    # Preserve raw API answer separately from the deterministic execution plan.
    write_json(folder/(phase+'.execution.json'),answer)
    if not level:write_json(folder/(phase+'.json'),answer)
    if phase=='initial':requests=[(d['node_id'],r) for d in answer['directions'] for r in range(cfg['initial_replicates'])]
    else:requests=[(d['node_id'],d['replicate']) for d in answer['next_trials']]
    trials=make_job(repo,cfg,requests,folder/(phase+'.job.json'))
    baseline=numeric_initial(panel,cfg)
    audit={'condition':condition,'phase':phase,'active_modules':modules,
        'instruction_sha256':{m:digest(repo/'skills'/'agentic-steer'/m/'SKILL.md') for m in ['interface']+modules},
        'chosen_trials':[(j['node_id'],j['replicate']) for j in trials],
        'initial_differs_from_head_baseline':bool(initial is None and {d['node_id'] for d in answer['directions']}!={d['node_id'] for d in baseline['directions']}),
        'functional_outputs':{'hypotheses':len(answer['hypotheses']),'audits':len(answer['audits']),'comparisons':len(answer['comparisons']),'updates':len(answer['updates'])},
        'code_audit':code_audit,
        'evidence_boundary':'Only the requested condition-specific initial observations were disclosed'}
    write_json(folder/(phase+'.audit.json'),audit)
    return trials


def summarize(repo,cfg):
    study=Path(repo)/cfg['study'];rows=[];archives=[]
    for folder in sorted((study/'conditions').iterdir()):
        if not (folder/'feedback.execution.json').exists():continue
        initial=read_json(folder/'initial.execution.json');feedback=read_json(folder/'feedback.execution.json')
        requests=[(d['node_id'],r) for d in initial['directions'] for r in range(cfg['initial_replicates'])]+[(d['node_id'],d['replicate']) for d in feedback['next_trials']]
        obs=observed_trials(study,requests);hits=sum(x['elite_hit'] for x in obs);n=len(obs)
        lo=0. if hits==0 else float(beta.ppf(.025,hits,n-hits+1));hi=1. if hits==n else float(beta.ppf(.975,hits+1,n-hits))
        qualified=[x['pic50'] for x in obs if x['qualified_structure']];energy=[x['mmff_relief_per_heavy'] for x in obs if x['mmff_relief_per_heavy'] is not None]
        rows.append({'condition':folder.name,'attempts':n,'elite_hits':hits,'hit_rate':hits/n,'hit_rate_95_CI':[lo,hi],
            'mean_pic50_all':float(np.mean([x['pic50'] for x in obs])),
            'best_qualified_pic50':max(qualified) if qualified else None,'qualified_rate':len(qualified)/n,
            'mean_mmff_relief_per_heavy':float(np.mean(energy)) if energy else None,
            'directions_observed':len({x['node_id'] for x in obs})})
        for node in sorted({x['node_id'] for x in obs}):
            vals=[x for x in obs if x['node_id']==node]
            archives.append({'condition':folder.name,'node_id':node,'hypotheses':initial['hypotheses'],
                'observations':vals,'evidence_scope':'Native stochastic suffix from one actual historical state; small n',
                'retained':True,'next_trial_actions':[x for x in feedback['next_trials'] if x['node_id']==node]})
    write_json(study/'performance.json',rows);write_json(study/'direction_archives.json',archives)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--repo',default='.')
    p.add_argument('--action',choices=['prepare','initial','feedback','summarize'],required=True);p.add_argument('--condition',default='C0');a=p.parse_args()
    cfg=read_json(a.config)
    if a.action=='prepare':result=make_panel(a.repo,a.config);print(json.dumps({'candidates':len(result['candidate_cards'])}))
    elif a.action=='summarize':print(json.dumps(summarize(a.repo,cfg),ensure_ascii=False))
    else:print(json.dumps({'trials':[t['trial_id'] for t in decision(a.repo,cfg,a.condition,a.action)]}))


if __name__=='__main__':main()
