"""Freeze literal Skills and compact cohort evidence for sequential Agent calls."""
import argparse,json
from pathlib import Path
import pandas as pd
from evomolsteer.io import read_json,write_json,digest


def verified_response(output,role):
    output=Path(output);request=read_json(output/(role+'.request.json'));response=read_json(output/(role+'.response.json'))
    for key,suffix in [('input_sha256','.input.json'),('instruction_sha256','.instructions.md')]:
        if request[key]!=digest(output/(role+suffix)) or response.get(key)!=request[key]:
            raise ValueError('Actual Agent input/instruction binding changed')
    if response.get('role')!=role:raise ValueError('Agent response role differs')
    return response


def prepare(repo,role='Analyst'):
    repo=Path(repo).resolve();docs=repo/'docs/experiments/dynamic_contrast10_20261008'
    evidence=docs/'rest_mining';out=docs/'round02_agents';out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(evidence/'manifest.json');effects=pd.read_csv(evidence/'whole_window_effects.csv')
    cohorts=pd.read_csv(evidence/'event_cohorts.csv');grid=cohorts.groupby('time').agg(
        positive_mean=('positive_n','mean'),negative_mean=('negative_n','mean'),ambiguous_mean=('ambiguous_n','mean'),
        threshold_mean=('threshold','mean'),score_positive=('positive_score_mean','mean'),score_negative=('negative_score_mean','mean'),
        root_mean=('root_n','mean'),observed_batches=('identifiable','sum')).reset_index()
    selected=manifest['policy']['selected_feature_indices']
    packet={'schema_version':'dynamic-regional-agent-evidence-1.0','role':role,
        'objective':'Final predicted target affinity first, strain secondary; discover beneficial spatial regions without new neural fitting',
        'learned_support':manifest['window'],'representation':'Endpoint forecast world coordinates at score_time; current update must end in support',
        'evidence':{'E_cohort':manifest,'E_effects':effects.to_dict('records'),'E_timecourse':grid.to_dict('records'),
            'E_fit':{k:v for k,v in read_json(evidence/'effect_functions.json').items() if k in effects.loc[effects.feature_index.isin(selected),'feature'].tolist()},
            'E_prior_failure':read_json(repo/'configs/experiments/elite_path20_v1/recommendation.json')},
        'historical_incumbent':read_json(repo/'configs/experiments/skill_ablation_v1/incumbent.json'),
        'experimental_contract':{'only_change':'Add one empirical regional scalar to unchanged R26 endpoint attractor',
            'registry':['endpoint_dynamic_region'],'regional_weight':.05,'regional_temperature':1.,'regional_bound':2.,
            'regional_response':'contrast','allowed_region_indices':selected,'seed':42,'screen_batch':36,
            'derivative_path':'Actual FLOWR endpoint coordinate VJP, detached affinity outputs; no added production forward calls',
            'regional_formula':'Common-variance, equal-batch Gaussian moment kernels; bounded tanh(log positive kernel - log lower-score kernel). Keep original teacher-mixture attraction.',
            'graph_changes':'Allowed, no graph or intraligand pair-distance acceptance gate',
            'parameter_origin':'Exploratory registered scalar strengths, not effect-derived optimal weights'},
        'limitations':['Observed online joint-head association, not independent endpoint rescore or terminal causality',
            'Five unidentifiable event/batches are missing evidence; cohort fits use observed cells and report coverage',
            'Landmark softmin and radial shells are coordinate proxies, not energetic interactions',
            'Screen selection is adaptive; confirmation panels frozen before observing their labels']}
    if role=='Designer':
        analyst=out/'Analyst.response.json'
        packet['analyst_result']=verified_response(out,'Analyst');packet['analyst_response_sha256']=digest(analyst)
        identifiable=cohorts[cohorts.identifiable]
        packet['evidence']['E_cohort_weight_coverage']={
            name:{'min':float(identifiable[name].min()),'median':float(identifiable[name].median()),'max':float(identifiable[name].max())}
            for name in ['positive_effective_n','negative_effective_n','positive_root_n','negative_root_n','margin']}
    write_json(out/(role+'.input.json'),packet)
    skill=repo/'skills'/role.lower()/'SKILL.md';module=repo/'skills/dynamic-cohort-contrast/SKILL.md'
    instructions=skill.read_text(encoding='utf-8')+'\n\n'+module.read_text(encoding='utf-8')+'\n\n'
    if role=='Analyst':
        instructions+='Task: Audit the complete-window dynamic cohorts. Return JSON with schema_version=dynamic-regional-analyst-1.0, role=Analyst, input_sha256, instruction_sha256, findings (claim, evidence_ids), regions (indices, direction, time_course, evidence_ids), limitations and recommendation (test or defer). Cite measurements; compare the recorded incumbent and counterevidence. No private reasoning transcript.\n'
    else:
        instructions+='Task: Design only the next registered additive regional contrast, without tuning other axes. Return JSON with schema_version=dynamic-regional-designer-1.0, role=Designer, input_sha256, instruction_sha256, decision (test_formula or defer), reward_view, selected_feature_indices, formula, justification, failure_modes and verification. If justified, keep the supplied experimental contract parameters. No private reasoning transcript.\n'
    (out/(role+'.instructions.md')).write_text(instructions,encoding='utf-8',newline='\n')
    request={'role':role,'input_sha256':digest(out/(role+'.input.json')),'instruction_sha256':digest(out/(role+'.instructions.md')),
        'base_skill_sha256':digest(skill),'module_skill_sha256':digest(module),'transport':'User-authorized sequential subagent API simulation'}
    write_json(out/(role+'.request.json'),request);return request

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--role',choices=['Analyst','Designer'],default='Analyst')
    a=p.parse_args();print(json.dumps(prepare(a.repo,a.role)))
