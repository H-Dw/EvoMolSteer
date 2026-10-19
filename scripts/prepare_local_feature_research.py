"""Freeze a real Analyst/Designer local-support decision and paired replay pilot."""
import argparse
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from evomolsteer.io import read_json,write_json,digest
from evomolsteer.path_replay.common import identity,read_batch
from evomolsteer.path_replay.analysis import distribution
from evomolsteer.ancestral_modes.support_mining import load_registry,compile_program
from evomolsteer.ancestral_modes.programs import numerical_identity,program_delta


def prepare(repo,master_path,round_id,operators):
    repo=Path(repo).resolve();m=read_json(master_path);base_path=repo/m['base_config'];cfg=read_json(base_path)
    bank=read_json(repo/m['teacher_bank']);registry,regions=load_registry(repo/m['local_support_registry'],bank)
    parent=read_json(repo/m['initial_program']);parent_id=parent['program_id']
    regions=[r for r in regions if r['shortlisted'] and r['template_id'] in {s['template_id'] for s in parent['mode_specs']}]
    if m.get('fixed_region_id'):regions=[r for r in regions if r['region_id']==m['fixed_region_id']]
    reference=next(c for c in cfg['conditions'] if c['id']=='ancestor_R02_v3');records=[];steps=[]
    for gi in m['group_indices']:
        receipt=read_batch(repo/m['local_base_work'],cfg,reference,cfg['groups'][gi])
        if not receipt or receipt['status']!='complete':raise ValueError('Missing verified paired reference')
        records.extend(receipt['records'])
        with gzip.open(repo/m['local_base_work']/f'methods/ancestor_R02_v3/group_{gi:03d}/guidance.json.gz','rt') as f:
            steps.extend(s for s in json.load(f)['steps'] if s['active'])
    response={'active_updates':len(steps),'nonzero_updates':sum(s['nonzero_molecules']>0 for s in steps),
              'nonzero_seed_updates':sum(s['nonzero_molecules'] for s in steps),
              'statistics':{k:float(np.mean([s[k] for s in steps if k in s])) for k in
                ('injected_rms_A_mean','predictive_flow_rms_A_mean','assignment_switch_fraction') if any(k in s for s in steps)}}
    entry={'program':parent,'metrics':distribution(records,len(records),cfg['tail_threshold']),'response':response}
    if m.get('parent_feedback'):
        from evomolsteer.ancestral_modes.replay_feedback import entry as feedback_entry
        entry=feedback_entry(repo,m['parent_feedback'],parent)
        write_json(repo/m['campaign_study']/round_id/'parent_feedback.json',entry)
    state={'reference':entry,'best':entry,'rounds':[],
           'original':{'best':max(a['historical_pic50'] for a in cfg['attempts'])},
           'prior_campaign_evidence':m.get('prior_campaign_evidence')}
    actions=[]
    for r in regions:
        for operator in operators:
            p=compile_program(parent,r,bank,registry['policy'],operator)
            p['program_id']=round_id
            # The first support experiment deliberately freezes the parent's
            # dose, time mapping and curvature. Later factors get separate runs.
            actions.append({'action_id':f'option_{len(actions):03d}','parent_id':parent_id,
                'region_ids':[r['region_id']],'phase':'local_support','reason':'Measured local support/observable; no full-cloud substitution',
                'program':p,'numeric_identity':numerical_identity(p,bank),'authoritative_program_delta':program_delta(parent,p)})
    if not actions:raise ValueError('No compatible local actions')
    folder=repo/m['campaign_study']/round_id
    write_json(folder/'allowed_actions.json',actions)
    from evomolsteer.ancestral_modes.agents import ask
    p,action,delivery=ask(repo,m,folder/'agents',bank,regions,state,actions,'local_support',modules_override=[])
    from evomolsteer.ancestral_modes.research import payload
    teacher=payload(bank,p)
    if read_json(repo/reference['reference'])==teacher:
        teacher_path=repo/reference['reference']
    else:
        teacher_path=repo/m['campaign_study']/'teachers'/(identity(teacher)+'.json')
        if not teacher_path.exists():write_json(teacher_path,teacher)
    p['teacher_sha256']=digest(teacher_path)
    # Record real decision provenance separately from the compiler's binding.
    p['local_feature_binding']['decision_provenance']={'analyst_sha256':p['analyst_sha256'],
        'designer_sha256':p['designer_sha256'],'delivery':(folder/'agents/delivery.json').relative_to(repo).as_posix()}
    program_path=folder/'program.json';write_json(program_path,p)
    from evomolsteer.ancestral_modes.programs import validate
    validate(p,read_json(teacher_path))
    zero={'id':'zero_replay','kind':'unguided','n':50*len(m['group_indices']),'window':None,'group_indices':m['group_indices']}
    baseline=copy.deepcopy(reference);baseline.update(id='R02_reference',n=zero['n'],group_indices=m['group_indices'],reuse_from='ancestor_R02_v3')
    if not m.get('reuse_parent',True):
        baseline.pop('reuse_from');baseline.update(role='frozen_current_reference',frozen_source_condition='ancestor_R02_v3')
    if m.get('reuse_current_controls'):
        shared=m['reuse_current_controls'];baseline.pop('reuse_from',None)
        zero['reuse_current_from']={'manifest':shared['manifest'],'condition':'zero_replay'}
        baseline.update(role='frozen_current_reference',frozen_source_condition='ancestor_R02_v3',
            reuse_current_from={'manifest':shared['manifest'],'condition':'R02_reference'})
    candidate={'id':round_id,'kind':'native_local','n':zero['n'],'group_indices':m['group_indices'],'window':p['window'],
        'program':program_path.relative_to(repo).as_posix(),'program_sha256':digest(program_path),
        'reference':teacher_path.relative_to(repo).as_posix(),'reference_sha256':digest(teacher_path)}
    manifest={'schema_version':'local-feature-path-research-1','name':m['campaign_id']+'/'+round_id,
        'base_config':m['base_config'],'base_config_sha256':digest(base_path),'base_protocol_sha256':cfg['protocol_sha256'],
        'base_work':cfg['remote_work'],'remote_work':m['remote_campaign']+'/'+round_id,
        'saved_tensor_policy':'saved_tensors_cpu_exact','batch_size':50,'batch_timeout_seconds':7200,
        'parent_condition_id':'R02_reference','conditions':[zero,baseline,candidate],
        'scientific_scope':{'source':'true historical initial tensors and ancestry-mapped primitive random inputs',
            'changed':program_delta(parent,p),'registry_identity':registry['identity'],
            'decision_chain':'real DeepSeek Analyst -> Designer option -> deterministic compiler -> validated local feature binding'}}
    for key in ('comparison_scope','diagnostic_evidence','native_zero_repeat_sources'):
        if key in m:manifest[key]=m[key]
    manifest['research_sha256']=identity(manifest)
    output=repo/m['config_dir']/(round_id+'.json');write_json(output,manifest)
    from evomolsteer.path_replay.local_research import load
    load(repo,output)
    print(json.dumps({'manifest':str(output),'program':str(program_path),'binding':p['local_feature_binding'],'delivery':delivery},ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--master',type=Path,required=True);p.add_argument('--round',required=True)
    p.add_argument('--operator',action='append',choices=['pointwise','pair_distance','pocket_distance','orientation'],required=True)
    a=p.parse_args();prepare(a.repo,a.master,a.round,a.operator)


if __name__=='__main__':main()
