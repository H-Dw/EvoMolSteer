"""CPU-only analysis of raw saved endpoints and paired, previously audited records."""
import os
for key in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:
    os.environ[key] = ''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import gzip
import hashlib
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
DOC = OUT.parents[1]
OLD = DOC.parent/'20261031_guidance_comparison5/monitoring/20261031_2017'
SCRATCH = Path('D:/llm_controlled_generation/tmp/comparison5_final')
METRIC = 'condition_coordinate_root_mean_square_deviation_angstrom'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
def paired(old,new):
    a={(r['logical_seed'],r['logical_slot']):r[METRIC] for r in old}
    b={(r['logical_seed'],r['logical_slot']):r[METRIC] for r in new}
    assert set(a)==set(b) and len(a)==50
    changes={k:b[k]-a[k] for k in a}
    return {'population':50,'mean_change':statistics.mean(changes.values()),
            'improved':sum(v<0 for v in changes.values()),'worsened':sum(v>0 for v in changes.values()),
            'equal':sum(v==0 for v in changes.values()),
            'seed_cluster_changes':{str(s):statistics.mean(v for k,v in changes.items() if k[0]==s) for s in sorted({k[0] for k in a})},
            'same_frozen_initial_states':True,'independent_validation':False}

old_manifest=read(OLD/'final_readout_manifest.json')
old_facts=read(OLD/'final_facts.json')
previous={}
for job in old_facts['jobs']:
    path=SCRATCH/(job['id']+'_final.json')
    assert sha(path)==old_manifest[path.name]
    previous[job['id']]=read(path)
new=[read(OUT/(name+'_results.json')) for name in ['03_prefix_joint_6pyy','06_five_windows_3w54']]
chemistry=[];paired_facts={};rounds=[];role_decisions=[]
for job in new:
    best=job['best'];terminal=OUT/(job['id']+'_terminal.pt.gz')
    member='jobs/'+job['id']+'/attempt_002/'+best['source']+'/terminal.pt.gz'
    manifests=[read(OUT.parent/'20261101_0118/archive_manifest.json')['files'],read(OUT/'archive_manifest.json')['files']]
    digest=next(m[member]['sha256'] for m in manifests if member in m)
    assert sha(terminal)==digest
    with gzip.open(terminal,'rb') as stream:
        tensors=torch.load(stream,map_location='cpu',weights_only=True)
    ids=torch.tensor([r['condition_generated_slots'] for r in best['raw_records']],dtype=torch.long)
    rows=torch.arange(50)[:,None]
    features={k:torch.tensor(v).argmax(-1) for k,v in best['chemical_condition']['features'].items()}
    exact=torch.ones(50,dtype=torch.bool);channels={}
    for key in ['atomics','charges','hybridization']:
        if key in features:
            matches=tensors[key][rows,ids].argmax(-1)==features[key]
            channels[key]={'matched':int(matches.sum()),'items':matches.numel(),'fraction':float(matches.double().mean())}
            exact &= matches.all(-1)
    k=ids.shape[1];left,right=torch.where(~torch.eye(k,dtype=torch.bool))
    matches=tensors['bonds'][rows,ids[:,left],ids[:,right]].argmax(-1)==features['bonds'][left,right]
    present=features['bonds'][left,right]!=0
    channels['induced_bonds']={'matched':int(matches.sum()),'items':matches.numel(),'fraction':float(matches.double().mean())}
    channels['present_target_bonds']={'matched':int(matches[:,present].sum()),'items':matches[:,present].numel(),'fraction':float(matches[:,present].double().mean())}
    exact &= matches.all(-1)
    chemistry.append({'job':job['id'],'source':best['source'],'terminal_sha256':digest,'population':50,
                      'controlled_atoms':k,'channels':channels,'all_controlled_features_match':int(exact.sum()),
                      'raw_terminal_not_projected':True,'includes_unbuilt_candidates':True,
                      'same_atom_identity_not_guaranteed':True})
    for old_id in [job['id'],'01_baseline_6pyy' if '6pyy' in job['id'] else '02_baseline_3w54']:
        old=previous[old_id]
        group=next(g for g in old['groups'] if g['path']==old['best_available_complete_population'])
        paired_facts[job['id']+'_vs_'+old_id]=paired(group['raw_records'],best['raw_records'])
    v=job['version_finalized']
    assert v['configured_rounds']==v['completed_rounds']==5 and v['version']=='GD01'
    for r in v['rounds']:
        rounds.append({'job':job['id'],**r})
    for r in job['roles']:
        role_decisions.append({'job':job['id'],**r})
summary=read(OUT/'result_summary.json')
summary['old_charged_attempts_preserved']=3790
summary['new_charged_attempts']={s:sum(j['attempts'][s] for j in new) for s in ['calibration','formal']}
summary['total_new_attempts']=sum(summary['new_charged_attempts'].values())
summary['total_old_and_new_attempts']=3790+summary['total_new_attempts']
summary['total_provider_responses']=sum(j['provider_responses'] for j in new)
summary['total_completion_tokens']=sum(j['completion_tokens'] for j in new)
summary['accepted_role_submissions']=sum(len(j['roles']) for j in new)
summary['implicit_field_completion']=False
summary['missing_selected_in_accepted_submissions']=0
summary['new_sampling_or_api_during_review']=False
summary['local_analysis_timestamp']=datetime.now(timezone(timedelta(hours=8))).isoformat()
write('final_facts.json',summary)
write('endpoint_chemical_preservation.json',{'cpu_only':True,'new_model_forward':False,'jobs':chemistry})
write('paired_comparisons.json',paired_facts)
write('rounds_and_range_events.json',rounds)
write('llm_round_decisions.json',role_decisions)
write('quality_source_summary.json',[{'job':j['id'],**q} for j in new for q in j['qualities']])
combined=[]
for old in old_facts['jobs']:
    replacement=next((j for j in new if j['id']==old['id']),None)
    if replacement:
        b=replacement['best']
        row={'id':old['id'],'teacher':replacement['teacher'],'experiment':replacement['experiment'],'skill_release':'2.8.2',
             'science_commit':'78e6e1803758f7016a2568b00176eaaa27ef6dc4','rounds':5,
             'schedule':b['schedule'],'rmsd':b['raw_rmsd'],'loss':b['raw_loss'],'built':b['built'],'source':b['source']}
    else:
        row={'id':old['id'],'teacher':old['teacher'],'experiment':old['experiment'],'skill_release':'2.8.1',
             'science_commit':old_facts['commit'],'rounds':old['rounds_completed'],'schedule':old['best_actual_scales'],
             'rmsd':old['best_raw_rmsd'],'loss':old['best_raw_loss'],'built':old['best_built'],'source':old['best_population_source']}
    combined.append(row)
write('combined_six_strategy_results.json',{'different_control_versions':True,'independent_validation':False,'jobs':combined})
print(json.dumps({'summary':{k:summary[k] for k in ['new_charged_attempts','total_new_attempts','total_old_and_new_attempts','total_provider_responses','total_completion_tokens','accepted_role_submissions']},'chemistry':chemistry,'paired':paired_facts,'combined':combined},ensure_ascii=True))
