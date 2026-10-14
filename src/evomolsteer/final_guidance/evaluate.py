"""Compact, deterministic paired feedback for final-first region experiments."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evomolsteer.io import read_json,write_json,write_table,digest
from .evidence import compact_regions
from .reward import HalfStateReward


def paired_interval(values,seed=42,repeats=4000):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return [None,None]
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(repeats,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def evaluate(repo,cfg,generated,output):
    repo=Path(repo);generated=Path(generated);output=Path(output);study=repo/cfg['study']
    done=read_json(generated/'COMPLETE.json');manifest=read_json(study/'manifest.json')
    source=read_json(generated/'source_audit.json');summaries=[];observables=[];paired=[];checks=[]
    baseline=read_json(generated/'unguided/records.json');baseline_by_slot={r['slot']:r for r in baseline}
    evidence_sources=[]
    for binding_path in sorted((study/'candidates').glob('*/bindings.json')):
        binding=read_json(binding_path);teacher=read_json(repo/binding['teacher'])
        analyst=read_json(binding_path.parent/'Analyst.json');packet=read_json(binding_path.parent/'terminal_packet.json')
        local=next(read_json(repo/p['program']) for p in binding['programs'] if read_json(repo/p['program'])['scope']=='local')
        full=next(read_json(repo/p['program']) for p in binding['programs'] if read_json(repo/p['program'])['scope']=='full')
        evidence_sources.append((local,full,teacher,analyst,packet))
    if len(evidence_sources)!=1:raise ValueError('Initial feedback adapter expects one frozen candidate; do not mix different teachers')
    local,full,teacher,analyst,packet=evidence_sources[0]
    binding=read_json(study/'candidates'/teacher['candidate_id']/'bindings.json')
    instructions={'Analyst':repo/'skills/final-candidate-analyst/SKILL.md',
        'Designer':repo/'skills/half-state-reward-designer/SKILL.md','contract':repo/'skills/final-region-guidance-workflow/references/contract.md'}
    skill_match=all(hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()==binding['instruction_sha256'][role]
        for role,path in instructions.items())
    if not skill_match:raise ValueError('Current instructions differ from actual delivered API instructions')
    local_reward=HalfStateReward(local,teacher);full_reward=HalfStateReward(full,teacher)
    names=['unguided']+[read_json(repo/p['program'])['program_id'] for p in read_json(study/'candidates'/teacher['candidate_id']/'bindings.json')['programs']]
    if done['conditions']!=len(names):raise ValueError('Incomplete condition matrix')
    for name in names:
        folder=generated/name;rows=read_json(folder/'records.json');audit=read_json(folder/'guidance_audit.json');geometry=read_json(folder/'geometry.json')
        if [r['slot'] for r in rows]!=list(baseline_by_slot):raise ValueError('Different initial random-input slots')
        if audit['input_pairing']!=source or not audit['no_smc'] or not audit['no_affinity_head_gradient']:raise ValueError('Pairing or forbidden-gradient audit differs')
        if audit['actual_steps']!=cfg['steps']:raise ValueError('Incomplete native suffix')
        program=audit['program'];expected=[] if program is None else program['expected_active_steps']
        if audit['actual_active_steps']!=expected:raise ValueError('Guidance window differs from decision')
        nonzero=sum(s['nonzero_molecules']>0 for s in audit['steps'])
        if nonzero!=len(expected):raise ValueError('An approved active update delivered no displacement')
        program_match=program is None or any(read_json(repo/item['program'])==program and digest(repo/item['program'])==item['sha256']
            for item in binding['programs'])
        if not program_match:raise ValueError('Executed program differs from compiled and hashed decision')
        direct_background=max(s.get('direct_unmatched_gradient_max',0) for s in audit['steps'])
        if program is not None and program['scope']=='local' and direct_background!=0:raise ValueError('Direct local loss touched unmatched points')
        checks.append({'condition':name,'expected_updates':len(expected),'actual_updates':len(audit['actual_active_steps']),
            'nonzero_updates':nonzero,'no_smc':audit['no_smc'],'direct_background_gradient_max':direct_background,
            'model_mediated_background_gradient_max':max(s.get('current_unmatched_gradient_max',0) for s in audit['steps']),
            'program_sha256_matches':program_match,'terminal_steps':audit['actual_steps']})
        scores=np.array([r['pic50'] if r['pic50'] is not None else np.nan for r in rows]);finite=np.isfinite(scores)
        differences=scores-np.array([baseline_by_slot[r['slot']]['pic50'] for r in rows])
        half=geometry['teacher_clock'];z=torch.as_tensor(half['coords_A'],dtype=torch.double);mask=torch.as_tensor(half['mask'],dtype=torch.bool)
        if half['representation']!='current_world_A' or half['time']!=cfg['teacher_time']:raise ValueError('Wrong real half-state feedback')
        local_dist=local_reward.value(z,mask)[1]['matched_rms_A'].numpy();full_dist=full_reward.value(z,mask)[1]['matched_rms_A'].numpy()
        energy=np.array([r['mmff_relief_per_heavy'] if r['mmff_relief_per_heavy'] is not None else np.nan for r in rows])
        summary={'condition':name,'scope':'unguided' if program is None else program['scope'],
            'start':None if program is None else program['window'][0],'n_attempts':len(rows),'n_scored':int(finite.sum()),
            'mean_pic50_scored':float(scores[finite].mean()) if finite.any() else None,'best_pic50':float(scores[finite].max()) if finite.any() else None,
            'paired_delta_pic50':float(np.nanmean(differences)),'paired_bootstrap95_low':paired_interval(differences)[0],
            'paired_bootstrap95_high':paired_interval(differences)[1],
            'valid_connected':sum(r['valid_connected'] for r in rows),'qualified_basic_structure':sum(r['qualified_structure'] for r in rows),
            'energy_coverage':int(np.isfinite(energy).sum()),'median_mmff_relief_per_heavy':float(np.nanmedian(energy)),
            'p90_mmff_relief_per_heavy':float(np.nanquantile(energy,.9)),
            'actual_half_local_match_RMS_A':float(local_dist.mean()),'actual_half_full_match_RMS_A':float(full_dist.mean()),
            'active_updates':len(expected),'nonzero_updates':sum(s['nonzero_molecules']>0 for s in audit['steps']),
            'actual_cumulative_control_RMS_A':audit['mean_cumulative_control_rms_A']}
        summaries.append(summary)
        for i,r in enumerate(rows):paired.append({'condition':name,'slot':r['slot'],'pic50':r['pic50'],
            'baseline_pic50':baseline_by_slot[r['slot']]['pic50'],'delta_pic50':differences[i],
            'half_local_match_RMS_A':local_dist[i],'half_full_match_RMS_A':full_dist[i]})
        final=geometry['terminal']
        if final['representation']!='terminal_world_A' or final['time']!=cfg['terminal_time']:raise ValueError('Forecast final feedback rejected')
        fields=compact_regions(np.asarray(final['coords_A']),manifest['landmarks_A'],manifest['region_radius_A'])
        for h in analyst['hypotheses']:
            col=manifest['feature_names'].index(h['observable']);v=fields[:,col];usable=np.isfinite(v)
            row=next(r for r in packet['one_to_rest'] if r['evidence_id']==h['evidence_id'])
            observables.append({'condition':name,'hypothesis_id':h['id'],'observable':h['observable'],
                'n_coordinate_supported':int(usable.sum()),'mean_generated_final':float(v[usable].mean()) if usable.any() else None,
                'historical_candidate_value':row['candidate_value'],'historical_rest_mean':row['rest_mean'],
                'not_the_optimized_reward_itself':True})
    write_table(output/'conditions.csv',summaries);write_table(output/'paired.csv',paired);write_table(output/'observable_response.csv',observables)
    write_json(output/'execution_checks.json',{'skills_sha256_match':skill_match,'rows':checks,'all_windows_match':True,
        'raw_output_bytes':sum(path.stat().st_size for path in generated.rglob('*') if path.is_file())})
    historical=pd.read_parquet(repo/cfg['terminal_metrics']);values=historical['pic50_on_rescore'].to_numpy(float)
    write_json(output/'summary.json',{'conditions':summaries,'historical_steer_reference':{'n':len(values),
        'mean_pic50':float(np.nanmean(values)),'best_pic50':float(np.nanmax(values)),'paired':False},
        'inference_commit_source':source,'source_complete_sha256':digest(generated/'COMPLETE.json'),
        'limits':['Small pilot from one initial batch; bootstrap intervals are descriptive and not multiplicity-adjusted',
            'One highest-final candidate analyzed; remaining candidates pending',
            'Final head prediction is not experimental affinity','Basic structure check is not full PoseBusters',
            'Starts share per-step dose, not total exposure','Regional imitation is not a causal test of occupancy or covariance alone']})
    return pd.DataFrame(summaries)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',default='.');parser.add_argument('--config',required=True)
    parser.add_argument('--generated',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(evaluate(args.repo,read_json(args.config),args.generated,args.output).to_string(index=False))


if __name__=='__main__':main()
