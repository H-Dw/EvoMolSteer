"""Evaluate sealed score-blind selections against observed Steer descendants.

Extinction censors terminal quality. Elite recognition and observed survival are
separate; this replay cannot estimate counterfactual affinity of extinct seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from evomolsteer.io import read_json, write_json, digest


def evaluate(trial, output):
    trial, output = Path(trial), Path(output)
    protocol = read_json(trial/'blind/protocol.json')
    manifest = read_json(trial/'manifest.json')
    oracle = read_json(trial/'oracle/candidate_truth.json')
    for name, expected in manifest['blind_files'].items():
        if digest(trial/'blind'/name) != expected:
            raise ValueError('Blind input was altered: '+name)
    truth = {(r['step'],r['id']):r for r in oracle['candidates']}
    records, summaries, continuations, contracts = [], [], [], []
    for step in protocol['start_steps']:
        file = trial/'blind/answers'/f'ranking_{step:03d}.json'
        answer = read_json(file)
        if abs(answer['t']-step/100) > 1e-8:
            raise ValueError('Wrong initial ranking time')
        ranked = sorted(answer['ranking'],key=lambda r:r['rank'])
        ids = [r['candidate_id'] for r in ranked]
        if [r['rank'] for r in ranked] != list(range(1,6)) or len(set(ids)) != 5:
            raise ValueError('Five distinct ranked directions required')
        if any((step,i) not in truth for i in ids):
            raise ValueError('Unknown candidate identity in sealed ranking')
        pool = [r for r in oracle['candidates'] if r['step']==step]
        n, k = len(pool),sum(r['has_elite_descendant'] for r in pool)
        survival = sum(r['future_observed'] for r in pool)
        known = [r for r in pool if r['quality_future_max_pic50'] is not None]
        best = max((r['quality_future_max_pic50'] for r in known),default=None)
        future_order = sorted(known,key=lambda r:-r['quality_future_max_pic50'])
        position = {r['id']:i+1 for i,r in enumerate(future_order)}
        hits, surviving, values, qcounts = 0,0,[],[]
        heads = np.array([r['intermediate_pic50'] for r in pool],float)
        for item in ranked:
            r = truth[(step,item['candidate_id'])]
            hits += r['has_elite_descendant']; surviving += r['future_observed']
            if r['quality_future_max_pic50'] is not None:
                values.append(r['quality_future_max_pic50'])
            qcounts.append(r['n_quality_descendants'])
            records.append({'t':step/100,'rank':item['rank'],'id':r['id'],
                'batch':r['batch'],'slot':r['slot'],'alias_count':len(r['aliases']),
                'n_descendants':r['n_descendants'],'n_quality_descendants':r['n_quality_descendants'],
                'future_observed':r['future_observed'],'has_elite_descendant':r['has_elite_descendant'],
                'future_quality_rank_among_observed':position.get(r['id']),
                'quality_future_max_pic50':r['quality_future_max_pic50'],
                'quality_future_mean_pic50':r['quality_future_mean_pic50'],
                'intermediate_pic50':r['intermediate_pic50'],
                'intermediate_score_percentile':float((heads<=r['intermediate_pic50']).mean()),
                'evidence':item.get('matched_description',''),'counterevidence':item.get('counterevidence',''),
                'confidence':item.get('confidence','')})
        summaries.append({'t':step/100,'pool':n,'observed_surviving_states':survival,
            'observed_quality_states':len(known),'observed_elite_states':k,
            'selected_observed':surviving,'selected_censored':5-surviving,
            'selected_elite':hits,'elite_precision_all_attempts':hits/5,
            'elite_recall_observed_states':hits/k if k else None,
            'random_five_expected_elite':5*k/n,'random_five_expected_surviving':5*survival/n,
            'random_five_probability_at_least_observed_hits':float(hypergeom.sf(hits-1,n,k,5)),
            'best_quality_future_pic50_selected':max(values) if values else None,
            'best_quality_future_pic50_available':best,
            'best_future_gap':best-max(values) if best is not None and values else None,
            'selected_quality_descendant_counts':qcounts,
            'ranking_sha256':digest(file),'method_summary':answer.get('method_summary','')})
        replay_file = trial/'blind/answers'/f'continuation_{step:03d}.json'
        if replay_file.exists():
            replay = read_json(replay_file)
            previous, previous_step = ids, step
            nodes = replay['nodes']
            first = nodes[0]
            if round(first['t']*100)!=step or [r['candidate_id'] for r in first['ranking']]!=ids:
                raise ValueError('Replay initial ranking changed')
            for node in nodes:
                at = round(node['t']*100)
                chosen = [r['candidate_id'] for r in node['ranking']]
                if len(chosen)>5 or len(set(chosen))!=len(chosen):
                    raise ValueError('Invalid replay cardinality')
                if at != step:
                    if at != previous_step+1:
                        raise ValueError('Every 0.01 replay node required')
                    eligible = [r for r in oracle['candidates'] if r['step']==at and r['parent_id'] in previous]
                    if node['eligible_count']!=len(eligible) or not set(chosen)<=set(r['id'] for r in eligible):
                        raise ValueError('Replay admitted an unrelated or unobserved child')
                    if len(chosen)!=min(5,len(eligible)):
                        raise ValueError('Replay must retain all eligible up to budget five')
                entries = [truth[(at,i)] for i in chosen]
                continuations.append({'start_t':step/100,'t':at/100,'chosen':len(chosen),
                    'eligible':node['eligible_count'],'observed_elite':sum(r['has_elite_descendant'] for r in entries),
                    'surviving':sum(r['future_observed'] for r in entries),
                    'best_quality_future_pic50':max((r['quality_future_max_pic50'] for r in entries if r['quality_future_max_pic50'] is not None),default=None),
                    'ids':chosen})
                previous,previous_step=chosen,at
        contracts.append({'step':step,'initial_ranking_valid':True,'replay_present':replay_file.exists()})
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'evaluation.json',{'definitions':{
        'elite':'Observed terminal descendant with final common pIC50 rescore >= training QC 95th percentile, valid connected, PB fast pass, converged MMFF and strain <= training QC 90th percentile',
        'censored':'No descendant survives in the recorded Steer run; terminal affinity is unknown',
        'best_future':'Retrospective maximum over quality-qualified observed descendants; larger descendant counts give more opportunities',
        'random_reference':'Exact uniform sampling of five distinct current states, a descriptive reference, not five independent hypothesis tests'},
        'elite_threshold':oracle['elite_threshold'],'strain_cap':oracle['strain_cap'],
        'starts':summaries,'rankings':records,'continuations':continuations,'contracts':contracts,
        'oracle_sha256':digest(trial/'oracle/candidate_truth.json'),
        'procedural_blinding_only':True,'labels_Steer_conditioned':True})
    pd.DataFrame(records).to_csv(output/'ranked_directions.csv',index=False,float_format='%.10g')
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False,float_format='%.10g')
    pd.DataFrame(continuations).to_csv(output/'continuations.csv',index=False,float_format='%.10g')
    print(pd.DataFrame(summaries)[['t','pool','observed_elite_states','selected_observed','selected_elite','best_future_gap']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.trial,a.output)
