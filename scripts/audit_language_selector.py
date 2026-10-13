"""Post-unblinding forensic audit of the selector's executed text matcher.

The archived selector is inspected before loading. Its public geometry ranking
is reconstructed; oracle labels are used only to diagnose omissions and losses.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evomolsteer.io import read_json, write_json, digest


def audit(trial, output):
    trial, output=Path(trial),Path(output)
    path=trial/'blind/run_selector.py'
    spec=importlib.util.spec_from_file_location('frozen_selector',path)
    selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
    oracle=read_json(trial/'oracle/candidate_truth.json')
    truth={(r['step'],r['id']):r for r in oracle['candidates']}
    starts=[]
    for step in [1,5,10,20,30]:
        protos=selector.cards(trial/f'blind/prototypes_{step:03d}.zh-CN.md',True)
        states=selector.cards(trial/f'blind/states_{step:03d}.zh-CN.md')
        distances={a['id']:min(selector.score(a,p) for p in protos) for a in states}
        ordered=sorted(distances,key=lambda i:(distances[i],i));position={i:j+1 for j,i in enumerate(ordered)}
        elite=[r for r in oracle['candidates'] if r['step']==step and r['has_elite_descendant']]
        rho,p=spearmanr([-distances[a['id']] for a in states],[truth[(step,a['id'])]['intermediate_pic50'] for a in states])
        starts.append({'t':step/100,'matching_vs_intermediate_affinity_spearman':float(rho),'spearman_p_descriptive':float(p),
            'all_elite_states':[{'id':r['id'],'batch':r['batch'],'slot':r['slot'],'geometric_match_rank':position[r['id']],
                'geometric_distance':distances[r['id']],'quality_future_max_pic50':r['quality_future_max_pic50']} for r in elite]})
    losses=[]
    for step in [1,5,10,20,30]:
        replay=read_json(trial/f'blind/answers/continuation_{step:03d}.json')
        nodes=replay['nodes']
        for before,after in zip(nodes,nodes[1:]):
            bt,at=round(before['t']*100),round(after['t']*100)
            parent_ids={r['candidate_id'] for r in before['ranking']}
            prev_best=max((truth[(bt,i)]['quality_future_max_pic50'] for i in parent_ids if truth[(bt,i)]['quality_future_max_pic50'] is not None),default=None)
            if prev_best is None:continue
            chosen={r['candidate_id'] for r in after['ranking']}
            protos=selector.cards(trial/f'blind/prototypes_{at:03d}.zh-CN.md',True)
            states=[a for a in selector.cards(trial/f'blind/states_{at:03d}.zh-CN.md') if a.get('parent') in parent_ids]
            if not states:raise ValueError('Observed survivor vanished without a child')
            rows=[]
            for a in states:
                sc,p=min(((selector.score(a,p),p) for p in protos),key=lambda z:z[0])
                r=truth[(at,a['id'])]
                av,pv=selector.vec(a),selector.vec(p)
                scales=[.08 if i<5 else .12 if i<22 else .25 if i<27 else .18 if i<44 else .35 for i in range(len(av))]
                error=np.abs(np.array(av)-pv)/scales/len(av)
                rows.append({'id':a['id'],'parent_id':r['parent_id'],'batch':r['batch'],'slot':r['slot'],
                    'chosen':a['id'] in chosen,'distance':sc,'matched_prototype':p['id'],
                    'has_elite_descendant':r['has_elite_descendant'],'best_quality_future_pic50':r['quality_future_max_pic50'],
                    'current_Rg':a['cur_rg'],'forecast_Rg':a['fc_rg'],'forecast_moments':a['fc_mom'],
                    'current_group_error':float(error[:22].sum()),'forecast_group_error':float(error[22:44].sum()),
                    'RMS_group_error':float(error[44:].sum()),
                    'forecast_region00_mass':a['fc_regions'][0],'forecast_pair2':a['fc_pair2'],'forecast_pair7':a['fc_pair7']})
            new_best=max((r['best_quality_future_pic50'] for r in rows if r['chosen'] and r['best_quality_future_pic50'] is not None),default=None)
            if new_best is None or new_best<prev_best-1e-8:
                rows.sort(key=lambda r:(r['distance'],r['id']))
                losses.append({'start_t':step/100,'before_t':bt/100,'after_t':at/100,'best_before':prev_best,'best_after':new_best,
                    'loss_due_to_budgeted_geometric_pruning':True,'eligible_in_rank_order':rows})
    log=read_json(trial/'blind/answers/action_log.json')
    seals=[];violations=[]
    read_initial=[]
    for entry in log:
        if entry['action']=='read' and entry['file'].startswith('states_') and len(read_initial)<5:
            read_initial.append(int(entry['file'].split('_')[1].split('.')[0]))
        if entry['action']=='seal':
            f=trial/'blind/answers'/entry['file']
            ok=digest(f)==entry['sha256'];seals.append({'file':entry['file'],'unchanged':ok})
            if not ok:violations.append(entry['file'])
    if read_initial!=[1,5,10,20,30] or violations:raise ValueError('Initial ordering or sealed answers failed')
    write_json(output/'selector_audit.json',{'starts':starts,'pruning_losses':losses,'seal_checks':seals,
        'initial_read_order':read_initial,'selector_sha256':digest(path),
        'delivered_but_omitted':['Absolute receptor-frame global centroid','Signed local dx/dy/dz changes per 0.01','Signed changes and rates of pair kernels and regional density'],
        'used':['Current and forecast Rg/diagonal moments','Short/long pair kernels','Minimum pair distance','Three Gaussian regional masses/local means/shell kernels','Three unsigned RMS values'],
        'scale_issue':'47 component absolute distances use heterogeneous fixed tolerances by vector index, not dimension-specific evidence-derived scales; equal number of current/forecast fields does not imply equal effective weight',
        'labels_used_only_after_answer_seal':True})
    print('Audited pruning losses:',[(r['start_t'],r['before_t'],r['after_t']) for r in losses])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.trial,a.output)
