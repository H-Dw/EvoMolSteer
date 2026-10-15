"""Compact cross-condition counterevidence without treating trials as repeats."""
from collections import defaultdict


def family_evidence(state):
    baseline=state['reference']['metrics'];groups=defaultdict(list)
    for r in state['rounds']:
        if r.get('stage')!='from_zero' or not r.get('program') or not r.get('metrics'):continue
        p=r['program'];m=r['metrics'];key=(p['operator'],p.get('target_schedule','fixed'),p['controller'])
        groups[key].append({'round':r['round'],'factor':r['factor'],'region':r['region'],
            'n':m['n'],'mean_delta_vs_reference':m['mean_all_pic50']-baseline['mean_all_pic50'],
            'tail_delta_vs_reference':m['U_tau']-baseline['U_tau'],'source':r['round']+'/results.json',
            'local_outcomes':r.get('local_outcomes')})
    return {'families':[{'operator':key[0],'target_schedule':key[1],'controller':key[2],
        'n_conditions':len(rows),'n_different_regions':len({r['region'] for r in rows}),
        'n_lower_tail_than_reference':sum(r['tail_delta_vs_reference']<0 for r in rows),
        'trials':rows} for key,rows in sorted(groups.items())],
        'reference_program':state['reference']['program']['program_id'],
        'limits':'Heterogeneous interventions on a reused prior pool. These are not independent replicates or isolated operator effects; no aggregate p-values or causal attribution.'}
