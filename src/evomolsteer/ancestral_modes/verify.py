"""Deterministic evidence that the bound local controller actually executed."""
from pathlib import Path
import numpy as np
from evomolsteer.io import read_json,write_json,digest
from .reward import horizon,temporal_gain


def observe(folder,output=None):
    folder=Path(folder);audit=read_json(folder/'guidance_audit.json');p=audit['program']
    active=[r for r in audit['steps'] if r['active']]
    checks={'configured_active_steps':audit['actual_active_steps']==p['expected_active_steps'],
        'completed_to_terminal':audit['actual_steps']==p['steps'],
        'no_smc':audit['no_smc'],'no_affinity_gradient':audit['no_affinity_head_gradient'],
        'no_extra_generator_forwards':all(r.get('additional_generation_forwards',0)==0 for r in active),
        'no_extra_affinity_forwards':all(r.get('additional_affinity_calls',0)==0 for r in active),
        'actual_clocks':all(np.allclose(r['actual_target_clocks'],horizon(p,r['time']),atol=2e-6) for r in active),
        'single_local_support':all(all(int(k)==len(p['mode_specs'][int(m)]['atom_slots']) for k,m in zip(r['direct_support_per_seed'],r['mode_indices'])) for r in active)}
    if p['routing']=='static':checks['static_assignment']=all(r['mode_indices']==[i%len(p['mode_specs']) for i in range(len(r['mode_indices']))] for r in active)
    if not p['attainment_response']:
        checks['post_normalization_schedule']=all(np.allclose(r['dose_gain_per_seed'],temporal_gain(p,r['time'],1/p['steps'])*
            (np.asarray(r['historical_contrast_gain_per_seed']) if p.get('advantage_evidence_gain',False) else 1.),atol=2e-6) for r in active)
    if p['routing'] in ('response','persistent_response'):
        checks['actual_routing_probes']=sum(r.get('routing_VJPs',0) for r in active)>=len(p['mode_specs'])
        checks['response_scores_delivered']=all(r.get('routing_response_scores') is not None for r in active)
    component={}
    for name in ('typed','charge'):
        rows=[r for r in active if name+'_gradient_rms_per_seed' in r]
        component[name]={'enabled_weight':p[name+'_weight'],'measured_probe_steps':len(rows),
            'max_coordinate_gradient_rms':max((max(r[name+'_gradient_rms_per_seed']) for r in rows),default=None),
            'mean_fraction_of_total_gradient':float(np.mean([v for r in rows for v in r[name+'_gradient_fraction_of_total']])) if rows else None,
            'mean_cosine_with_total_gradient':float(np.mean([v for r in rows for v in r[name+'_gradient_cosine_with_total']])) if rows else None}
        if p[name+'_weight']>0:checks[name+'_component_measured']=len(rows)>0
    result={'source_sha256':digest(folder/'guidance_audit.json'),'program_id':p['program_id'],
        'checks':checks,'passed':all(checks.values()),'routing_VJPs':sum(r.get('routing_VJPs',0) for r in active),
        'mode_switches':sum(r.get('mode_switches',0) for r in active),'active_steps':len(active),
        'nonzero_steps':sum(max(r.get('injected_rms_A_per_seed',[0]))>0 for r in active),
        'components':component,
        'matching_support_sizes':[len(s['atom_slots']) for s in p['mode_specs']],
        'differentiated_teacher_support_sizes':[len({i for trip in s['orientation_triplets'] for i in trip})
            if p['operator']=='orientation' and not(p['typed_weight'] or p['charge_weight']) else len(s['atom_slots']) for s in p['mode_specs']],
        'direct_mask_semantics':'Legacy direct mask is declared matching support, an upper bound. Pure orientation differentiates only triplet slots. Real generator VJP may affect other student atoms.',
        'limits':'Confirms instruction-to-function execution and measured response, not causal scientific benefit of a prompt module.'}
    if output:write_json(Path(output),result)
    if not result['passed']:raise ValueError('Controller execution differs from binding: '+str(checks))
    return result
