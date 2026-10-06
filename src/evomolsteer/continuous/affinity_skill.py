"""Observable behavior checks for the affinity-first Agent profile."""
from pathlib import Path
from ..io import digest, read_json, write_json

def classify(gain, meaningful=.05, flat=.02):
    if gain >= meaningful: return 'meaningful_gain'
    if gain > flat: return 'promising_gain'
    if gain < -flat: return 'negative_affinity'
    return 'flat_response'

def affinity_choice(rows):
    supported=[r for r in rows if r['valid_rate_change'] >= -.10 and r.get('score_coverage',1.) >= .99]
    if not supported: raise ValueError('No sufficiently covered viable affinity candidate')
    return max(supported,key=lambda r:(r['all_head_change_vs_native'],r.get('unique_head_change_vs_native',-1e9)))['round']

def audit_behavior(skill, task, response, output=None):
    """Check actual decisions against numeric inputs, not text or hash echoes."""
    if response.get('skill_sha256') != digest(skill): raise ValueError('Agent did not bind the supplied skill')
    if response.get('input_sha256') != digest(task): raise ValueError('Agent response uses a different input')
    data=read_json(task)
    if response['primary_objective']!='predicted_affinity': raise ValueError('Historical objective leaked into current profile')
    if response['selected_round'] != affinity_choice(data['candidate_programs']): raise ValueError('Agent retained a physical-first decision')
    checks=[]
    for case in data['response_cases']:
        actual=next(r for r in response['response_actions'] if r['id']==case['id'])
        required={'flat_response':'dose_escalation','negative_affinity':'geometry_target_revision',
                  'promising_gain':'parent_refinement','meaningful_gain':'secondary_energy_assessment'}[classify(case['head_change'])]
        if required not in actual['interventions']: raise ValueError('Agent did not respond to the required diagnostic branch')
        checks.append({'case':case['id'],'required_intervention':required,'passed':True})
    if response.get('per_step_affinity_calls',-1)!=0 or response.get('affinity_head_gradient') is not False:
        raise ValueError('Agent introduced forbidden head calls or gradients')
    if response.get('feature_priority')!='3D_coordinates' or not response.get('coordinate_features'):
        raise ValueError('Agent did not implement the coordinate feature priority')
    result={'passed':True,'selected_round':response['selected_round'],'checks':checks,
            'skill_sha256':digest(skill),'input_sha256':digest(task),
            'behavior_scope':'Observed task decisions; not proof of a causal LLM advantage'}
    if output: write_json(output,result)
    return result
