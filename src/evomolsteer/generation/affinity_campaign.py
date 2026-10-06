"""Affinity-primary program search, separate from historical physical-first rules."""
import copy
from ..continuous.affinity_skill import classify

FAMILIES=('motif_mixture','affinity_landmark','affinity_direction','affinity_pointcloud')
def select_parent(rows,secondary=False):
    supported=[r for r in rows if r['valid_rate_change']>=-.10 and r.get('head_coverage',1.)>=.99]
    if not supported:raise ValueError('No viable covered program')
    best=max(supported,key=lambda r:r['all_head_change_vs_native'])
    if secondary and best['all_head_change_vs_native']>=.05:
        near=[r for r in supported if r['all_head_change_vs_native']>=best['all_head_change_vs_native']-.01]
        return max(near,key=lambda r:(min(r['negative_MMFF_relative_change'],r['negative_MMFF_p90_relative_change']),r['all_head_change_vs_native']))
    return best

def propose(number,rows,programs):
    if number==1:return {'parent_round':0,'reward_view':'motif_mixture','native_rms_ratio':.30,'action':'dose_escalation',
                         'reason':'Sixfold increase from frozen historical eta .05; test delivered dose before replacing geometry.'}
    if number>=27:
        parent=next(r for r in rows if r['round']==26).get('frozen_winner')
        if parent is None:parent=select_parent([r for r in rows if r['round']<=26],True)['round']
        return {'parent_round':parent,'action':'frozen_validation','reason':'Replay frozen discovery affinity winner; no validation/heldout tuning.'}
    parent=select_parent(rows,secondary=number>=17);p=programs[parent['round']];last=rows[-1]
    category=classify(last['all_head_change_vs_native']);family=p['reward_view'];explore=(number-2)%8
    # Flat outcomes first increase dose; negative outcomes revise target geometry.
    if category=='negative_affinity' or explore in (2,3,5):
        family=FAMILIES[(FAMILIES.index(last['reward_view'])+1)%len(FAMILIES)]
        if family=='motif_mixture':family='affinity_pointcloud'
        plan={'parent_round':parent['round'],'reward_view':family,'native_rms_ratio':max(.3,p['native_rms_ratio']),
              'action':'geometry_target_revision','reason':'Inspect stronger dose response, then test a distinct pure coordinate target/correspondence.'}
    else:
        factor=1.7 if category=='flat_response' or explore in (0,1) else (.65 if explore in (4,7) else 1.25)
        dose=min(3.,max(.1,p['native_rms_ratio']*factor))
        plan={'parent_round':parent['round'],'native_rms_ratio':dose,'action':'dose_escalation' if factor>1 else 'parent_refinement',
              'reason':'Affinity-first parent retained; measured dose/temporal localization sensitivity, energy assessed only after clear head gain.'}
        if dose==p['native_rms_ratio'] and category=='flat_response':plan.update(reward_view='affinity_pointcloud',action='geometry_target_revision')
    if number>=9 and plan.get('reward_view',family)=='affinity_pointcloud':
        plan.update(teacher_neighbors=(1,4,8)[number%3],teacher_score_beta=(0.,2.,6.)[number%3],teacher_endpoint_temperature_A2=(1.,4.,16.)[(number//3)%3])
    if number>=13 and number%4==0:plan['time_ramp_power']=1.
    if number>=17 and category=='meaningful_gain' and number%3==0:plan['action']='secondary_energy_assessment'
    return plan

def update_program(parent,plan,window,reference_sha):
    p=copy.deepcopy(parent)
    allowed={'reward_view','native_rms_ratio','teacher_neighbors','teacher_score_beta','teacher_endpoint_temperature_A2','time_ramp_power','geometry_block_weights'}
    p.update({k:v for k,v in plan.items() if k in allowed})
    p.update(window=list(window),reference_sha256=reference_sha,record_reward_response=True,
        objective_profile='affinity_primary_coordinate30',affinity_head_gradient=False,additional_per_step_affinity_calls=0)
    p['constraints']={'max_atom_step_A':.15,'max_cumulative_rms_A':6.,'severe_receptor_clash_A':.8,'backtrack_attempts':7}
    if p['reward_view']!='motif_mixture':
        p.update(teacher_neighbors=p.get('teacher_neighbors',4),teacher_score_beta=p.get('teacher_score_beta',2.),
            teacher_endpoint_temperature_A2=p.get('teacher_endpoint_temperature_A2',4.),geometry_block_weights=[.25,1.,1.,2.],
            target_definition='recorded_affinity_strata',family='coordinate_only_affinity_teacher')
    if p['reward_view']=='affinity_pointcloud':p['geometry_block_weights_role']='Monitoring only; actual reward uses matched full coordinate point clouds'
    return p
