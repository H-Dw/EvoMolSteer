"""Registered coordinate experiments; retain the global affinity winner on failure.

Sparse stage functions are grounded in observed node effects. Dose, coordinate
correspondence and temporal weighting are separate interventions. Heldout seeds
cannot change the program selected at the end of discovery.
"""
import copy
from .affinity_campaign import select_parent
from ..continuous.affinity_skill import classify

SPARSE_KEYS=('geometry_feature_weights','geometry_direction_mode',
             'geometry_direction_functions','geometry_direction_times',
             'coordinate_prior_sha256','geometry_weight_normalization','geometry_curve_fidelity_audit')


def propose_sparse(number,rows,programs):
    parent=select_parent([r for r in rows if r['round']<=26],secondary=number>=17)
    if number>=27:
        winner=next(r for r in rows if r['round']==26)['frozen_winner']
        return dict(parent_round=winner,template_round=winner,action='frozen_validation',
                    reason='Replay frozen global affinity winner; heldout outcomes cannot alter its formula or dose.')
    plan=dict(parent_round=parent['round'],global_parent_retained=True)
    if number==6:
        return dict(plan,template_round=None,sparse=True,action='node_fidelity_intervention',
                    reason='Literal retested Agent response: supported endpoint coordinate fields; reject contradicted polynomial curves; match the validated VJP baseline dose.')
    # Repeatable counterfactuals compare sparse stage effects, dense coordinate
    # targets, local regional contrasts and conditional point-cloud attraction.
    mode=(number-7)%8;cycle=(number-7)//8;comparison_dose=(.3,.51,.867)[cycle%3]
    if mode in (0,4,7):
        k=(number-7)//8
        return dict(plan,template_round=5,sparse=False,reward_view='endpoint_pointcloud' if mode!=4 else 'endpoint_landmark',
                    native_rms_ratio=.3 if mode==0 else .51,
                    teacher_neighbors=(4,1,8)[k%3],teacher_score_beta=(2.,0.,6.)[k%3],
                    teacher_endpoint_temperature_A2=(4.,1.,16.)[k%3],time_ramp_power=0.,
                    action='geometry_target_revision',reason='Change coordinate target/correspondence using recorded teachers, without live affinity inference; keep global parent.')
    sparse_rows=[r for r in rows if 'geometry_feature_weights' in programs[r['round']]]
    try:chosen=select_parent(sparse_rows,secondary=number>=17)
    except ValueError:chosen=next(r for r in rows if r['round']==6)
    template=chosen['round'];p=programs[template];last=rows[-1]
    response=classify(last['all_head_change_vs_native'])
    plan.update(template_round=template,sparse=True,reward_view='endpoint_direction',action='supported_coordinate_refinement')
    if mode==1:
        plan.update(direction_mode='node_contrast',geometry_block_weights=[0.,1.,2.,1.],native_rms_ratio=comparison_dose,time_ramp_power=0.,
                    reason='Compare mean batch-specific node discriminants to empirical-effect interpolation at matched baseline dose.')
    elif mode==3:
        plan.update(direction_mode='linear_node_effect',geometry_block_weights=[0.,1.,2.,0.],native_rms_ratio=comparison_dose,time_ramp_power=0.,
                    reason='Isolate supported shape and pair-distance fields; retain compact coordinate regional evidence as a comparator.')
    elif mode==5:
        plan.update(direction_mode='linear_node_effect',geometry_block_weights=[0.,0.,0.,1.],native_rms_ratio=comparison_dose,time_ramp_power=0.,
                    reason='Isolate supported receptor-local coordinate regions; this tests their affinity contribution rather than atom-type changes.')
    elif mode==6:
        plan.update(direction_mode='linear_node_effect',geometry_block_weights=[0.,1.,2.,1.],native_rms_ratio=comparison_dose,time_ramp_power=1.,
                    reason='Test continuous positive temporal weighting over the entire learned window; no guidance after its end.')
    else:
        # A negative result triggers target revision, never automatic escalation.
        if response=='negative_affinity':
            plan.update(direction_mode='linear_node_effect' if p['geometry_direction_mode']=='node_contrast' else 'node_contrast',
                        native_rms_ratio=.3,geometry_block_weights=[0.,1.,2.,1.],time_ramp_power=0.,action='geometry_target_revision',
                        reason='Negative affinity requires changing the coordinate discriminant, not increasing its dose.')
        else:
            factor=1.7 if response=='flat_response' else 1.25
            plan.update(native_rms_ratio=min(3.,max(.1,p['native_rms_ratio']*factor)),action='dose_escalation',
                        reason='Test delivered-dose sensitivity of the measured best supported coordinate program; preserve earlier winner on regression.')
    return plan


def apply_sparse_fields(program,plan,prior):
    p=copy.deepcopy(program)
    if plan.get('action')=='frozen_validation':return p
    if not plan.get('sparse',False):
        for key in SPARSE_KEYS:p.pop(key,None)
        return p
    mode=plan.get('direction_mode',p['geometry_direction_mode'])
    if mode not in ('node_contrast','linear_node_effect'):raise ValueError('Only empirically faithful registered stage functions')
    p['geometry_direction_mode']=mode
    p['geometry_direction_functions']=({f:prior['node_effect_functions'][f]
        for f,w in zip(prior['features'],p['geometry_feature_weights']) if w>0} if mode=='linear_node_effect' else {})
    p['geometry_direction_times']=prior['times']
    return p
