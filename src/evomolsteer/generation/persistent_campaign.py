"""Coordinate target counterfactuals with global-parent retention and frozen checks."""
import copy
from .affinity_campaign import select_parent
from .sparse_campaign import SPARSE_KEYS
from ..continuous.persistent_design import HISTORY_KEYS

PERSISTENT_KEYS=HISTORY_KEYS+('history_prior_sha256','history_prior_relative_path',
                          'persistent_designer_response_sha256','persistent_behavior_audit_sha256')


def propose_persistent(number,rows,programs):
    parent=select_parent([r for r in rows if r['round']<=26],secondary=number>=17)
    if number>=27:
        winner=next(r for r in rows if r['round']==26)['frozen_winner']
        return dict(parent_round=winner,template_round=winner,action='frozen_validation',
                    reason='Frozen discovery program; no heldout-dependent changes to its formula, priors or dose.')
    common=dict(parent_round=parent['round'],global_parent_retained=True,time_ramp_power=0.)
    if number==9:
        return dict(common,template_round=None,persistent=True,action='bounded_coordinate_target_revision',
                    reason='Literal new Agent design: bounded supported elite-prototype attraction at matched eta .3, history0.')
    # Counterfactual pairs separate delivered strength, full coordinate detail,
    # local supported feature geometry and forecast-change persistence.
    specs={
        10:dict(template_round=7,reward_view='endpoint_pointcloud',native_rms_ratio=.51),
        11:dict(template_round=9,persistent=True,history_strength=.25),
        12:dict(template_round=7,reward_view='endpoint_pointcloud',native_rms_ratio=.867),
        13:dict(template_round=9,persistent=True,history_strength=1.),
        14:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_score_beta=6.),
        15:dict(template_round=9,persistent=True,history_strength=4.),
        16:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_neighbors=1,teacher_endpoint_temperature_A2=1.),
        17:dict(template_round=9,persistent=True,history_strength=.25,geometry_block_weights=[0.,0.,0.,1.]),
        18:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_neighbors=8,teacher_endpoint_temperature_A2=16.),
        19:dict(template_round=9,persistent=True,geometry_block_weights=[0.,1.,2.,0.]),
        21:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_score_beta=0.),
        22:dict(template_round=9,persistent=True,geometry_salience_mode='constant',mixture_temperature=1.5),
        24:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_endpoint_temperature_A2=.5),
    }
    if number in specs:
        s=specs[number];kind='supported_lineage_increment' if s.get('history_strength',0)>0 else 'coordinate_target_or_dose_counterfactual'
        return dict(common,**s,action=kind,reason='Registered '+kind+' experiment; preserve the measured global winner and inspect paired window/final response.')
    if number in (20,23,25,26):
        p=programs[parent['round']];factor={20:.65,23:1.25,25:.8,26:1.1}[number]
        return dict(common,template_round=parent['round'],persistent=p['reward_view']=='endpoint_supported_attractor',
                    native_rms_ratio=min(3.,max(.1,p['native_rms_ratio']*factor)),
                    action='parent_refinement',reason='Refine the measured best affinity program with a distinct dose; retain it if the experiment regresses.')
    raise ValueError('Unregistered new persistent-profile round')


def apply_persistent_fields(program,plan):
    p=copy.deepcopy(program)
    if plan['action']=='frozen_validation':return p
    if p['reward_view']!='endpoint_supported_attractor':
        for key in PERSISTENT_KEYS:p.pop(key,None)
        for key in SPARSE_KEYS:p.pop(key,None)
        return p
    for key in HISTORY_KEYS:
        if key in plan:p[key]=plan[key]
    p['target_definition']='bounded_supported_elite_prototypes_and_lineage_increments'
    return p
