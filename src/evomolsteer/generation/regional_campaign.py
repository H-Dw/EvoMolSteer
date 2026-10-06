"""Revised search after failed compressed target; full coordinates stay intact."""
import copy
import numpy as np
from .affinity_campaign import select_parent
from .persistent_campaign import apply_persistent_fields
from ..continuous.regional_design import REGIONAL_KEYS


def propose_regional(number,rows,programs,prior):
    parent=select_parent([r for r in rows if r['round']<=26],secondary=number>=17)
    if number>=27:
        winner=next(r for r in rows if r['round']==26)['frozen_winner']
        return dict(parent_round=winner,template_round=winner,action='frozen_validation',reason='Frozen discovery formula and dose; no heldout retuning.')
    common=dict(parent_round=parent['round'],global_parent_retained=True,time_ramp_power=0.)
    specs={
        12:dict(template_round=7,reward_view='endpoint_pointcloud',native_rms_ratio=.867),
        13:dict(template_round=None,regional=True),
        14:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_score_beta=6.),
        15:dict(template_round=13,regional=True,coordinate_region_weights=prior['region_weights']),
        16:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_neighbors=1,teacher_endpoint_temperature_A2=1.),
        17:dict(template_round=13,regional=True,coordinate_region_radius_A=3.),
        18:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_neighbors=8,teacher_endpoint_temperature_A2=16.),
        19:dict(template_round=13,regional=True,coordinate_region_radius_A=7.),
        21:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_score_beta=0.),
        22:dict(template_round=13,regional=True,region_salience_mode='observed_abs_effect'),
        24:dict(template_round=7,reward_view='endpoint_pointcloud',teacher_endpoint_temperature_A2=.5),
    }
    if number in specs:
        s=specs[number]
        return dict(common,**s,action='regional_coordinate_counterfactual' if s.get('regional') else 'coherent_coordinate_counterfactual',
            reason='After active compressed-target regression, compare intact coherent coordinates and learned spatial emphasis; retain global affinity winner.')
    if number in (20,23,25,26):
        p=programs[parent['round']];factor={20:.65,23:1.25,25:.8,26:1.1}[number]
        return dict(common,template_round=parent['round'],regional=p['reward_view']=='endpoint_regional_pointcloud',
            native_rms_ratio=min(3.,max(.1,p['native_rms_ratio']*factor)),action='parent_refinement',
            reason='Dose refinement of measured global affinity winner; regressions never overwrite the parent.')
    raise ValueError('Unregistered regional-profile round')


def apply_regional_fields(program,plan,prior):
    if plan['action']=='frozen_validation':return copy.deepcopy(program)
    p=apply_persistent_fields(program,plan)
    if p['reward_view']!='endpoint_regional_pointcloud':
        for key in REGIONAL_KEYS:
            if key not in ('mixture_temperature','pointcloud_delta_A'):p.pop(key,None)
        return p
    for key in REGIONAL_KEYS:
        if key in plan:p[key]=plan[key]
    if plan.get('region_salience_mode')=='observed_abs_effect':
        salience=np.ones((len(prior['times']),len(prior['region_weights'])))
        for r,w in enumerate(p['coordinate_region_weights']):
            if w:
                key=f'landmark_{r:02d}_soft_mass'
                if key not in prior['supported_node_functions']:raise ValueError('No supported empirical regional amplitude')
                salience[:,r]=np.clip(np.abs(prior['supported_node_functions'][key]['values_z']),.1,2.)
        p['coordinate_region_node_salience']=salience.tolist()
    p['target_definition']='coherent_elite_pointcloud_with_detached_regional_attention'
    return p
