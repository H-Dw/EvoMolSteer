"""Explore registered endpoint geometry while retaining the global affinity parent."""
from .affinity_campaign import select_parent
from ..continuous.affinity_skill import classify

FAMILIES=('endpoint_direction','endpoint_landmark','endpoint_pointcloud')


def propose_endpoint(number,rows,programs):
    global_parent=select_parent([r for r in rows if r['round']<=26],secondary=number>=17)
    if number>=27:
        winner=next(r for r in rows if r['round']==26)['frozen_winner']
        return {'parent_round':winner,'template_round':winner,'action':'frozen_validation',
            'reason':'Frozen global affinity-primary winner; no heldout-driven retuning.'}
    if number==5:
        return {'parent_round':global_parent['round'],'template_round':None,'action':'representation_intervention',
            'reason':'Literal Agent design: endpoint direction with actual FLOWR coordinate VJP, using the stronger endpoint geometry associations.'}
    endpoint=[r for r in rows if r['reward_view'] in FAMILIES]
    try:parent=select_parent(endpoint,secondary=number>=17) if endpoint else None
    except ValueError:parent=None
    # If a new family has poor yield, restart its declared lower-dose template;
    # keep that failed report and the global legacy winner, never stop exploration.
    template=parent['round'] if parent else 5;p=programs[template];last=rows[-1];category=classify(last['all_head_change_vs_native'])
    mode=(number-8)%6 if number>7 else 0;family=p['reward_view'];dose=p['native_rms_ratio']
    plan={'parent_round':global_parent['round'],'template_round':template,'action':'parent_refinement',
        'global_parent_retained':True,'reason':'Explore registered endpoint program; retain the measured global affinity parent for final selection.'}
    if endpoint and parent is None:
        plan.update(reward_view=FAMILIES[(FAMILIES.index(family)+1)%3],native_rms_ratio=.1,action='geometry_target_revision',
            reason='Retain failed-yield report and restart another coordinate family at the lower declared dose; global viable parent preserved.')
    elif number==7:
        plan.update(reward_view='endpoint_pointcloud',native_rms_ratio=.3,teacher_neighbors=4,teacher_score_beta=2.,teacher_endpoint_temperature_A2=4.,
            action='geometry_target_revision',reason='Complete the full endpoint point-cloud comparison at the declared baseline dose.')
    elif category=='negative_affinity' or mode==1:
        previous=last['reward_view'] if last['reward_view'] in FAMILIES else family
        plan.update(reward_view=FAMILIES[(FAMILIES.index(previous)+1)%3],native_rms_ratio=.3,
            action='geometry_target_revision',reason='Poor final affinity or planned representation comparison: revise the coordinate target, warm-start dose .3.')
    elif mode==3:
        plan.update(reward_view='endpoint_direction',geometry_block_weights=[0.,1.,2.,0.],native_rms_ratio=.3,
            action='geometry_target_revision',reason='Isolate supported shape/pair fields from landmark fields, including tiny floor-scaled occupancy terms.')
    elif mode==5:
        k=(number//6)%3
        plan.update(reward_view='endpoint_pointcloud',native_rms_ratio=.3,teacher_neighbors=(1,4,8)[k],
            teacher_score_beta=(0.,2.,6.)[k],teacher_endpoint_temperature_A2=(1.,4.,16.)[k],action='geometry_target_revision',
            reason='Test conditional teacher correspondence and affinity-weighted coordinate modes, without live head inference.')
    else:
        factor=1.7 if category=='flat_response' else (.7 if mode==2 else 1.25)
        plan.update(native_rms_ratio=min(3.,max(.1,dose*factor)),action='dose_escalation' if factor>1 else 'parent_refinement')
        if mode==4:plan['time_ramp_power']=1.
        if plan['native_rms_ratio']==dose and category=='flat_response':
            plan.update(reward_view=FAMILIES[(FAMILIES.index(family)+1)%3],native_rms_ratio=.3,action='geometry_target_revision')
    return plan
