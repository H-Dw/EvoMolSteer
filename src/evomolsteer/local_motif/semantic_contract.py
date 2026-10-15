"""Numerical identity and source/support checks shared by both roles."""
import copy
import ast
import inspect
import textwrap
import hashlib
import json
import numpy as np
import jsonschema
from evomolsteer.final_guidance.reward import active_updates
from .reward import MotifReward,OPERATORS


def selected_branch_signature(source,operator):
    tree=ast.parse(textwrap.dedent(source));fn=tree.body[0]
    for i,node in enumerate(fn.body):
        if not isinstance(node,ast.If) or not isinstance(node.test,ast.Compare):continue
        if not isinstance(node.test.left,ast.Name) or node.test.left.id!='operator':continue
        current=node;selected=None
        while isinstance(current,ast.If):
            if current.test.comparators[0].value==operator:selected=current.body;break
            if len(current.orelse)==1 and isinstance(current.orelse[0],ast.If):current=current.orelse[0]
            else:selected=current.orelse;break
        fn.body[i:i+1]=selected;break
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()


def numerical_code_identity(p):
    from .reward import MotifControl
    from evomolsteer.final_guidance.native import Control,instrument,instrument_coord_telemetry
    return {'selected_reward':selected_branch_signature(inspect.getsource(MotifReward.evaluate),p['operator']),
        **{f.__qualname__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in
           [MotifControl.forward,Control.after,instrument,instrument_coord_telemetry]}}


def identity(p,t):
    r=MotifReward(p,t)
    value={k:p[k] for k in ['operator','representation','window','steps','controller','assignment_element_cost','native_options']}
    value['target']=r.target.tolist();value['labels']=r.labels if p['assignment_element_cost'] else None
    value['time_schedule']=p.get('time_schedule','constant')
    value['target_schedule']=p.get('target_schedule','fixed')
    if value['target_schedule']=='historical_path':
        value['path']=t['historical_path'];value['lookahead_fraction']=p.get('lookahead_fraction',.2)
    value['dose']={k:v for k,v in p['dose'].items() if k!='pointcloud_delta_A' or p['operator']=='pointwise'}
    if p['operator'] in ('point_band','pocket_distance','pair_distance'):value['band_A']=p['base_band_A']*p['band_multiplier']
    if p['operator']=='orientation':value.update(triplets=p['orientation_triplets'],orientation_band=p['orientation_band'])
    if p['operator']=='local_moments':value['moment_shape_weight']=p.get('moment_shape_weight',1.)
    if p['operator']=='pocket_distance':value['pocket_sites']=p['pocket_sites']
    if p['controller']=='bounded':value['gradient_floor_multiplier']=p['gradient_floor_multiplier']
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def references(answer,bundle,region=None):
    allowed=set(bundle['evidence'])
    for key in ['evidence_refs','counterevidence_refs']:
        for ref in answer[key]:
            if ref not in allowed:raise jsonschema.ValidationError('Unknown evidence ID',validator='enum',validator_value=sorted(allowed),instance=ref)
    if region and not any(r.startswith(region+'/') for r in answer['evidence_refs']):
        raise jsonschema.ValidationError('Selected region lacks a queried evidence reference',validator='region_binding',validator_value=region)


def bind_region(p,r,t):
    p.update(candidate_id=t['candidate_id'],template_id=r['template_id'],region_id=r['region_id'],atom_slots=r['atom_slots'],
        pocket_sites=r['pocket_sites'],orientation_triplets=r['orientation_triplets'],base_band_A=r['base_band_A'],teacher_time=t['time'])
    return p


def validate_program(p,t,window):
    if not window[0]<=p['window'][0]<p['window'][1]<=window[1]:raise ValueError('Control outside learned window')
    if not .025<=p['dose']['native_rms_ratio']<=2:raise ValueError('Unregistered dose')
    if not 0<p['band_multiplier']<=4 or p['assignment_element_cost']<0:raise ValueError('Invalid geometry scale')
    if p.get('target_schedule','fixed')=='historical_path':
        if p['representation']!='bridge' or 'historical_path' not in t:raise ValueError('Missing compatible path target')
        if not 0<p.get('lookahead_fraction',.2)<=1:raise ValueError('Invalid path lookahead fraction')
    p['expected_active_steps']=active_updates(p['steps'],*p['window']);MotifReward(p,t)
    return p


def variants(parent,analysis,region,teacher,master):
    factor=analysis['factor'];p=bind_region(copy.deepcopy(parent),region,teacher)
    values={
        'region':[region['region_id']], 'operator':[x for x in OPERATORS if x!=parent['operator']],
        'dose.native_rms_ratio':sorted(set([max(.025,parent['dose']['native_rms_ratio']*.5),min(2.,parent['dose']['native_rms_ratio']*1.5)])),
        'band_multiplier':sorted(set([max(.125,parent['band_multiplier']*.5),min(4.,parent['band_multiplier']*1.5)])),
        'controller':['bounded' if parent['controller']=='unit_rms' else 'unit_rms'],
        'assignment_element_cost':[x for x in [0.,.2,.5,1.] if x!=parent['assignment_element_cost']],
        'representation':['endpoint' if parent['representation']=='bridge' else 'bridge'],
        'window_start':[s for s in master['allowed_starts'] if s!=parent['window'][0] and s<parent['window'][1]],
        'gradient_floor_multiplier':[.25,.5,1.,2.], 'dose.pointcloud_delta_A':[.5,1.,2.],
        'orientation_band':[max(.005,parent['orientation_band']*.5),min(.9,parent['orientation_band']*1.5)],
        'moment_shape_weight':[.25,.5,1.,2.,4.],
        'time_schedule':[s for s in ('constant','early','late') if s!=parent.get('time_schedule','constant')]
        ,'target_schedule':[s for s in ('fixed','historical_path') if s!=parent.get('target_schedule','fixed')]
    }[factor]
    result={}
    for n,v in enumerate(values):
        q=copy.deepcopy(p)
        if factor=='operator':q['operator']=v
        elif factor=='dose.native_rms_ratio':q['dose']['native_rms_ratio']=v
        elif factor=='assignment_element_cost':q['assignment_element_cost']=v
        elif factor=='window_start':q['window'][0]=v
        elif factor=='gradient_floor_multiplier':q['gradient_floor_multiplier']=v
        elif factor=='dose.pointcloud_delta_A':q['dose']['pointcloud_delta_A']=v
        elif factor!='region':q[factor]=v
        if q['operator']!=analysis['observable_id']:continue
        try:validate_program(q,teacher,master['learning_window'])
        except ValueError:continue
        if factor=='band_multiplier' and q['operator'] not in ('point_band','pocket_distance','pair_distance'):continue
        if factor=='dose.pointcloud_delta_A' and q['operator']!='pointwise':continue
        if factor=='gradient_floor_multiplier' and q['controller']!='bounded':continue
        if factor=='orientation_band' and q['operator']!='orientation':continue
        if factor=='moment_shape_weight' and q['operator']!='local_moments':continue
        result['V%02d'%n]=q
    return result
