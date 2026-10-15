"""Numerical identity and source/support checks shared by both roles."""
import copy
import hashlib
import json
import numpy as np
import jsonschema
from evomolsteer.final_guidance.reward import active_updates
from .reward import MotifReward


def identity(p,t):
    r=MotifReward(p,t)
    value={k:p[k] for k in ['operator','representation','window','steps','controller','assignment_element_cost','native_options']}
    value['target']=r.target.tolist();value['labels']=r.labels if p['assignment_element_cost'] else None
    value['dose']={k:v for k,v in p['dose'].items() if k!='pointcloud_delta_A' or p['operator']=='pointwise'}
    if p['operator']!='pointwise':value['band_A']=p['base_band_A']*p['band_multiplier']
    if p['operator']=='orientation':value.update(triplets=p['orientation_triplets'],orientation_band=p['orientation_band'])
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
    p['expected_active_steps']=active_updates(p['steps'],*p['window']);MotifReward(p,t)
    return p


def variants(parent,analysis,region,teacher,master):
    factor=analysis['factor'];p=bind_region(copy.deepcopy(parent),region,teacher)
    values={
        'region':[region['region_id']], 'operator':[x for x in ('pointwise','point_band','pocket_distance','orientation','pair_distance') if x!=parent['operator']],
        'dose':sorted(set([max(.025,parent['dose']['native_rms_ratio']*.5),min(2.,parent['dose']['native_rms_ratio']*1.5)])),
        'band_multiplier':sorted(set([max(.125,parent['band_multiplier']*.5),min(4.,parent['band_multiplier']*1.5)])),
        'controller':['bounded' if parent['controller']=='unit_rms' else 'unit_rms'],
        'matching':[x for x in [0.,.2,.5,1.] if x!=parent['assignment_element_cost']],
        'representation':['endpoint' if parent['representation']=='bridge' else 'bridge'],
        'window_start':[s for s in master['allowed_starts'] if s!=parent['window'][0] and s<parent['window'][1]],
        'gradient_floor':[.25,.5,1.,2.], 'curvature':[.5,1.,2.]
    }[factor]
    result={}
    for n,v in enumerate(values):
        q=copy.deepcopy(p)
        if factor=='operator':q['operator']=v
        elif factor=='dose':q['dose']['native_rms_ratio']=v
        elif factor=='matching':q['assignment_element_cost']=v
        elif factor=='window_start':q['window'][0]=v
        elif factor=='gradient_floor':q['gradient_floor_multiplier']=v
        elif factor=='curvature':q['dose']['pointcloud_delta_A']=v
        elif factor!='region':q[factor]=v
        if q['operator']!=analysis['observable_id']:continue
        try:validate_program(q,teacher,master['learning_window'])
        except ValueError:continue
        if factor=='band_multiplier' and q['operator']=='pointwise':continue
        if factor=='curvature' and q['operator']!='pointwise':continue
        if factor=='gradient_floor' and q['controller']!='bounded':continue
        result['V%02d'%n]=q
    return result
