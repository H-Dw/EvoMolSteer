import copy
import pytest
from jsonschema.exceptions import ValidationError
from evomolsteer.continuous.motif_agents import validate,DESIGNER,ANALYST


def bundle():
    kinds=('centroid_x','centroid_y','centroid_z','shell_2','shell_4','shell_6','pair_1_5','pair_3','pair_5')
    return {'window':[.1,.4],'regions':{'r':{}},'evidence':[{'evidence_id':'e','feature':'r::NOS_C::proposal_motif_pair_3'}],
        'features':{f'r::NOS_C::proposal_motif_{k}':{} for k in kinds}}


def design():
    return {'schema_version':'motif-agent-1.0','agent':'Designer','design_status':'exploratory',
        'architecture':'motif_mixture','regions':['r'],'channel':'NOS_C','window':[.1,.4],
        'native_rms_ratio':.1,'mixture_temperature':.5,'robust_delta':2.,'time_ramp_power':0.,
        'motif_components':'pair','rationale':'Exploratory relative-geometry ablation, not causal affinity direction',
        'evidence_ids':['e'],'limitations':['No causal claim']}


def test_strict_observable_window_and_evidence_contract():
    b=bundle();d=design();assert validate(d,DESIGNER,b)
    for field,value in [('window',[0,.5]),('regions',['unknown']),('evidence_ids',['fake']),('architecture','graph_lock')]:
        with pytest.raises((ValueError,ValidationError)):validate({**d,field:value},DESIGNER,b)
    b['features'].pop('r::NOS_C::proposal_motif_shell_2')
    with pytest.raises(ValueError,match='Missing controlled'):validate(d,DESIGNER,b)


def test_contrast_requires_declared_bound_and_no_arbitrary_code():
    d=design();d['architecture']='motif_contrast'
    with pytest.raises(ValueError):validate(d,DESIGNER,bundle())
    assert validate({**d,'contrast_bound_nats':2.},DESIGNER,bundle())
    with pytest.raises(ValidationError):validate({**d,'contrast_bound_nats':2.,'python':'arbitrary execution'},DESIGNER,bundle())
